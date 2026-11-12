"""Evaluate selections under a frozen protocol; optionally run SeDPO alone."""
import argparse
import datetime as dt
import platform
import sys
from pathlib import Path
from common import LETTERS, digest, file_sha, load_protocol, messages, metrics, paired, validate_selection, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--task', choices=['arc_c', 'openbookqa'], required=True)
    parser.add_argument('--sedpo-selection', type=Path, required=True)
    parser.add_argument('--coopes-selection', type=Path)
    parser.add_argument('--sedpo-only', action='store_true')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.sedpo_only and args.coopes_selection:
        parser.error('SeDPO-only execution must not load CoopES selections')
    if not args.sedpo_only and args.coopes_selection is None:
        parser.error('Supply --coopes-selection for a paired run, or use --sedpo-only')
    if args.out.exists() or args.out.with_suffix('.complete.json').exists():
        raise FileExistsError('Do not overwrite an existing paired run')
    protocol, data, protocol_sha = load_protocol(args.protocol, args.task)
    selections, sources = {}, {}
    inputs = [('sedpo', args.sedpo_selection)]
    if not args.sedpo_only:
        inputs.append(('coopes', args.coopes_selection))
    for method, path in inputs:
        source, selections[method] = validate_selection(path, protocol_sha=protocol_sha, task=args.task,
                                                       pool=data['pool'], queries=data['queries'])
        if source['method'] != method:
            raise ValueError(f'{path} is not the requested {method} selection')
        sources[method] = {'path': str(path.resolve()), 'sha256': file_sha(path), 'selector_manifest': source['selector_manifest']}
    state = {'schema_version': 1, 'status': 'validated', 'task': args.task,
             'protocol_sha256': protocol_sha, 'protocol': protocol,
             'selection_sources': sources, 'methods': {method: {'records': []} for method in selections},
             'execution_scope': 'sedpo_only' if args.sedpo_only else 'paired_rerun',
             'manifest': {'evaluator_sha256': file_sha(__file__), 'common_sha256': file_sha(Path(__file__).with_name('common.py')),
                          'python': sys.version, 'platform': platform.platform()},
             'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write(args.out, state)
    if args.validate_only:
        print('Selections validated. No GPU evaluation or comparison result has been produced.')
        return
    try:
        import random
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor
        setting = protocol['evaluation']
        random.seed(setting['seed'])
        np.random.seed(setting['seed'])
        torch.manual_seed(setting['seed'])
        torch.cuda.manual_seed_all(setting['seed'])
        root = args.model.resolve(strict=True)
        files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in ['.json', '.jinja', '.model', '.txt', '.safetensors', '.bin'])
        state['manifest']['model_files'] = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': file_sha(p)} for p in files]
        state['manifest'].update({'model_path': str(root), 'torch': torch.__version__, 'transformers': transformers.__version__,
                                  'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                                  'reader_objects': 1, 'methods_in_this_process': list(selections),
                                  'attention_implementation': setting['attention_implementation']})
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        if not tokenizer.chat_template:
            raise ValueError('An explicit chat template is required')
        token_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(row) != 1 for row in token_ids) or len({row[0] for row in token_ids}) != 4:
            raise ValueError('Need one distinct canonical token per answer letter')
        allowed = [row[0] for row in token_ids]
        state['manifest']['chat_template_sha256'] = digest(tokenizer.chat_template)
        state['manifest']['answer_tokens'] = dict(zip(LETTERS, allowed))
        model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, torch_dtype=torch.float16,
                                                    device_map={'': 0}, attn_implementation=setting['attention_implementation']).eval()
        generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=1,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                       bos_token_id=tokenizer.bos_token_id, repetition_penalty=1.0, use_cache=True)
        state['manifest']['generation_config'] = generation.to_dict()
        state['status'] = 'running'
        write(args.out, state)

        class LetterConstraint(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if not bool(torch.isfinite(scores).all().item()):
                    raise FloatingPointError('Non-finite scores; refusing to record an A fallback')
                mask = torch.full_like(scores, float('-inf'))
                mask[:, allowed] = scores[:, allowed]
                return mask

        pool, gold = ({row['id']: row for row in data[role]} for role in ['pool', 'gold'])
        # Query first, then method: both methods use the same loaded object,
        # renderer, tokenizer, special-token handling, batch size and decoder.
        for position, query in enumerate(data['queries']):
            for method in selections:
                demo_ids = selections[method][query['id']]
                text = tokenizer.apply_chat_template(messages(query, demo_ids, pool, setting['system']),
                                                     tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(text, return_tensors='pt', add_special_tokens=False, truncation=False)
                length = encoded['input_ids'].shape[1]
                if length > setting['max_input_tokens']:
                    raise ValueError('Overlong input; neither method is silently truncated')
                with torch.inference_mode():
                    result = model.generate(**encoded.to('cuda'), generation_config=generation,
                                            logits_processor=[LetterConstraint()], return_dict_in_generate=True,
                                            output_logits=True, remove_invalid_values=False)
                if len(result.logits) != 1 or not bool(torch.isfinite(result.logits[0]).all().item()):
                    raise FloatingPointError('Non-finite raw logits')
                generated = result.sequences[0, length:].tolist()
                if len(generated) != 1 or generated[0] not in allowed:
                    raise ValueError('Invalid answer token')
                state['methods'][method]['records'].append({
                    'id': query['id'], 'gold': gold[query['id']]['gold'], 'pred': allowed.index(generated[0]),
                    'demo_ids': demo_ids, 'prompt_sha256': digest(text), 'input_tokens': length,
                    'generated_token_ids': generated, 'raw_output': tokenizer.decode(generated),
                    'letter_logits': result.logits[0][0, allowed].float().cpu().tolist()})
            if position % 10 == 0 or position == len(data['queries']) - 1:
                write(args.out, state)
                print(f'Queries: {position + 1}/500; methods={list(selections)}', flush=True)
        for method, group in state['methods'].items():
            if len(group['records']) != 500 or len({row['id'] for row in group['records']}) != 500:
                raise ValueError('Incomplete/duplicate paired predictions')
            group['metrics'] = metrics(group['records'])
        if args.sedpo_only:
            state['comparison_scope'] = 'SeDPO only; published CoopES scores are external, unpaired references; original test IDs and decoding are unknown. No method-superiority proof or paired significance test.'
        else:
            state['paired_accuracy_test'] = paired(state['methods']['sedpo']['records'], state['methods']['coopes']['records'])
        state['status'] = 'complete'
        state['completed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(args.out, state)
        write(args.out.with_suffix('.complete.json'), {'status': 'complete', 'result_sha256': file_sha(args.out),
                                                      'n_per_method': 500, 'protocol_sha256': protocol_sha})
        print({name: group['metrics'] for name, group in state['methods'].items()})
        print(state.get('paired_accuracy_test', state.get('comparison_scope')))
    except BaseException as error:
        state['status'] = 'failed'
        state['error'] = f'{type(error).__name__}: {error}'
        write(args.out, state)
        raise


if __name__ == '__main__':
    main()
