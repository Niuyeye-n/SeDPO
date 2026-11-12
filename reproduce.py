#!/usr/bin/env python3
"""Portable entry point for the four SeDPO rows; standard-library CLI/verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'repro'))
from common import file_sha, read, write, load_protocol, metrics, messages, validate_selection, content_key

BUNDLE = ROOT / 'results/tablei_20261111'
JOBS = {
    'qwen_arc': ('qwen', 'arc_c', 'close_qa', 9152, 86, 1e-5),
    'qwen_obqa': ('qwen', 'openbookqa', 'obqa_only', 11567, 108, 3e-5),
    'llama_arc': ('llama', 'arc_c', 'close_qa', 9355, 88, 1e-5),
    'llama_obqa': ('llama', 'openbookqa', 'obqa_only', 11366, 106, 3e-5),
}


def verify(bundle=BUNDLE):
    """Independently check archived results, not historical machine absolute paths."""
    protocol_path = bundle / 'protocol_sedpo_only_sdpa.json'
    checked = []
    for tag, (_, task, _, _, _, _) in JOBS.items():
        protocol, data, digest = load_protocol(protocol_path, task)
        directory = bundle / 'table_runs' / tag
        result_path = directory / 'result.json'
        state, marker = read(result_path), read(directory / 'result.complete.json')
        if marker['status'] != 'complete' or state['status'] != 'complete':
            raise ValueError(f'{tag}: not complete')
        if file_sha(result_path) != marker['result_sha256']:
            raise ValueError(f'{tag}: completion checksum mismatch')
        if digest != state['protocol_sha256'] or digest != marker['protocol_sha256'] or state['task'] != task:
            raise ValueError(f'{tag}: protocol/task mismatch')
        if state['execution_scope'] != 'sedpo_only' or set(state['methods']) != {'sedpo'}:
            raise ValueError('Only SeDPO final results are accepted')
        _, selection = validate_selection(directory/'selection.json', protocol_sha=digest, task=task,
                                          pool=data['pool'], queries=data['queries'])
        records = state['methods']['sedpo']['records']
        gold = {r['id']: r['gold'] for r in data['gold']}
        if len(records) != 500 or len({r['id'] for r in records}) != 500 or {r['id'] for r in records} != set(gold):
            raise ValueError(f'{tag}: expected 500 unique frozen questions')
        for r in records:
            if r['gold'] != gold[r['id']] or r['demo_ids'] != selection[r['id']]:
                raise ValueError(f'{tag}: gold or selected demo mismatch')
            expected = state['manifest']['answer_tokens']['ABCD'[r['pred']]]
            if r['generated_token_ids'] != [expected] or r['raw_output'] != 'ABCD'[r['pred']]:
                raise ValueError(f'{tag}: invalid output token')
            import math
            if len(r['letter_logits']) != 4 or not all(math.isfinite(x) for x in r['letter_logits']):
                raise ValueError(f'{tag}: non-finite logits')
        measured = metrics(records)
        if measured != state['methods']['sedpo']['metrics']:
            raise ValueError(f'{tag}: metrics differ from recomputed predictions')
        checked.append({'job': tag, **measured})
    return checked


def source_check():
    for relative, expected in read(ROOT/'repro/runtime_source_sha256.json').items():
        if file_sha(ROOT/'src'/relative) != expected:
            raise ValueError(f'Vendored runtime source changed: {relative}')


def model_check(model, reader, allow_different=False):
    expected = read(BUNDLE/'table_runs'/('qwen_arc' if reader=='qwen' else 'llama_arc')/'result.json')['manifest']
    differences = [f['name'] for f in expected['model_files']
                   if not (model/f['name']).is_file() or file_sha(model/f['name']) != f['sha256']]
    if differences and not allow_different:
        raise ValueError('Reader files differ from original: '+', '.join(differences)+
                         '. Supply exact model files, or explicitly use --allow-different-model for a NEW run.')
    return {'original_model_files_match': not differences, 'different_or_missing_files': differences}


def job_paths(work, tag):
    reader, task, cluster, _, _, _ = JOBS[tag]
    base=work/tag
    return {'base':base, 'scored':base/'data/scored', 'pool':base/'data/prompt_pool',
            'random':base/'find_random', 'train':base/'training', 'evaluation':base/'evaluation',
            'cache':work/'cache', 'cluster':cluster, 'task':task, 'reader':reader}


def check_training_inputs(base,tag,allow_different=False):
    reader,task,cluster,*_=JOBS[tag]
    known=read(ROOT/'repro/training_input_sha256.json')
    names=[f'{reader}/scored/{cluster}/{task}_scored_{split}.json' for split in ['train','valid']]
    names+=[f'{reader}/prompt_pool/{cluster}/{task}_prompts.json']
    changed=[name for name in names if not (base/name).is_file() or file_sha(base/name)!=known[name]]
    if changed and not allow_different:raise ValueError('Training input hash mismatch: '+', '.join(changed))
    groups=[read(base/name) for name in names[:2]]
    ids=[{r['id'] for r in rows} for rows in groups]
    contents=[{content_key({'question':r.get('question',r.get('question_stem')),'options':r['choices']['text']}) for r in rows} for rows in groups]
    if ids[0]&ids[1] or contents[0]&contents[1]:raise ValueError('Training/validation queries overlap')
    if [len(x) for x in groups]!=[810,90] or [len(x) for x in ids]!=[270,30]:
        raise ValueError('Expected 810/90 entries from 270/30 unique queries')
    return {'files_match':not changed,'different_files':changed}


def commands(args, tag):
    p=job_paths(args.work_dir.resolve(),tag)
    reader,task,cluster,_,_,lr=JOBS[tag]
    model=getattr(args,reader+'_model')
    if model is None:raise ValueError(f'--{reader}-model is required for {tag}')
    model=str(model.resolve())
    pool=p['pool']/cluster/f'{task}_prompts.json'
    scored=p['scored']/cluster
    python=sys.executable
    def py(script,*options):return [python,str(ROOT/'src'/script),*map(str,options)]
    stages=[]
    if args.mode=='rescore':
        for step in [1,2,3]:
            random_file=p['random']/f'{task}_step{step}.json'
            opts=[f'output_path={random_file}',f'task_name={task}','+ds_size=10000','L=50',
                  f'prompt_pool_path={pool}',f'cache_dir={p["cache"]}',f'hydra.run.dir={p["base"]}/hydra/finder{step}']
            if step==1:opts+=['+n_score=300']
            else:opts += [f'train_path={scored}/{task}_scored_train_step{step-1}.json',
                           f'valid_path={scored}/{task}_scored_valid_step{step-1}.json']
            script=f'scripts/find_random_step{step}.py'
            stages.append((f'find{step}',py(script,*opts)))
            stages.append((f'score{step}',py('scripts/scorer.py',f'example_file={random_file}',
                f'output_train_file={scored}/{task}_scored_train_step{step}.json',
                f'output_valid_file={scored}/{task}_scored_valid_step{step}.json',
                'batch_size=5',f'task_name={task}',f'model_name={model}',f'prompt_pool_path={pool}',
                f'cache_dir={p["cache"]}',f'hydra.run.dir={p["base"]}/hydra/scorer{step}')))
        stages.append(('merge',py('scripts/merge_data.py',
            *[f'step{s}_{split}={scored}/{task}_scored_{split}_step{s}.json' for s in [1,2,3] for split in ['train','valid']],
            f'train_output_path={scored}/{task}_scored_train.json',f'valid_output_path={scored}/{task}_scored_valid.json',
            f'hydra.run.dir={p["base"]}/hydra/merge')))
    stages.append(('plan',[python,str(ROOT/'repro/plan_training.py'),'--train',str(scored/f'{task}_scored_train.json'),
                           '--valid',str(scored/f'{task}_scored_valid.json'),'--out-dir',str(p['train'])]))
    # Warmup is inserted at execution after checking the actual expansion.
    train=py('DPR/train_dense_encoder.py','train_datasets=[se2_dataset]','dev_datasets=[se2_valid_dataset]',
         'train=biencoder_se2',f'output_dir={p["train"]}/model',f'datasets.train_clusters={cluster}',
         f'datasets.train_file={p["scored"]}',f'datasets.valid_file={p["scored"]}',
         'datasets.hard_neg=true','datasets.top_k=20','train.hard_negatives=20','train.batch_size=8',
         'train.num_train_epochs=6','train.num_save_epochs=2',f'train.learning_rate={lr}',
         'train.warmup_steps=ACTUAL_WARMUP',f'datasets.prompt_pool_path={p["pool"]}',
         'datasets.prompt_setup_type=qa','datasets.task_setup_type=q',f'encoder.cache_dir={p["cache"]}',
         'seed=12345','model_file=null',f'hydra.run.dir={p["train"]}')
    stages.append(('train',train))
    out=p['evaluation'];cp=(args.checkpoint.resolve() if getattr(args,'checkpoint',None) else p['train']/'model/dpr_biencoder.best_valid')
    proto=BUNDLE/'protocol_sedpo_only_sdpa.json'
    stages.append(('embed',py('DPR/generate_dense_embeddings.py',f'model_file={cp}','ctx_src=dpr_se2',
        'shard_id=0','num_shards=1',f'out_file={out}/index',
        'ctx_sources.dpr_se2._target_=sedpo_adapter.ControlledContexts','ctx_sources.dpr_se2.prompt_setup_type=qa',
        f'ctx_sources.dpr_se2.prompt_pool_path={BUNDLE}',f'+ctx_sources.dpr_se2.protocol_file={proto}',
        f'+ctx_sources.dpr_se2.task_name={task}',f'encoder.cache_dir={p["cache"]}',f'hydra.run.dir={out}/embedding_log')))
    stages.append(('retrieve',py('DPR/dense_retriever_beam_score.py',f'model_file={cp}',
        'qa_dataset=qa_se2','ctx_datatsets=[dpr_se2]',f'encoded_ctx_files=[{out}/index_*]',f'out_file={out}/prompts.json',
        'datasets.qa_se2._target_=sedpo_adapter.ControlledQuestions',f'datasets.qa_se2.cache_dir={p["cache"]}',
        f'+datasets.qa_se2.protocol_file={proto}',f'datasets.qa_se2.task_name={task}',
        'ctx_sources.dpr_se2._target_=sedpo_adapter.ControlledContexts','ctx_sources.dpr_se2.prompt_setup_type=qa',
        f'ctx_sources.dpr_se2.prompt_pool_path={BUNDLE}',f'+ctx_sources.dpr_se2.protocol_file={proto}',
        f'+ctx_sources.dpr_se2.task_name={task}','beam_size=3','shot_num=4',f'encoder.cache_dir={p["cache"]}',
        f'hydra.run.dir={out}/retrieval_log')))
    stages.append(('selection',[python,str(ROOT/'repro/export_sedpo_selection.py'),'--protocol',str(proto),
        '--task',task,'--beam-output',str(out/'prompts_beam_score_0.json'),'--checkpoint',str(cp),
        '--repo',str(ROOT/'src'),'--out',str(out/'selection.json')]))
    stages.append(('evaluate',[python,str(ROOT/'repro/evaluate_sedpo.py'),'--protocol',str(proto),'--task',task,
        '--sedpo-selection',str(out/'selection.json'),'--model',model,'--out',str(out/'result.json')]))
    if args.command=='evaluate':
        stages=[(name,cmd) for name,cmd in stages if name in ['embed','retrieve','selection','evaluate']]
    return stages


def bootstrap(cache):
    """Pin official dataset revisions; preserve saved datasets consumed by original tasks."""
    from datasets import load_dataset
    specs=[('arc_c','allenai/ai2_arc','ARC-Challenge','210d026faf9955653af8916fad021475a3f00453'),
           ('openbookqa','allenai/openbookqa','additional','388097ea7776314e93a529163e0fea805b8a6454')]
    for name,hub,config,revision in specs:
        out=cache/'saved_datasets'/name
        if out.exists():raise FileExistsError(f'Refusing unknown pre-existing source dataset: {out}')
        dataset=load_dataset(hub,config,revision=revision,cache_dir=str(cache/'hf'))
        dataset.save_to_disk(str(out))


def run(args):
    source_check()
    jobs=list(JOBS) if args.job=='all' else [args.job]
    if args.command=='plan':
        print(json.dumps({tag:commands(args,tag) for tag in jobs},ensure_ascii=False,indent=2));return
    if args.command=='evaluate' and (args.job=='all' or not args.checkpoint.is_file()):
        raise ValueError('evaluate requires one --job and an existing --checkpoint')
    if args.command=='run' and args.mode=='archived-inputs' and args.training_inputs is None:
        raise ValueError('--training-inputs is required for archived-inputs mode')
    if args.command=='run' and args.mode=='archived-inputs':
        for tag in jobs:check_training_inputs(args.training_inputs,tag,args.allow_different_data)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; verify and plan work without GPU, run does not.')
    if args.work_dir.exists():raise FileExistsError('Use a NEW work directory; automatic training resume/overwrite is disabled.')
    if any(' ' in str(p.resolve()) for p in [args.work_dir,ROOT]):
        raise ValueError('Hydra runtime paths must not contain spaces; clone/run under a simple Linux path.')
    storage=args.work_dir.resolve().parent
    while not storage.exists():storage=storage.parent
    required_gib=(18*len(jobs)+2) if args.command=='run' else 2
    if shutil.disk_usage(storage).free < required_gib*1024**3:
        raise RuntimeError(f'Need at least {required_gib} GiB free working storage; old checkpoints are never automatically deleted.')
    models={}
    for reader in {JOBS[t][0] for t in jobs}:
        model=getattr(args,reader+'_model')
        if model is None:raise ValueError(f'--{reader}-model is required')
        models[reader]=model_check(model.resolve(),reader,args.allow_different_model)
    args.work_dir.mkdir(parents=True)
    write(args.work_dir/'run_manifest.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'mode':args.mode,'jobs':jobs,'models':models,'protocol_sha256':file_sha(BUNDLE/'protocol_sedpo_only_sdpa.json'),
        'source_manifest_sha256':file_sha(ROOT/'repro/runtime_source_sha256.json'),
        'allow_different_data':args.allow_different_data})
    if args.command=='run' and args.mode=='rescore':bootstrap(args.work_dir.resolve()/'cache')
    env=os.environ.copy();env.pop('PT_MODEL',None)
    env.update({'PYTHONPATH':os.pathsep.join(map(str,[ROOT/'repro',ROOT/'src',ROOT/'src/DPR'])),
        'HYDRA_FULL_ERROR':'1','PYTHONUNBUFFERED':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
        'IF_PREFER_LOSS':'2','DIV_REJECT':'1','PEFT_BETA':'0.1','SCORE_RATIO':'1.0','GRAD_CKPT':'1'})
    for tag in jobs:
        p=job_paths(args.work_dir.resolve(),tag)
        for directory in [p['base'],p['random'],p['train'],p['evaluation'],p['scored']/p['cluster'],p['pool']/p['cluster']]:
            directory.mkdir(parents=True,exist_ok=True)
        if args.command=='run' and args.mode=='archived-inputs':
            for role in ['scored','prompt_pool']:
                source=args.training_inputs/p['reader']/role/p['cluster']
                dest=p[role if role=='scored' else 'pool']/p['cluster']
                if not source.is_dir():raise FileNotFoundError(source)
                for file in source.glob('*.json'):shutil.copy2(file,dest/file.name)
        stages=commands(args,tag)
        for name,cmd in stages:
            if name=='train':
                plan=read(p['train']/'training_plan.json')
                expected=JOBS[tag]
                original=read(BUNDLE/'evidence'/({'qwen_arc':'qwen_arc_warmup86','qwen_obqa':'qwen_obqa_warmup108'}.get(tag,tag))/'training_plan.json') if tag!='qwen_arc' else None
                mismatch=plan['expanded_training_samples']!=expected[3] or plan['warmup_steps']!=expected[4]
                if original:mismatch |= any(plan[k]!=original[k] for k in ['train_sha256','valid_sha256'])
                else:
                    expected_hashes=read(ROOT/'repro/training_input_sha256.json')
                    mismatch |= any(file_sha(p['scored']/p['cluster']/f'{p["task"]}_scored_{split}.json') !=
                                    expected_hashes[f'qwen/scored/close_qa/arc_c_scored_{split}.json'] for split in ['train','valid'])
                if mismatch and not args.allow_different_data:
                    raise ValueError(f'{tag}: scoring inputs/expansion differ from historical run; keep evidence and inspect, or use --allow-different-data for a new configuration')
                cmd=[x.replace('ACTUAL_WARMUP',str(plan['warmup_steps'])) for x in cmd]
            log=p['base']/(name+'.log')
            print(tag,name,flush=True)
            with (p['base']/'commands.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'stage':name,'argv':cmd})+'\n')
            with log.open('wb') as f:subprocess.run(cmd,cwd=ROOT/'src',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            if name=='train' and not (p['train']/'model/dpr_biencoder.best_valid').is_file():
                raise RuntimeError(f'Missing best checkpoint: {tag}')
        write(p['base']/'job.complete.json',{'status':'complete','job':tag,'result_sha256':file_sha(p['evaluation']/'result.json')})
        finished=[]
        for completed in jobs:
            result=args.work_dir/completed/'evaluation/result.json'
            marker=args.work_dir/completed/'job.complete.json'
            if marker.is_file():
                state=read(result)
                finished.append({'job':completed,**state['methods']['sedpo']['metrics']})
        write(args.work_dir/'summary.json',{'completed':finished,'requested_jobs':jobs})
        import csv
        with (args.work_dir/'summary.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['job','n','accuracy_percent','weighted_f1_percent','macro_f1_percent'])
            writer.writerows([r['job'],r['n'],r['accuracy'],r['f1_weighted'],r['f1_macro']] for r in finished)
    print('Finished. New outputs are in',args.work_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    verify_parser=sub.add_parser('verify',help='CPU-only verification of all four archived rows')
    verify_parser.add_argument('--bundle',type=Path,default=BUNDLE)
    sub.add_parser('check-source',help='Check vendored runtime source checksums')
    for name in ['plan','run','evaluate']:
        p=sub.add_parser(name)
        p.add_argument('--job',choices=['all',*JOBS],default='all')
        p.add_argument('--mode',choices=['archived-inputs','rescore'],default='archived-inputs')
        p.add_argument('--training-inputs',type=Path)
        p.add_argument('--work-dir',type=Path,required=True)
        p.add_argument('--qwen-model',type=Path)
        p.add_argument('--llama-model',type=Path)
        p.add_argument('--allow-different-model',action='store_true')
        p.add_argument('--allow-different-data',action='store_true')
        if name=='evaluate':p.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='verify':
        for r in verify(args.bundle):print(f"{r['job']}: n={r['n']} Accuracy={r['accuracy']:.2f} weighted-F1={r['f1_weighted']:.4f} macro-F1={r['f1_macro']:.4f}")
    elif args.command=='check-source':source_check();print('Runtime source checksums OK')
    else:run(args)

if __name__=='__main__':main()
