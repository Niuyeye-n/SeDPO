"""Verify completed SeDPO runs and export exactly the four requested table rows."""
import argparse
from pathlib import Path
from common import file_sha, load_protocol, metrics, read, write

ROWS = [('llama_obqa', 'OpenBookQA', 'Llama3-8B', 'openbookqa', 75.20, 75.3462),
        ('qwen_obqa', 'OpenBookQA', 'Qwen2.5-7B', 'openbookqa', 87.60, 87.5913),
        ('llama_arc', 'ARC-C', 'Llama3-8B', 'arc_c', 82.00, 82.3391),
        ('qwen_arc', 'ARC-C', 'Qwen2.5-7B', 'arc_c', 88.40, 88.4130)]

# Transcribed from the user-provided Table I image. These are external reported
# scores; the exporter never overwrites them with newly run values.
EXISTING = {
    'llama_obqa': [(72.54,72.7058),(74.80,75.0695),(71.60,72.4391),(74.40,74.5382),(74.00,74.3878),(75.20,75.3462)],
    'qwen_obqa': [(85.88,85.9349),(85.60,85.6265),(86.80,86.8582),(86.40,86.4058),(87.40,87.4649),(87.60,87.5913)],
    'llama_arc': [(80.20,80.2511),(82.00,81.8013),(80.00,80.1478),(82.00,82.0690),(82.00,81.9810),(82.00,82.3391)],
    'qwen_arc': [(86.60,86.6102),(88.00,87.9605),(87.80,87.7748),(87.20,87.4091),(85.11,85.5301),(88.40,88.4130)]}
METHODS = [('Random','ACL 2022'), ('Coverage','EMNLP 2023'), ('SelfAdaptive','ACL 2023'),
           ('DataCompression','ACL 2024'), ('SE2','ACL 2024'), ('CoopES','Ours')]


def full_table(output, rows):
    markdown = ['# Table I：原表数值与新增 SeDPO', '',
                '| Dataset | LLMs | Method | Venue | Accuracy (%) | Weighted-F1 (%) |',
                '|---|---|---|---|---:|---:|']
    tex = [r'% Requires: \usepackage{booktabs,multirow}', r'\begin{table*}[t]', r'\centering',
           r'\caption{Original reported results and newly evaluated SeDPO reference results on two datasets.}',
           r'\label{tab:sedpo_reference}', r'\begin{tabular}{llllrr}', r'\toprule',
           r'Dataset & LLMs & Method & Venue & Accuracy(\%) & Weighted-F1(\%) \\', r'\midrule']
    for group, row in enumerate(rows):
        tag, dataset, model = row['tag'], row['dataset'], row['model']
        for index, ((method, venue), (accuracy, f1)) in enumerate(zip(METHODS, EXISTING[tag])):
            markdown.append(f'| {dataset} | {model} | {method} | {venue} | {accuracy:.2f} | {f1:.4f} |')
            dataset_cell = f'\\multirow{{14}}{{*}}{{{dataset}}}' if group % 2 == 0 and index == 0 else ''
            model_cell = f'\\multirow{{7}}{{*}}{{{model}}}' if index == 0 else ''
            tex.append(f'{dataset_cell} & {model_cell} & {method} & {venue} & {accuracy:.2f} & {f1:.4f} ' + r'\\')
        metric = row.get('metrics')
        acc = f"{metric['accuracy']:.2f}" if metric else '--'
        f1 = f"{metric['f1_weighted']:.4f}" if metric else '--'
        markdown.append(f'| {dataset} | {model} | SeDPO† | NeurIPS 2025 | {acc} | {f1} |')
        tex.append(r' & & SeDPO$^{\dagger}$ & NeurIPS 2025 & ' + acc + ' & ' + f1 + r' \\')
        if group == 3:
            tex.append(r'\bottomrule')
        elif group % 2 == 0:
            tex.append(r'\cmidrule(lr){2-6}')
        else:
            tex.append(r'\midrule')
    note = ('† SeDPO 为本次新运行结果，其他方法是用户提供原表中的报告值，均未重跑。'
            '虽然采用各 500 题、50 候选、k=4，但原实验的具体样本 ID 和完整推理配置未知，'
            '因此这是外部参考比较，不作为严格同设置或配对显著性证明。未完成的结果以 -- 标注；不跨来源加粗评定最佳。')
    markdown += ['', note, '']
    tex += [r'\end{tabular}', r'\vspace{3pt}', r'\begin{minipage}{0.98\textwidth}\footnotesize',
            r'$\dagger$ Newly evaluated SeDPO results; all other values are imported from the supplied original table. '
            r'SeDPO uses 500 test questions, 50 training candidates and $k=4$. The original exact sample IDs and '
            r'inference artifacts were unavailable, so these are external reference comparisons, not a matched '
            r'or paired-significance proof. Dashes indicate pending runs.', r'\end{minipage}', r'\end{table*}', '']
    (output / 'TableI_新增SeDPO完整表.md').write_text('\n'.join(markdown), encoding='utf-8')
    (output / 'TableI_with_SeDPO.tex').write_text('\n'.join(tex), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    output = args.bundle / 'table_export'
    output.mkdir(exist_ok=True)
    protocol_sha = file_sha(args.bundle / 'protocol_sedpo_only_sdpa.json')
    rows = []
    for tag, dataset, model, task, reference_acc, reference_f1 in ROWS:
        row = {'tag': tag, 'dataset': dataset, 'model': model, 'method': 'SeDPO', 'status': 'pending',
               'coopes_published_reference': {'accuracy': reference_acc, 'f1_weighted': reference_f1},
               'comparison_type': 'external_reference_not_paired'}
        result = args.bundle / 'table_runs' / tag / 'result.json'
        marker = result.with_suffix('.complete.json')
        if result.exists() and marker.exists():
            completion, state = read(marker), read(result)
            if completion['result_sha256'] != file_sha(result) or state['status'] != 'complete':
                raise ValueError(f'Incomplete or changed result: {tag}')
            if state['protocol_sha256'] != protocol_sha or state['task'] != task:
                raise ValueError('Table row does not use the frozen task/protocol')
            if state['execution_scope'] != 'sedpo_only' or set(state['methods']) != {'sedpo'}:
                raise ValueError('Only the requested SeDPO row is accepted')
            records = state['methods']['sedpo']['records']
            if len(records) != 500 or len({record['id'] for record in records}) != 500:
                raise ValueError('Expected 500 unique predictions per table row')
            _, expected, _ = load_protocol(args.bundle / 'protocol_sedpo_only_sdpa.json', task)
            gold = {record['id']: record['gold'] for record in expected['gold']}
            if {record['id'] for record in records} != set(gold) or any(record['gold'] != gold[record['id']] for record in records):
                raise ValueError('Table predictions are not the locked test IDs/gold')
            reader_name = 'llama3-8b' if tag.startswith('llama') else 'qwen2.5-7b'
            if reader_name not in state['manifest']['model_path'].lower():
                raise ValueError('Table row was evaluated with the wrong reader path')
            verified = metrics(records)
            if verified != state['methods']['sedpo']['metrics']:
                raise ValueError('Saved metrics do not equal recomputed per-question metrics')
            row.update({'status': 'complete', 'metrics': verified, 'result_path': str(result),
                        'result_sha256': file_sha(result), 'model_manifest': state['manifest']})
        rows.append(row)
    if args.require_complete and any(row['status'] != 'complete' for row in rows):
        raise ValueError('Do not mark the four-row experiment complete while any row is missing')
    write(output / 'sedpo_four_rows.json', {'scope': 'Only four SeDPO rows; existing methods were not rerun',
                                            'protocol_sha256': protocol_sha, 'rows': rows,
                                            'limitation': 'Original CoopES test IDs, pool IDs, exact model artifacts and request settings were not supplied. New results align documented dimensions but are external-reference comparisons, not a strict paired proof.'})
    lines = ['# Table I：新增 SeDPO 四行', '',
             '> 仅运行 SeDPO。原表其他方法的数值保留为论文报告值，不重新运行。', '',
             '| Dataset | LLMs | Method | Accuracy (%) | Weighted-F1 (%) | Macro-F1 (%) | 状态 |',
             '|---|---|---|---:|---:|---:|---|']
    latex = []
    for row in rows:
        m = row.get('metrics')
        acc, f1w, f1m = (f"{m['accuracy']:.2f}", f"{m['f1_weighted']:.4f}", f"{m['f1_macro']:.4f}") if m else ('—', '—', '—')
        lines.append(f"| {row['dataset']} | {row['model']} | SeDPO† | {acc} | {f1w} | {f1m} | {'已完成并核验' if m else '待完成'} |")
        latex += [f"% {row['dataset']} / {row['model']}",
                  f" & & SeDPO$^{{\\dagger}}$ & NeurIPS 2025 & {acc if m else '--'} & {f1w if m else '--'} \\\\"]
    lines += ['', '## 本次设置', '',
              '- 两个数据集各固定 500 道官方 test 题；每个任务的检索候选池固定为 50 条官方 train 示例，k=4。原始题目 ID 和文件哈希均已保存。',
              '- ARC-C 的 500 题按固定 seed/ID 哈希选取；OpenBookQA 包含完整 500 道 test。两者都未声称就是原表那批未知 ID。',
              '- Reader 使用当前服务器的 Qwen2.5-7B-Instruct 和 Llama3-8B-Instruct，本地权重与 tokenizer 的 SHA256 记入结果。',
              '- 两个 reader 分别构造自身的偏好数据并训练检索器，L=50、n_score=300、6 epochs、生效 batch=64、训练 seed=12345。',
              '- ARC-C 学习率 1e-5，OBQA 为上游任务默认 3e-5；warmup 取实际总更新步数的约 10%，不改变 batch 或其他训练超参。',
              '- Reader 使用同一评测程序、多轮四示例、确定性单 token 字母约束、fp16、SDPA、batch=1；输入不静默截断。OBQA 示例与目标均提供 fact。',
              '- eager/fp16 曾出现非有限数值，该失败运行及原配置被保留，未输出有效分数；SDPA 修订只修复数值运行问题，不依据测试准确率选择配置。', '',
              '## 插入原表时的表注', '',
              '† SeDPO 为本次新运行结果；其余方法为原论文报告值。SeDPO 使用固定 500 题和 50 条候选、k=4。由于未取得原实验的样本 ID、模型制品与完整推理配置，二者为外部参考比较，不能据此宣称严格同设置的配对优劣或显著性。', '',
              '将每个模型的 multirow 行数由 6 改为 7，每个数据集由 12 改为 14。四条 LaTeX 插入行见同目录 sedpo_insert_rows.tex；Macro-F1 保留在实验记录中，主表列沿用 Accuracy 与 Weighted-F1。', '',
              '所有数字按实测填写。即使 SeDPO 超过 CoopES，也保留结果，不通过更换样本、模板或检查点制造较低分数。', '']
    (output / 'SeDPO_TableI_四行结果.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'sedpo_insert_rows.tex').write_text('\n'.join(latex) + '\n', encoding='utf-8')
    full_table(output, rows)
    from runtime_ledger import collect
    collect(args.bundle)
    print(f"Verified complete rows: {sum(row['status']=='complete' for row in rows)}/4; output={output}")


if __name__ == '__main__':
    main()
