"""Reconstruct actual stage wall times from existing logs and immutable results.

No GPU calls, training changes or invented timestamps. Missing historical
boundaries remain explicitly unavailable. UTC inputs are displayed in UTC+8.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from common import read, write

TZ = timezone(timedelta(hours=8))
STAGE = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:[+-]\d\d:\d\d|Z))\s+(\S+)\s*$')


def events(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = STAGE.match(line)
        if match:
            result.append((datetime.fromisoformat(match[1].replace('Z', '+00:00')), match[2]))
    return result


def first(items, name):
    return next((stamp for stamp, event in items if event == name), None)


def interval(start, end, source, now):
    if start is None:
        return {'status': 'not_recorded_or_not_started', 'seconds': None, 'source': source}
    final = end is not None
    elapsed = ((end or now) - start).total_seconds()
    if elapsed < 0:
        raise ValueError('Negative stage interval')
    return {'status': 'complete' if final else 'running', 'seconds': elapsed,
            'started_beijing': start.astimezone(TZ).isoformat(),
            'ended_beijing': end.astimezone(TZ).isoformat() if final else None,
            'source': source}


def duration(value):
    if value is None:
        return '未单独记录／尚未开始'
    seconds = round(value)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (f'{hours}小时' if hours else '') + f'{minutes}分{seconds:02d}秒'


def collect(bundle):
    bundle = Path(bundle)
    now = datetime.now(timezone.utc)
    active = events(bundle / 'table_pipeline.log')
    original = events(bundle / 'runtime_sources/arc_controlled_pipeline.log')
    preliminary = events(bundle / 'sedpo_run_sdpa.log')
    jobs = [('qwen_arc', 'Qwen2.5-7B', 'ARC-C', None, 'TRAIN_ARC_WARMUP86', 'RETRIEVE_ARC_WARMUP86'),
            ('qwen_obqa', 'Qwen2.5-7B', 'OpenBookQA', None, 'TRAIN_QWEN_OBQA_WARMUP108', 'EVALUATE_QWEN_OBQA_FINAL_ROW'),
            ('llama_arc', 'Llama3-8B', 'ARC-C', 'SCORE_llama_arc_THREE_STAGES', 'TRAIN_llama_arc', 'EVALUATE_llama_arc_FINAL_ROW'),
            ('llama_obqa', 'Llama3-8B', 'OpenBookQA', 'SCORE_llama_obqa_THREE_STAGES', 'TRAIN_llama_obqa', 'EVALUATE_llama_obqa_FINAL_ROW')]
    entries = []
    for tag, model, task, score_marker, train_marker, evaluate_marker in jobs:
        log = original if tag == 'qwen_arc' else active
        result_path = bundle / 'table_runs' / tag / 'result.json'
        result = read(result_path) if result_path.is_file() else None
        begin = datetime.fromisoformat(result['started_utc']) if result else None
        finish = datetime.fromisoformat(result['completed_utc']) if result and result.get('status') == 'complete' else None
        training = interval(first(log, train_marker), first(log, evaluate_marker),
                            'arc_controlled_pipeline.log stage boundaries' if tag == 'qwen_arc' else 'table_pipeline.log stage boundaries', now)
        retrieval_start = first(preliminary, 'SEDPO_RETRIEVAL_arc_c') if tag == 'qwen_arc' else first(active, evaluate_marker)
        retrieval = interval(retrieval_start, begin, 'retrieval stage boundary to evaluator.started_utc; includes embedding and selection export/audit', now)
        evaluation = interval(begin, finish, 'result.json started_utc/completed_utc; includes full weight hashing, model loading and 500-query evaluation', now)
        if score_marker:
            start = first(active, score_marker)
            # The first following stage marks the end of score+merge+planning,
            # before any optional disk wait. Do not charge that wait to scoring.
            end = next((stamp for stamp, name in active if start and stamp > start), None)
            scoring = interval(start, end, 'table_pipeline.log score stage to next stage; includes merge and training-plan validation', now)
        elif tag == 'qwen_obqa':
            path = bundle / 'runtime_sources/qwen_obqa_original_pipeline.log'
            text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
            # These original shell lines lack an ISO date. The same log's
            # dated Python records establish the 2026-11-11 experiment date.
            start = end = None
            if '[12:27:51]' in text and '[13:10:00]' in text and '2026-11-11' in text:
                start = datetime(2026, 11, 11, 12, 27, 51, tzinfo=TZ)
                end = datetime(2026, 11, 11, 13, 10, 0, tzinfo=TZ)
            scoring = interval(start, end, 'qwen_obqa_original_pipeline.log original scoring shell boundaries; preference data reused in repaired run', now)
        else:
            scoring = {'status': 'historical_exact_start_unavailable', 'seconds': None,
                       'source': 'Original ARC preference data was reused; no independently verified complete scoring start/end interval has been recovered. Do not fill zero.'}
        stages = {'preference_scoring': scoring, 'retriever_training': training,
                  'embedding_retrieval_and_export': retrieval, 'reader_evaluation': evaluation}
        complete = all(item['status'] == 'complete' for item in stages.values())
        entries.append({'tag': tag, 'model': model, 'dataset': task, 'stages': stages,
                        'complete_main_path_seconds': sum(item['seconds'] for item in stages.values()) if complete else None,
                        'known_completed_stage_seconds': sum(item['seconds'] for item in stages.values() if item['status'] == 'complete')})
    wait_start = first(active, 'WAIT_VERIFIED_CHECKPOINT_BACKUP_FOR_DISK_SPACE')
    wait_end = next((stamp for stamp, name in active if wait_start and stamp > wait_start and name != 'WAIT_VERIFIED_CHECKPOINT_BACKUP_FOR_DISK_SPACE'), None)
    wait = interval(wait_start, wait_end, 'table_pipeline.log initial disk-backup wait, separately reported', now)
    output = bundle / 'table_export'
    output.mkdir(exist_ok=True)
    write(output / 'runtime_summary.json', {'updated_beijing': now.astimezone(TZ).isoformat(),
        'unit': 'wall-clock seconds', 'rows': entries, 'initial_wait_for_storage': wait,
        'accounting': 'Main path counts original preference scoring once plus the repaired training/retrieval/evaluation. Historical failed runs, earlier checkpoints/reruns, idle queue time, and overall instance billing time are not silently added to or represented by this main-path subtotal.'})
    lines = ['# SeDPO 四组运行时间记录', '', f"更新时间：{now.astimezone(TZ).strftime('%Y-%m-%d %H:%M:%S')}（北京时间）。", '',
             '| 模型 | 数据集 | 偏好数据构造 | 检索器训练 | 编码／检索／导出 | 500题评测全过程 |',
             '|---|---|---:|---:|---:|---:|']
    for row in entries:
        values = []
        for key in ['preference_scoring', 'retriever_training', 'embedding_retrieval_and_export', 'reader_evaluation']:
            value = row['stages'][key]
            if value['status'] == 'historical_exact_start_unavailable':
                values.append('早期完整计时缺失')
            elif value['status'] == 'not_recorded_or_not_started':
                values.append('尚未开始')
            else:
                values.append(('进行中，已' if value['status'] == 'running' else '') + duration(value['seconds']))
        lines.append('| ' + ' | '.join([row['model'], row['dataset']] + values) + ' |')
    lines += ['', '计时口径：', '',
              '- 来源是阶段日志的真实起止时间和逐题结果 JSON 的 started_utc/completed_utc；不是预估耗时。',
              '- 评测全过程包含模型权重哈希校验、加载、500 题推理及结果保存，不能标成“纯 GPU 推理时间”。',
              '- 检索阶段包含候选编码、beam 检索和示例选择文件导出校验。训练阶段包含初始化、六轮更新、验证及保存。',
              '- Qwen 的修复训练复用原偏好数据，打分成本不填 0。OBQA 原打分耗时已从原始日志恢复；ARC-C 尚缺可核验的完整打分起止边界，因此该项明确留缺。',
              '- Llama 偏好构造时间含三轮打分、合并和训练步数检查；尚在运行的阶段只显示已耗时。',
              f"- 初次等待备份腾盘：{duration(wait['seconds'])}，单独列出，不计入训练。",
              '- 历史失败尝试、旧训练版本、重复评测和租机空闲时间不混入上述主流程；这里不是整台实例的计费总时长。', '',
              '精确秒数、开始／结束时间与来源见同目录 runtime_summary.json。后续每组完成后自动更新。', '']
    (output / '运行时间汇总.md').write_text('\n'.join(lines), encoding='utf-8')
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    print(collect(parser.parse_args().bundle))
