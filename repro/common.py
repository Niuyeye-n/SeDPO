"""Shared, CPU-only contracts for a newly controlled SeDPO/CoopES rerun."""
import hashlib
import json
import math
from pathlib import Path
import unicodedata

LETTERS = 'ABCD'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def content_key(row):
    normalize = lambda text: ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()
    return digest([normalize(row['question']), [normalize(x) for x in row['options']]])


def validate_rows(rows, *, labeled, task):
    ids, contents = set(), set()
    for row in rows:
        if not isinstance(row['id'], str) or row['id'] in ids:
            raise ValueError('Missing, non-string, or duplicate original ID')
        if len(row['options']) != 4 or any(not x.strip() for x in row['options']):
            raise ValueError('Expected four nonempty choices')
        if not row['question'].strip():
            raise ValueError('Empty question')
        if task == 'openbookqa' and not row.get('fact', '').strip():
            raise ValueError('OpenBookQA requires fact for every example and target')
        if labeled:
            if type(row.get('gold')) is not int or not 0 <= row['gold'] < 4:
                raise ValueError('Invalid gold')
        elif any(key in row for key in ['gold', 'answerKey', 'label', 'answer']):
            raise ValueError('Selector-facing test queries must not include gold labels')
        content = content_key(row)
        if content in contents:
            raise ValueError('Repeated question content')
        ids.add(row['id'])
        contents.add(content)
    return ids, contents


def load_protocol(path, task, *, include_gold=True):
    path = Path(path).resolve()
    protocol = read(path)
    if protocol['status'] != 'frozen_new_common_rerun':
        raise ValueError('Protocol has not been frozen')
    if protocol['evaluation']['k'] != 4:
        raise ValueError('This protocol requires k=4')
    data, sources = {}, protocol['tasks'][task]['files']
    for role, source in sources.items():
        if role == 'gold' and not include_gold:
            continue
        location = (path.parent / source['path']).resolve()
        if not location.is_relative_to(path.parent):
            raise ValueError('Protocol file escapes bundle directory')
        if file_sha(location) != source['sha256']:
            raise ValueError(f'Changed protocol input: {role}')
        data[role] = read(location)
    pools = {}
    for role in ['pool', 'dev', 'queries']:
        pools[role] = validate_rows(data[role], labeled=role != 'queries', task=task)
    for first, second in [('pool', 'dev'), ('pool', 'queries'), ('dev', 'queries')]:
        if pools[first][1] & pools[second][1]:
            raise ValueError(f'Content overlap between {first} and {second}')
    if len(data['pool']) != 50 or len(data['queries']) != 500 or len(data['dev']) != 50:
        raise ValueError('Expected 50 training candidates / 50 dev queries / 500 test queries')
    if include_gold:
        gold = data['gold']
        if len(gold) != 500 or {str(row['id']) for row in gold} != pools['queries'][0]:
            raise ValueError('Test gold IDs do not match query IDs')
        if any(type(row['gold']) is not int or not 0 <= row['gold'] < 4 for row in gold):
            raise ValueError('Invalid test gold labels')
    return protocol, data, file_sha(path)


def validate_selection(path, *, protocol_sha, task, pool, queries):
    selection = read(path)
    if selection['method'] not in ['sedpo', 'coopes']:
        raise ValueError('Unknown method')
    if selection['protocol_sha256'] != protocol_sha or selection['task_name'] != task:
        raise ValueError('Selection belongs to a different protocol/task')
    if selection['pool_sha256'] != digest(pool) or selection['queries_sha256'] != digest(queries):
        raise ValueError('Selection used different candidates or test questions')
    provenance = selection['selector_manifest']
    if provenance['test_labels_used'] is not False:
        raise ValueError('Test-label-guided selection cannot be used in this comparison')
    if any(split not in ['train', 'dev'] for split in provenance['tuning_splits']):
        raise ValueError('Selection/tuning may use train or dev only')
    if not provenance.get('implementation_files'):
        raise ValueError('Actual selector implementation provenance is required')
    for source in provenance['implementation_files']:
        if not source.get('path') or len(source.get('sha256', '')) != 64:
            raise ValueError('Missing selector code path/hash')
        # A hash records identity, not correctness of the claimed algorithm.
        if Path(source['path']).is_file() and file_sha(source['path']) != source['sha256']:
            raise ValueError('Selector implementation changed since selection')
    pool_ids, query_ids = {r['id'] for r in pool}, {r['id'] for r in queries}
    selected = {}
    for row in selection['rows']:
        query_id, demos = row['query_id'], row['demo_ids']
        if query_id in selected or query_id not in query_ids:
            raise ValueError('Unknown/repeated selected test ID')
        if len(demos) != 4 or len(set(demos)) != 4 or not set(demos) <= pool_ids:
            raise ValueError('Each method must choose four different members of the same candidate pool')
        selected[query_id] = demos
    if set(selected) != query_ids:
        raise ValueError('Selection does not cover exactly the same 500 questions')
    return selection, selected


def question_block(row):
    # Both selectors go through this one renderer, including fact1 for OBQA.
    prefix = f"Fact: {row['fact']}\n\n" if row.get('fact') else ''
    return prefix + row['question'] + '\n\n' + '\n'.join(f'{LETTERS[i]}. {text}' for i, text in enumerate(row['options']))


def messages(query, demo_ids, pool, system):
    answer = [{'role': 'system', 'content': system}]
    for key in demo_ids:
        example = pool[key]
        answer += [{'role': 'user', 'content': question_block(example)},
                   {'role': 'assistant', 'content': LETTERS[example['gold']]}]
    answer.append({'role': 'user', 'content': question_block(query) + '\n\nAnswer:'})
    return answer


def metrics(records):
    matrix = [[0] * 4 for _ in range(4)]
    for row in records:
        if type(row['gold']) is not int or type(row['pred']) is not int or not (0 <= row['gold'] < 4 and 0 <= row['pred'] < 4):
            raise ValueError('Invalid predictions must not be dropped or replaced with A')
        matrix[row['gold']][row['pred']] += 1
    n = len(records)
    if not n:
        raise ValueError('No predictions')
    support = [sum(line) for line in matrix]
    predicted = [sum(line[i] for line in matrix) for i in range(4)]
    f1 = [2 * matrix[i][i] / (support[i] + predicted[i]) if support[i] + predicted[i] else 0 for i in range(4)]
    return {'n': n, 'accuracy': 100 * sum(matrix[i][i] for i in range(4)) / n,
            'f1_weighted': 100 * sum(f1[i] * support[i] for i in range(4)) / n,
            'f1_macro': 25 * sum(f1), 'confusion_matrix': matrix,
            'gold_distribution': dict(zip(LETTERS, support)), 'pred_distribution': dict(zip(LETTERS, predicted))}


def paired(first, second):
    metrics(first)
    metrics(second)
    a, b = ({r['id']: r for r in rows} for rows in [first, second])
    if len(a) != len(first) or len(b) != len(second) or set(a) != set(b):
        raise ValueError('Paired test IDs do not match uniquely')
    if any(a[key]['gold'] != b[key]['gold'] for key in a):
        raise ValueError('Paired gold labels disagree')
    gains = sum(a[key]['pred'] != a[key]['gold'] and b[key]['pred'] == b[key]['gold'] for key in a)
    losses = sum(a[key]['pred'] == a[key]['gold'] and b[key]['pred'] != b[key]['gold'] for key in a)
    d = gains + losses
    p = min(1., 2 * sum(math.comb(d, i) for i in range(min(gains, losses) + 1)) / 2**d) if d else 1.
    return {'n': len(a), 'sedpo_wrong_coopes_right': gains, 'sedpo_right_coopes_wrong': losses,
            'coopes_minus_sedpo_accuracy_pp': 100 * (gains - losses) / len(a),
            'mcnemar_exact_two_sided_p': p,
            'interpretation': 'Accuracy comparison for this fixed test set; not a universal proof of method superiority; unadjusted p-value'}
