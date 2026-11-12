"""Package committed code, full result evidence and training inputs; never credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]

VERIFY_SCRIPT='''import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
for relative,expected in manifest['files'].items():
    p=(root/relative).resolve()
    if not p.is_relative_to(root):raise ValueError('Unsafe manifest path')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    if h.hexdigest()!=expected['sha256'] or p.stat().st_size!=expected['bytes']:
        raise ValueError('Checksum mismatch: '+relative)
sys.path.insert(0,str(root/'code'))
import reproduce
reproduce.source_check()
for row in reproduce.verify():print(row['job'],row['n'],row['accuracy'],row['f1_weighted'])
for job in reproduce.JOBS:reproduce.check_training_inputs(root/'training_inputs',job)
print('PASS: all files, four prediction sets and all training inputs match the package manifest')
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-inputs',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Do not overwrite an existing release ZIP')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    prefix='SeDPO_results_20261112/'
    manifest={'code_commit':commit,'format_version':1,'files':{},'excluded':['reader LLM weights','retriever/optimizer checkpoints','virtual environment','credentials'],
              'scope':'Historical full results and inputs; packaging verification is CPU-only, not a new GPU rerun'}
    with zipfile.ZipFile(args.out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        def raw(relative,data):
            if relative in manifest['files']:raise ValueError('Duplicate package member: '+relative)
            z.writestr(prefix+relative,data)
            manifest['files'][relative]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        def disk(relative,path):
            if relative in manifest['files']:raise ValueError('Duplicate package member: '+relative)
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            z.write(path,prefix+relative)
            manifest['files'][relative]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
        for path in paths:
            if path:
                data=subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=ROOT)
                raw('code/'+path,data)
        for role,base in [('training_inputs',args.training_inputs),('results',args.results),('results/evidence',args.evidence)]:
            for path in sorted(base.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and '鍥' not in path.name:
                    disk(role+'/'+path.relative_to(base).as_posix(),path)
        for path in sorted(args.reports.iterdir()):
            if path.is_file() and path.suffix in ['.md','.docx','.json'] and path.name!='核验清单.json':
                if path.suffix=='.md':
                    text=path.read_text(encoding='utf-8').replace((args.reports/'assets').as_posix()+'/', 'assets/')
                    raw('reports/'+path.name,text.encode('utf-8'))
                else:disk('reports/'+path.name,path)
        for path in (args.reports/'assets').glob('*.png'):disk('reports/assets/'+path.name,path)
        raw('verify_package.py',VERIFY_SCRIPT.encode('utf-8'))
        raw('README.md',('''# SeDPO 完整结果与复现材料

解压整个目录后运行：`python verify_package.py`。只用 Python 标准库，无需 GPU。

- `code/`：Git 提交 '''+commit+''' 的源码与归档结果，完整运行说明见 code/README.md。
- `results/table_runs/`：四组最终逐题结果。results 另保留原始协议、检索中间结果和历史诊断，完成状态以 table_runs 四组 complete 标记为准。
- `results/evidence/`：原训练/打分日志、配置、模型归档清单、软硬件记录。
- `training_inputs/`：四组实际合并偏好输入和全训练候选池，重训时可跳过昂贵 teacher 打分。
- `reports/`：Word、Markdown、用户标黄疑问回应及完整统计。报告的 fact 信息以逐条核查更正为准；报告中的历史绝对路径仅作溯源。
- `MANIFEST.json`：逐文件大小与 SHA256；本清单校验交付完整性，不替代论文实验的可比性审查。

不包含 reader 权重、retriever/optimizer 权重、虚拟环境或凭据。最佳权重的原位置和哈希见 evidence 的 retention_manifest；无法仅凭这个包在没有模型的情况下重跑 LLM。四组历史实验已完成，本次交付验证没有重新跑 GPU 训练/推理。CoopES 原稿的具体运行材料仍未知，不据此预设方法优劣。
''').encode('utf-8'))
        z.writestr(prefix+'MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
    with zipfile.ZipFile(args.out) as z:
        failure=z.testzip()
        if failure:raise ValueError('ZIP CRC failure: '+failure)
    digest=hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix('.zip.sha256').write_text(digest+'  '+args.out.name+'\n',encoding='utf-8')
    print(json.dumps({'zip':str(args.out),'files':len(manifest['files']),'bytes':args.out.stat().st_size,'sha256':digest,'commit':commit},ensure_ascii=False))

if __name__=='__main__':main()
