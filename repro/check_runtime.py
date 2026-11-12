"""No-GPU checks of real DPR loaders, all stage configs and optional training inputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'src/DPR')]
os.environ['IF_PREFER_LOSS']='2'
os.environ['SCORE_RATIO']='1.0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import reproduce

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-inputs',type=Path)
    p.add_argument('--original-repo',type=Path,help='Read actual server training inputs without copying them')
    args=p.parse_args()
    from preflight import main as unused
    subprocess.run([sys.executable,str(ROOT/'repro/preflight.py'),'--protocol',str(reproduce.BUNDLE/'protocol_sedpo_only_sdpa.json'),'--with-dpr'],check=True)
    # Hydra --cfg resolves options without invoking training/model loading.
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'repro'),str(ROOT/'src'),str(ROOT/'src/DPR')])
    fake=argparse.Namespace(command='plan',work_dir=Path(tempfile.gettempdir())/'sedpo_dry_plan_only',mode='rescore',
                           qwen_model=Path('/models/qwen'),llama_model=Path('/models/llama'))
    seen=set();count=0
    for stage,command in reproduce.commands(fake,'qwen_arc'):
        script=Path(command[1])
        if not script.is_relative_to(ROOT/'src') or script in seen:continue
        seen.add(script)
        command=[x.replace('ACTUAL_WARMUP','86') for x in command]
        if stage=='train':
            command=[sys.executable,str(ROOT/'repro/check_train_config.py'),*command[2:]]
        else:command+=['--cfg','job','--resolve']
        result=subprocess.run(command,cwd=ROOT/'src',env=env,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(f'Hydra config failed: {stage}\n{result.stdout}\n{result.stderr}')
        print('Hydra configuration OK:',stage,flush=True);count+=1
    from dpr.data.biencoder_data import Se2Dataset
    expansion={}
    if args.training_inputs or args.original_repo:
        for tag,(reader,task,cluster,expected,_,_) in reproduce.JOBS.items():
            base=(args.training_inputs/reader if args.training_inputs else args.original_repo/('my_data_qwen' if reader=='qwen' else 'my_data_llama_table'))
            train=Se2Dataset(file=str(base/'scored'),top_k=20,train_clusters=cluster,split='train',hard_neg=True,
                             prompt_pool_path=str(base/'prompt_pool'),prompt_setup_type='qa',task_setup_type='q')
            train.load_data()
            if len(train)!=expected:raise ValueError(f'{tag}: DPR expansion {len(train)} != {expected}')
            expansion[tag]=len(train)
            print('Real DPR expanded training data OK:',tag,len(train),flush=True)
            del train;gc.collect()
    print(json.dumps({'status':'passed','hydra_entrypoints':count,'actual_dpr_expansion':expansion,'gpu_models_loaded':False}))

if __name__=='__main__':main()
