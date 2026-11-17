"""Sequential development acceptance after source labels finish; never starts formal training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_poweradj.run_store import atomic_json,file_hash

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--solver',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--run-prefix',required=True)
    p.add_argument('--reuse-data-audit',type=Path,help='Reuse successful input audit bound to this unchanged manifest; no repeated label solves')
    a=p.parse_args();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=a.dataset_dir.resolve()/'manifest.json';summary=a.dataset_dir.resolve()/'summary.json'
    def status(phase,**kw):
        atomic_json(out/'progress.json',{'phase':phase,'formal_training_started':False,**kw})
    status('waiting_for_source_labels')
    while not summary.exists():time.sleep(5)
    if json.loads(summary.read_text())['status']!='passed':raise RuntimeError('Source migration failed')
    env=dict(os.environ,RL_POWERADJ_SOLVER=str(a.solver.resolve()))
    def run(name,args):
        status(name)
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:
            status('failed',failed_phase=name,log=str(out/(name+'.log')))
            raise RuntimeError(f'{name} failed, see {out/(name+".log")}')
    if a.reuse_data_audit:
        cached=json.loads(a.reuse_data_audit.read_text(encoding='utf-8'))
        if cached.get('status')!='passed' or not cached.get('source_only') or cached.get('manifest_sha256')!=file_hash(manifest):
            raise ValueError('Cached data audit does not match this source manifest')
        shutil.copyfile(a.reuse_data_audit,out/'data-audit.json')
    else:
        run('data_audit',['tools/audit_source_manifest.py','--manifest',str(manifest),'--output',str(out/'data-audit.json')])
    phases={}
    for mode in ('resources','parity','lifecycle','prerun'):
        directory=ROOT/'artifacts'/'runs'/(a.run_prefix+'-'+mode)
        run(mode,['tools/validate_source_release.py','--manifest',str(manifest),'--solver',str(a.solver.resolve()),
                  '--output-dir',str(directory),'--mode',mode])
        phases[mode]=directory/'evidence.json'
    run('tests',['-m','pytest','-o','addopts=','-q','--junitxml='+str(out/'tests.xml')])
    args=['tools/seal_source_readiness.py','--manifest',str(manifest),'--solver',str(a.solver.resolve()),
          '--config','config/candidate-source-training.json','--tests',str(out/'tests.xml'),
          '--data-audit',str(out/'data-audit.json'),'--output',str(out/'ready.json')]
    for mode,path in phases.items():args.extend(['--'+mode,str(path)])
    run('seal',args)
    status('ready_for_training',readiness=str(out/'ready.json'))

if __name__=='__main__':main()
