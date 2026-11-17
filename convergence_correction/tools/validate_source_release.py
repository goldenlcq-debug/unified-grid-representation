"""Real source-only engineering checks. Never reads target-grid directories."""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import time
import psutil
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_poweradj.source_training import train_source
from rl_poweradj.graph_recipe import GraphRecipe
from rl_poweradj.readiness import code_manifest
from rl_poweradj.run_store import atomic_json,file_hash
from rl_poweradj.data_preflight import input_card_hashes

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--solver',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mode',choices=['parity','lifecycle','prerun','resources'],required=True)
    args=p.parse_args()
    root=args.output_dir.resolve()
    if root.exists():raise FileExistsError(root)
    if not root.is_relative_to(ROOT/'artifacts'/'runs'):raise ValueError('Acceptance must use run subtree')
    root.mkdir(parents=True)
    original=json.loads(args.manifest.read_text(encoding='utf-8'))
    needed=100 if args.mode=='prerun' else 3
    rows=[]
    for split in ('train','validation'):
        candidates=[r for r in original['cases'] if r['split']==split and r.get('flow_label',r.get('fresh_label'))=='nonconverged']
        # Spread engineering cases across source parents without consulting any target outcome.
        groups={}
        for r in candidates:groups.setdefault(r['parent_case_id'],[]).append(r)
        ordered=[]
        depth=0
        while len(ordered)<needed and any(depth<len(g) for g in groups.values()):
            ordered.extend(g[depth] for g in groups.values() if depth<len(g));depth+=1
        for record in ordered[:needed if split=='train' else 3]:
            record=copy.deepcopy(record)
            record['label_evidence']['path']=str((args.manifest.resolve().parent/record['label_evidence']['path']).resolve())
            rows.append(record)
    manifest=root/'acceptance-manifest.json'
    atomic_json(manifest,{**original,'cases':rows,'purpose':'engineering_subset_not_formal_performance'})
    before={r['source']:input_card_hashes(r['source']) for r in rows}
    snapshot=code_manifest(ROOT)
    recipe=GraphRecipe(total_episodes=3,horizon=4,batch_size=1,warmup=1,replay_capacity=64,
                       evaluation_interval=2,checkpoint_interval=1,checkpoint_keep=3,device='cuda',seed=13)
    if args.mode=='prerun':
        # Bounded engineering smoke, not a premature formal performance run.
        recipe=GraphRecipe(total_episodes=20,horizon=20,batch_size=16,warmup=16,replay_capacity=512,
                           evaluation_interval=10,checkpoint_interval=5,device='cuda',seed=13)
    if args.mode=='resources':
        recipe=GraphRecipe(total_episodes=2,horizon=8,batch_size=16,warmup=1,replay_capacity=64,
                           evaluation_interval=2,checkpoint_interval=1,device='cuda',seed=13)
    alive=True;peak=[0]
    def monitor():
        while alive:
            info=psutil.Process().memory_info()
            peak[0]=max(peak[0],info.rss,getattr(info,'peak_wset',0));time.sleep(.1)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    def run(name,**kwargs):
        return train_source(ROOT,manifest,root/name,args.solver,recipe,acceptance=True,
                            acceptance_cases=needed,**kwargs)
    started=time.monotonic();facts={}
    try:
        if args.mode=='parity':
            continuous=run('continuous')
            run('resumed',pause_after_episode=2)
            checkpoint=json.loads((root/'resumed'/'progress.json').read_text())['checkpoint']
            resumed=run('resumed',resume=checkpoint)
            a=torch.load(json.loads((root/'continuous'/'selected.json').read_text())['final_model'],map_location='cpu',weights_only=False)
            b=torch.load(json.loads((root/'resumed'/'selected.json').read_text())['final_model'],map_location='cpu',weights_only=False)
            delta=max(float((a['online'][k]-b['online'][k]).abs().max()) for k in a['online'])
            comparable=('episodes','transitions','updates','successes','skipped_converged','lengths','losses','batch_sizes','sampler')
            equal=all(continuous[k]==resumed[k] for k in comparable)
            if delta!=0 or not equal:raise RuntimeError(f'Parity failed: delta={delta}, metrics={equal}')
            facts.update(parameter_max_delta=delta,metrics_equal=equal)
        elif args.mode=='lifecycle':
            run('resumed',pause_after_episode=recipe.total_episodes)
            status=json.loads((root/'resumed'/'progress.json').read_text())
            if status['status']!='paused' or (root/'resumed'/'selected.json').exists():raise RuntimeError('Pause boundary failed')
            run('resumed',resume=status['checkpoint'])
            if not (root/'resumed'/'selected.json').exists():raise RuntimeError('Final validation/publication missing')
            facts.update(final_episode_pause_resume_published=True)
        else:
            progress=run('run')
            if progress['updates']<1:raise RuntimeError('No learned updates in acceptance')
            facts.update(episodes=progress['episodes'],transitions=progress['transitions'],updates=progress['updates'],
                         successes=progress['successes'],lengths=progress['lengths'],last_loss=progress['losses'][-1])
        after={r['source']:input_card_hashes(r['source']) for r in rows}
        if before!=after:raise RuntimeError('Source files changed')
        if snapshot!=code_manifest(ROOT):raise RuntimeError('Code changed during acceptance')
        summary_files=list(root.glob('*/summary.json'))
        summaries=[json.loads(p.read_text(encoding='utf-8')) for p in summary_files]
        atomic_json(root/'evidence.json',{'status':'passed','mode':args.mode,'code':snapshot,
            'solver_sha256':file_hash(args.solver),'device':'cuda','source_only':True,
            'external_test_used':False,'formal_training_started':False,'source_inputs_unchanged':True,
            'seconds':time.monotonic()-started,'process_peak_rss_bytes':peak[0],
            'cuda_peak_allocated':max(s['cuda_peak_allocated'] for s in summaries),
            'encoder_frozen':all(s['encoder_frozen'] for s in summaries),**facts})
        print(json.dumps({'status':'passed','mode':args.mode,**facts}),flush=True)
    finally:
        alive=False;worker.join(timeout=2)

if __name__=='__main__':main()
