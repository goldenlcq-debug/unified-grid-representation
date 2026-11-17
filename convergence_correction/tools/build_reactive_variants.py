"""Generate quarantined shunt-only copies using an explicitly approved policy."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_poweradj.case_adapter import CaseAdapter
from rl_poweradj.psasp import PSASPRunner,CaseWorkspace
from rl_poweradj.data_preflight import input_card_hashes
from rl_poweradj.run_store import atomic_json,file_hash

def build(source,policy,output,solver,family,parent,trials=10,flips=1,seed=0):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if output==source or output.is_relative_to(source):raise ValueError('Output must be outside source')
    if not family or not parent or not 1<=trials<=100 or flips<1:raise ValueError('Invalid generation parameters')
    adapter=CaseAdapter(source);candidates=adapter.authorize(policy)
    by_key={d.key:d for d,t in candidates}
    allowed={(d.key,t) for d,t in candidates}
    togglable=[d for d in by_key.values() if (d.key,0) in allowed and (d.key,1) in allowed]
    if flips>len(togglable):raise ValueError('Insufficient authorized toggle targets')
    before=input_card_hashes(source)
    runner=PSASPRunner(solver)
    with CaseWorkspace(source) as initial:
        base=runner.run(initial)
        if not base.converged:raise ValueError('Generation parent must freshly converge')
    output.mkdir(parents=True)
    rng=np.random.default_rng(seed);records=[];attempts=[];seen=set()
    for number in range(trials):
        name=f'variant_{number:04d}'
        selection=sorted(int(i) for i in rng.choice(len(togglable),flips,replace=False))
        if tuple(selection) in seen:
            attempts.append({'trial':number,'status':'duplicate_selection'});continue
        seen.add(tuple(selection))
        with CaseWorkspace(source) as work:
            a=CaseAdapter(work);changes=[]
            for i in selection:
                d=togglable[i];old=int(a.lines[d.row]['Mark'])
                a.apply(a.devices[d.key],1-old)
                changes.append({'key':d.key,'old':old,'new':1-old})
            cards=input_card_hashes(work)
            if {k:v for k,v in cards.items() if k!='LF.L2'}!={k:v for k,v in before.items() if k!='LF.L2'}:
                raise RuntimeError('Generation changed non-L2 inputs')
            try:result=runner.run(work)
            except Exception as exc:
                attempts.append({'trial':number,'status':'solver_error','error':str(exc)});
                atomic_json(output/'attempts.json',attempts)
                raise  # Never relabel a hardware/solver error as a failed flow.
            # Restore in the isolated workspace to verify an authorized path exists.
            destination=output/name
            shutil.copytree(work,destination)
            restore=CaseAdapter(work)
            for change in reversed(changes):
                if (change['key'],change['old']) not in allowed:
                    raise ValueError('Reverse recovery is not authorized')
                restore.apply(restore.devices[change['key']],change['old'])
            restored=runner.run(work)
            if not restored.converged:raise RuntimeError('Parent restoration failed')
            evidence=output/(name+'.label.json')
            atomic_json(evidence,{'input_cards':cards,'solver_sha256':file_hash(solver),
                'converged':result.converged,'error':result.error,'recovery_verified':True,
                'recovery_steps':len(changes),'recovery_answer_diagnostic_only':changes})
            records.append({'case_id':name,'grid_family_id':family,'parent_case_id':parent,
                'source':name,'split':'quarantine','policy':policy,'input_cards':cards,
                'fresh_label':'converged' if result.converged else 'nonconverged',
                'label_evidence':{'path':evidence.name,'sha256':file_hash(evidence)}})
            attempts.append({'trial':number,'status':'converged' if result.converged else 'nonconverged'})
        atomic_json(output/'attempts.json',attempts)
        atomic_json(output/'manifest.json',{'schema':1,'split_protocol':'single_source_zero_shot',
                                          'source_family_id':'','cases':records})
    if input_card_hashes(source)!=before:raise RuntimeError('Source changed')
    return records

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,type=Path);p.add_argument('--policy',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--solver',required=True,type=Path)
    p.add_argument('--family',required=True);p.add_argument('--parent',required=True)
    p.add_argument('--trials',type=int,default=10);p.add_argument('--flips',type=int,default=1);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    build(a.source,json.loads(a.policy.read_text(encoding='utf-8')),a.output,a.solver,a.family,a.parent,a.trials,a.flips,a.seed)

if __name__=='__main__':main()
