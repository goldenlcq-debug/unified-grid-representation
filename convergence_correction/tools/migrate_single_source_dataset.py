"""Preserve source splits; bind physical control policies and fresh PSASP labels.

Resumable diagnostic only. Does not train a model or touch source case files.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_poweradj.paths import ProjectPaths
from rl_poweradj.control import ControlPolicy
from rl_poweradj.case_adapter import CaseAdapter
from rl_poweradj.data_preflight import input_card_hashes
from rl_poweradj.psasp import PSASPRunner,CaseWorkspace
from rl_poweradj.run_store import atomic_json,file_hash
from rl_poweradj.grid_types import digest
from rl_poweradj.solver_lock import SolverLock

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--solver',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0,help='Debug subset only; zero migrates all splits')
    parser.add_argument('--no-recompute',action='store_true',help='Preserve completed checks; bind remaining historical labels without invoking PSASP')
    args=parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    paths=ProjectPaths.from_root(ROOT)
    old=json.loads(paths.flow_manifest.read_text(encoding='utf-8'))
    old={r['case']:r for r in old['records']}
    policy=ControlPolicy.load(paths.control_policy)
    solver_hash=file_hash(args.solver)
    contract={'schema':1,'solver_sha256':solver_hash,'old_manifest':file_hash(paths.flow_manifest),
              'policy':policy.digest,'limit':args.limit,'split_protocol':'single_source_zero_shot'}
    run=output/'migration-contract.json'
    if run.exists() and json.loads(run.read_text())!=contract:raise ValueError('Migration contract changed')
    atomic_json(run,contract)
    runner=PSASPRunner(args.solver,audit_dir=output/'solver')
    records=[];started=time.monotonic();counts={};sources_unchanged=True
    with SolverLock(output/'migration.lock'):
        for split in ('train','validation','test'):
            cases=paths.cases(split)
            if args.limit:cases=cases[:args.limit]
            for case in cases:
                item_path=output/'records'/(case.name+'.json')
                before=input_card_hashes(case)
                if item_path.exists():
                    record=json.loads(item_path.read_text(encoding='utf-8'))
                    if record['input_cards']!=before or record['split']!=split:raise ValueError('Resume source changed')
                    ev=record['label_evidence'];evpath=output/ev['path']
                    if file_hash(evpath)!=ev['sha256']:raise ValueError('Changed label evidence')
                else:
                    a=CaseAdapter(case)
                    if a.n!=2000:raise ValueError('Non-source grid in source catalog')
                    names={}
                    for device in a.devices.values():names.setdefault(device.name,[]).append(device)
                    devices={}
                    for action in policy.actions:
                        if len(names.get(action.device_id,[]))!=1:raise ValueError('Ambiguous authorized physical device')
                        d=names[action.device_id][0]
                        entry=devices.setdefault(d.key,{'key':d.key,'signature':d.signature,'targets':[]})
                        entry['targets'].append(action.target_status)
                    evidence=output/'labels'/(case.name+'.json')
                    if args.no_recompute:
                        label={'input_cards':before,'solver_sha256':None,
                               'converged':bool(old[case.name]['first_ac_converged']),
                               'error':old[case.name]['first_ac_error'],'fresh_invocation':False,
                               'label_source':'historical_manifest','historical_case':case.name,
                               'historical_manifest_path':str(paths.flow_manifest.resolve()),
                               'historical_manifest_sha256':file_hash(paths.flow_manifest)}
                    else:
                        with CaseWorkspace(case) as workspace:result=runner.run(workspace)
                        label={'input_cards':before,'solver_sha256':solver_hash,
                               'converged':result.converged,'error':result.error,'fresh_invocation':True}
                    atomic_json(evidence,label)
                    record={'case_id':case.name,'source':str(case),'grid_family_id':'source_2000',
                        'parent_case_id':old[case.name]['parent_case'],'split':split,'input_cards':before,
                        'policy':{'schema':1,'approved':True,'devices':list(devices.values())},
                        'flow_label':'converged' if label['converged'] else 'nonconverged',
                        'label_status':'recomputed' if label['fresh_invocation'] else 'historical',
                        'label_evidence':{'path':evidence.relative_to(output).as_posix(),'sha256':file_hash(evidence)}}
                    if input_card_hashes(case)!=before:raise RuntimeError('Source changed during label check')
                    atomic_json(item_path,record)
                if 'flow_label' not in record:
                    record['flow_label']=record['fresh_label'];record['label_status']='recomputed'
                records.append(record)
                key=split+':'+record['flow_label'];counts[key]=counts.get(key,0)+1
                if len(records)%20==0:
                    atomic_json(output/'progress.json',{'completed':len(records),'counts':counts,'seconds':time.monotonic()-started})
                    print(f'migrated {len(records)}; {counts}; seconds={time.monotonic()-started:.1f}',flush=True)
        manifest={'schema':1,'split_protocol':'single_source_zero_shot','source_family_id':'source_2000',
                  'label_policy':'historical_allowed_reset_fresh' if args.no_recompute else 'fresh_required',
                  'purpose':'debug_subset' if args.limit else 'source_training_dataset','cases':records}
        atomic_json(output/'manifest.json',manifest)
        atomic_json(output/'summary.json',{'status':'passed','cases':len(records),'counts':counts,
            'label_status_counts':{s:sum(r['label_status']==s for r in records) for s in ('recomputed','historical')},
            'remaining_recomputation_stopped_by_user':args.no_recompute,
            'manifest_sha256':file_hash(output/'manifest.json'),'source_inputs_unchanged':sources_unchanged,
            'source_only':True,'formal_training_started':False,'seconds':time.monotonic()-started})
        print(json.dumps({'status':'passed','cases':len(records),'counts':counts}),flush=True)

if __name__=='__main__':main()
