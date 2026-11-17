"""Full source manifest audit, binding all inputs, labels and frozen assets."""
import argparse
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_poweradj.grid_manifest import audit_manifest
from rl_poweradj.foundation import verify_foundation_assets,default_foundation_paths,UnifiedFrozenEncoder
from rl_poweradj.run_store import atomic_json,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit_manifest(a.manifest,require_training=True)
    if any(r['split'] not in ('train','validation','test') for r in result['cases']):
        raise ValueError('Release source manifest must contain source data only')
    counts=dict(Counter(r['split'] for r in result['cases']))
    families={r['grid_family_id'] for r in result['cases']}
    if len(families)!=1:raise ValueError('Source manifest contains other grids')
    encoder=UnifiedFrozenEncoder.from_project(ROOT,'cuda')
    checked=set()
    for record in result['cases']:
        if record['validation_key'] in checked:continue
        graph=encoder.encode_graph(record['source'])
        if len(graph.nodes)!=2000:raise ValueError('Non-source frozen graph')
        checked.add(record['validation_key'])
        print(f'frozen graph parse checks {len(checked)}',flush=True)
    encoder.verify_frozen()
    output={'status':'passed','kind':'source_data_audit','manifest_sha256':file_hash(a.manifest),
            'case_count':len(result['cases']),'counts':counts,'source_only':True,'families':sorted(families),
            'parents':{s:len({r['parent_case_id'] for r in result['cases'] if r['split']==s}) for s in counts},
            'labels':dict(Counter(r['flow_label'] for r in result['cases'])),
            'label_status_counts':dict(Counter(r.get('label_status','recomputed') for r in result['cases'])),
            'label_policy':result['label_policy'],
            'dataset_digest':result['digest'],'frozen_input_content_checks':len(checked),
            'assets':verify_foundation_assets(default_foundation_paths(ROOT)[0])}
    atomic_json(a.output,output);print(output,flush=True)

if __name__=='__main__':main()
