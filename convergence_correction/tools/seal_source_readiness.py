"""Seal only successful, current-code candidate/source engineering evidence."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_poweradj.graph_recipe import GraphRecipe
from rl_poweradj.graph_release import release_identity,verify_release
from rl_poweradj.run_store import atomic_json,file_hash

def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','solver','config','tests','data-audit','lifecycle','parity','prerun','resources','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    suites=ET.parse(a.tests).getroot().iter('testsuite')
    total=0
    for suite in suites:
        if any(int(suite.get(k,'0')) for k in ('failures','errors','skipped')):raise ValueError('Tests not all passing')
        total+=int(suite.get('tests','0'))
    if total<100:raise ValueError('Incomplete regression suite')
    evidence={}
    for name in ('tests','data_audit','lifecycle','parity','prerun','resources'):
        path=getattr(a,name).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Evidence must be inside project')
        if name!='tests':
            data=json.loads(path.read_text(encoding='utf-8'))
            if data.get('status')!='passed':raise ValueError(f'{name} not passed')
            if name=='data_audit':
                if not data.get('source_only') or data['manifest_sha256']!=file_hash(a.manifest):
                    raise ValueError('Wrong source data audit')
            else:
                if data.get('formal_training_started') is not False or data.get('external_test_used') is not False:
                    raise ValueError('Acceptance must not train formally or use target test data')
        evidence[name]={'path':path.relative_to(ROOT).as_posix(),'sha256':file_hash(path)}
    recipe=GraphRecipe.load(a.config)
    identity=release_identity(ROOT,a.manifest,a.solver,recipe)
    for name in ('lifecycle','parity','prerun','resources'):
        data=json.loads(getattr(a,name).read_text(encoding='utf-8'))
        if data.get('code')!=identity['code']:raise ValueError(f'{name} used another code snapshot')
        if data.get('solver_sha256')!=identity['solver_sha256']:raise ValueError('Acceptance used another solver')
    atomic_json(a.output,{'kind':'candidate_source_release','status':'ready_for_training',
                         'identity':identity,'test_count':total,'evidence':evidence})
    print(verify_release(ROOT,a.output,a.manifest,a.solver,recipe),flush=True)

if __name__=='__main__':main()
