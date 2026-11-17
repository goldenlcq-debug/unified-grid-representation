"""Bounded development acceptance, never a formal performance experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_poweradj.foundation import UnifiedFrozenEncoder
from rl_poweradj.case_adapter import CaseAdapter
from rl_poweradj.grid_types import GraphObservation
from rl_poweradj.graph_q_network import CandidateGraphQ
from rl_poweradj.graph_dqn import GraphDQN,GraphReplay,GraphTransition
from rl_poweradj.graph_environment import GraphPowerAdjustmentEnv
from rl_poweradj.paths import ProjectPaths,discover_cases
from rl_poweradj.control import ControlPolicy
from rl_poweradj.psasp import PSASPRunner
from rl_poweradj.data_preflight import input_card_hashes
from rl_poweradj.run_store import atomic_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--solver',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    paths=ProjectPaths.from_root(ROOT)
    sources=[paths.cases('train')[0],*discover_cases(ROOT/'检测不同网架适配性的数据')]
    before=[input_card_hashes(x) for x in sources]
    encoder=UnifiedFrozenEncoder.from_project(ROOT,'cpu')
    net=CandidateGraphQ()
    count=sum(p.numel() for p in net.parameters())
    rows=[]
    for source in sources:
        start=time.monotonic();a=CaseAdapter(source);g=encoder.encode_graph(source)
        nodes,edges=a.dynamic();devices=list(a.devices.values())
        ce=np.array([d.edge for d in devices for _ in (0,1)],np.int64)
        targets=np.tile(np.array([0,1],np.int64),len(devices))
        # Read-only scoring of discovered candidates is not execution authorization.
        obs=GraphObservation(g,nodes,edges,np.zeros(5,np.float32),ce,targets,
            np.ones(len(ce),bool),tuple(d.key for d in devices for _ in (0,1)))
        with torch.no_grad():q=net(obs)
        assert len(q)==len(ce) and torch.isfinite(q).all()
        assert count==sum(p.numel() for p in net.parameters())
        rows.append({'case':source.name,'nodes':len(nodes),'edges':len(edges),
                     'candidate_count':len(q),'parameters':count,'seconds':time.monotonic()-start,
                     'rss_bytes':psutil.Process().memory_info().rss})
        print(json.dumps(rows[-1]),flush=True)
    # Existing train-case policy is already explicitly authorized. Convert its
    # names to unique physical locators, refusing duplicates instead of guessing.
    source=sources[0];a=CaseAdapter(source)
    approved=ControlPolicy.load(paths.control_policy)
    names={}
    for d in a.devices.values():names.setdefault(d.name,[]).append(d)
    chosen={}
    for item in approved.actions:
        if len(names.get(item.device_id,[]))!=1:raise ValueError('Ambiguous old policy')
        d=names[item.device_id][0]
        record=chosen.setdefault(d.key,{'key':d.key,'signature':d.signature,'targets':[]})
        record['targets'].append(item.target_status)
    policy={'schema':1,'approved':True,'devices':list(chosen.values())}
    agent=GraphDQN(encoder,width=128,layers=2,seed=41)
    replay=GraphReplay(8);losses=[]
    with GraphPowerAdjustmentEnv(encoder,PSASPRunner(args.solver),max_steps=3) as env:
        s,info=env.reset(source,policy)
        for _ in range(3):
            if s is None or env.done:break
            action=agent.choose(s,epsilon=1.)
            if action is None:break
            ns,reward,term,trunc,info=env.step(action)
            transition=GraphTransition(s,action,reward,ns,term or trunc,'existing_train',source.name)
            replay.add(transition);losses.append(agent.learn([transition]));s=ns
    if not losses:raise RuntimeError('No real PSASP learning transition exercised')
    checkpoint=args.output_dir/'acceptance.pt'
    agent.save(checkpoint,replay,{'episodes':1},{'purpose':'acceptance_only'})
    restored,rr,_,_=GraphDQN.load(checkpoint,encoder,expected_run_contract={'purpose':'acceptance_only'})
    a1=agent.learn(replay.items[:1]);a2=restored.learn(rr.items[:1])
    delta=max(float((p-q).detach().abs().max()) for p,q in zip(agent.online.parameters(),restored.online.parameters()))
    if delta!=0 or a1!=a2:raise RuntimeError(f'Resume differs: delta={delta}, losses={a1},{a2}')
    # New source cases are solved without control authorization: empty policy,
    # terminal initial solve only, no action and no fabricated training samples.
    initial=[]
    for source in sources[1:]:
        with GraphPowerAdjustmentEnv(encoder,PSASPRunner(args.solver)) as env:
            _,info=env.reset(source,{'schema':1,'approved':True,'devices':[]})
            initial.append({'case':source.name,**info})
    encoder.verify_frozen()
    unchanged=before==[input_card_hashes(x) for x in sources]
    if not unchanged:raise RuntimeError('Source cards changed')
    atomic_json(args.output_dir/'acceptance.json',{'status':'passed','purpose':'engineering_acceptance_only',
        'read_only_multi_grid_scoring':rows,'authorized_old_grid_learning_losses':losses,
        'new_grids_no_action_initial_solves':initial,'representation_audit':agent.audit,
        'resume_parameter_max_delta':delta,'encoder_frozen':True,'source_inputs_unchanged':unchanged,
        'formal_training_started':False})

if __name__=='__main__':main()
