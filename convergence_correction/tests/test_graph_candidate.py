from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from rl_poweradj.grid_types import FrozenGraph, GraphObservation
from rl_poweradj.graph_q_network import CandidateGraphQ
from rl_poweradj.graph_dqn import GraphTransition,GraphReplay,GraphDQN
from rl_poweradj.case_adapter import CaseAdapter

def state(n=3,m=2,a=4,seed=1):
    rng=np.random.default_rng(seed)
    graph=FrozenGraph(str((n,m,seed)),rng.normal(size=(n,192)).astype('f'),
        rng.normal(size=(m,192)).astype('f'),rng.normal(size=(16,192)).astype('f'),
        np.array([np.arange(m)%n,np.arange(m)%n],dtype=np.int64))
    edges=np.zeros((m,5),np.float32);edges[:,1]=1
    result=GraphObservation(graph,np.ones((n,8),np.float32),edges,np.zeros(5,np.float32),
        np.arange(a,dtype=np.int64)%m,np.arange(a,dtype=np.int64)%2,
        np.ones(a,bool),tuple(str(i) for i in range(a)))
    result.validate();return result

def test_shared_parameters_and_mixed_batch():
    net=CandidateGraphQ(16,1)
    count=sum(p.numel() for p in net.parameters())
    a,b=state(),state(9,5,10)
    qa,qb=net(a),net(b)
    q,ptr=net.forward_batch([a,b])
    assert np.array_equal(ptr,[0,4,14])
    torch.testing.assert_close(q,torch.cat((qa,qb)))
    assert sum(p.numel() for p in net.parameters())==count

def test_candidate_permutation_and_chunking():
    net=CandidateGraphQ(16,1);s=state();p=np.array([3,1,0,2])
    t=replace(s,candidate_edges=s.candidate_edges[p],targets=s.targets[p],
              mask=s.mask[p],device_keys=tuple(s.device_keys[i] for i in p))
    torch.testing.assert_close(net(t),net(s)[p])
    torch.testing.assert_close(net(s,chunk_size=1),net(s,chunk_size=99))

def test_empty_candidates_and_invalid_shapes():
    s=state(a=0)
    assert CandidateGraphQ(16,1)(s).shape==(0,)
    with pytest.raises(ValueError):replace(s,nodes=np.zeros((1,8))).validate()
    s=state()
    with pytest.raises(ValueError):replace(s,targets=np.full(4,3)).validate()

def test_replay_interns_graph_and_snapshots_dynamic():
    s=state();r=GraphReplay(2)
    t=GraphTransition(s,0,1.,s,True,'family','case')
    r.add(t);r.add(t)
    assert len(r.graphs)==1
    assert r.items[0].state.graph is r.items[1].state.graph
    s.nodes[:]=0
    assert r.items[0].state.nodes.any()
    with pytest.raises(MemoryError):r.sample(np.random.default_rng(1),1,node_budget=1)

@pytest.fixture(scope='module')
def encoder():
    from rl_poweradj.foundation import UnifiedFrozenEncoder
    return UnifiedFrozenEncoder.from_project(Path(__file__).resolve().parents[1],'cpu')

def test_real_frozen_gradient_and_portable_resume(encoder,tmp_path):
    agent=GraphDQN(encoder,width=16,layers=1,seed=9)
    s=state();end=replace(s,mask=np.zeros(4,bool))
    t=GraphTransition(s,1,2.,end,True,'family','case');r=GraphReplay();r.add(t)
    assert np.isfinite(agent.learn([t]))
    assert agent.audit['encoder_gradients_absent']
    for k in ('node_token_gradient','edge_token_gradient','global_token_gradient'):
        assert agent.audit[k]>0
    agent.save(tmp_path/'test.pt',r,{'episodes':1},{'run':'test'})
    restored,rr,progress,_=GraphDQN.load(tmp_path/'test.pt',encoder,expected_run_contract={'run':'test'})
    assert progress=={'episodes':1}
    l1=agent.learn([t]);l2=restored.learn(rr.items)
    assert l1==l2
    for p,q in zip(agent.online.parameters(),restored.online.parameters()):assert torch.equal(p,q)
    encoder.verify_frozen()
    with pytest.raises(ValueError):GraphDQN.load(tmp_path/'test.pt',encoder,expected_run_contract={'run':'different'})
    assert restored.choose(end) is None
    assert np.isfinite(restored.learn([replace(t,terminal=False)]))

def test_optimizer_rejects_foreign_parameters(encoder):
    agent=GraphDQN(encoder,width=8,layers=1)
    agent.optimizer.add_param_group({'params':[next(encoder.bundle.encoder.parameters())]})
    with pytest.raises(RuntimeError):agent.guard()

def test_greedy_evaluation_does_not_advance_exploration_rng(encoder):
    import copy
    agent=GraphDQN(encoder,width=8,layers=1)
    before=copy.deepcopy(agent.rng.bit_generator.state)
    agent.choose(state(),epsilon=0.)
    assert before==agent.rng.bit_generator.state

def test_production_update_allowlist_rejects_target_and_validation(encoder):
    agent=GraphDQN(encoder,width=8,layers=1)
    agent.training_cases={('source','train_case')}
    s=state()
    before=[p.detach().clone() for p in agent.online.parameters()]
    for family,case in [('target','external_case'),('source','validation_case')]:
        with pytest.raises(ValueError,match='source training split'):
            agent.learn([GraphTransition(s,0,1.,s,True,family,case)])
    assert agent.updates==0
    assert all(torch.equal(a,b) for a,b in zip(before,agent.online.parameters()))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for release')
def test_cuda_candidate_update_and_resume(encoder,tmp_path):
    agent=GraphDQN(encoder,device='cuda',width=16,layers=1,seed=31)
    s=state(n=4,m=3,a=2)
    s.graph.edge_index[:,2]=[1,2]
    s.edges[2]=[1,0,0,.1,0]  # Exercise deterministic CUDA message aggregation.
    t=GraphTransition(s,1,1.,s,False,'source','synthetic')
    replay=GraphReplay();replay.add(t)
    assert np.isfinite(agent.learn([t]))
    path=tmp_path/'cuda.pt';agent.save(path,replay,{'episodes':1},{'purpose':'synthetic_test'})
    restored,rr,_,_=GraphDQN.load(path,encoder,device='cuda')
    assert agent.learn([t])==restored.learn(rr.items)
    for p,q in zip(agent.online.parameters(),restored.online.parameters()):assert torch.equal(p,q)

def test_graph_cache_matches_missing_state_inputs(encoder,monkeypatch):
    from rl_poweradj.paths import ProjectPaths
    from rl_poweradj.psasp import CaseWorkspace
    from rl_poweradj.model import L2_FIELDS,read_table,write_table,is_shunt
    if hasattr(encoder,'_graph_cache'):encoder._graph_cache.clear()
    original=encoder.bundle.encode;calls=[]
    def counted(raw):calls.append(1);return original(raw)
    monkeypatch.setattr(encoder.bundle,'encode',counted)
    with CaseWorkspace(ProjectPaths.from_root().cases('train')[0]) as case:
        before=encoder.encode_graph(case)
        rows=read_table(case/'LF.L2',L2_FIELDS)
        row=next(r for r in rows if is_shunt(r));row['Mark']=1-int(row['Mark'])
        write_table(case/'LF.L2',rows)
        after=encoder.encode_graph(case)
        assert before is after and len(calls)==1
        assert not before.nodes.flags.writeable
        row['X']=float(row['X'])+.001
        write_table(case/'LF.L2',rows)
        changed=encoder.encode_graph(case)
        assert changed.key!=before.key and len(calls)==2

@pytest.mark.parametrize('name',['case118','case300'])
def test_new_grid_adapter_and_local_tokens(encoder,name):
    p=Path(__file__).resolve().parents[1]/'\u8de8\u7f51\u8bad\u7ec3\u6570\u636e'/name/'g001'/'baseline'/'input'
    adapter=CaseAdapter(p);graph=encoder.encode_graph(p)
    assert graph.nodes.shape==(adapter.n,192)
    assert graph.edges.shape==(len(adapter.rows),192)
    nodes,edges=adapter.dynamic()
    assert np.isfinite(nodes).all() and np.isfinite(edges).all()
    with pytest.raises(ValueError):adapter.authorize(adapter.proposal())

def test_precise_edit_on_copy_and_stale_policy(tmp_path):
    from rl_poweradj.paths import ProjectPaths
    from rl_poweradj.psasp import CaseWorkspace
    source=ProjectPaths.from_root().cases('train')[0]
    with CaseWorkspace(source) as p:
        a=CaseAdapter(p);device=next(iter(a.devices.values()))
        policy=a.proposal();policy['approved']=True;policy['devices']=policy['devices'][:1]
        assert len(a.authorize(policy))==2
        before=(p/'LF.L2').read_bytes().splitlines(keepends=True)
        a.apply(device,1-int(a.lines[device.row]['Mark']))
        after=(p/'LF.L2').read_bytes().splitlines(keepends=True)
        assert sum(x!=y for x,y in zip(before,after))==1
        policy['devices'][0]['signature']='wrong'
        with pytest.raises(ValueError):a.authorize(policy)

def test_family_sampler_resume_and_no_validation_sampling():
    from rl_poweradj.grid_manifest import FamilySampler
    records=[{'grid_family_id':f,'parent_case_id':p,'split':s}
             for f,p,s in [('a','one','train'),('a','two','train'),('a','three','train'),('a','four','validation'),('target','one','external_test')]]
    a=FamilySampler(records,7)
    for _ in range(4):a.next()
    b=FamilySampler(records,0);b.load_state_dict(a.state_dict())
    left=[a.next() for _ in range(100)];right=[b.next() for _ in range(100)]
    assert left==right and set(left)=={0,1,2}
    with pytest.raises(ValueError,match='one source'):
        FamilySampler(records+[{'grid_family_id':'target','parent_case_id':'x','split':'train'}])

def test_family_manifest_rejects_leakage_and_stale_cards(tmp_path):
    import copy,json
    from rl_poweradj.grid_manifest import audit_manifest
    from rl_poweradj.data_preflight import input_card_hashes
    from rl_poweradj.paths import ProjectPaths
    source=ProjectPaths.from_root().cases('train')[0];a=CaseAdapter(source)
    policy=a.proposal();policy['approved']=True;policy['devices']=[]
    row={'case_id':'one','grid_family_id':'family','parent_case_id':'parent','split':'train',
         'source':str(source),'policy':policy,'input_cards':input_card_hashes(source)}
    other={**row,'case_id':'two','split':'test'}
    p=tmp_path/'manifest.json'
    p.write_text(json.dumps({'schema':1,'split_protocol':'single_source_zero_shot','source_family_id':'family','cases':[row,other]}))
    with pytest.raises(ValueError,match='leakage'):audit_manifest(p)
    bad=copy.deepcopy(row);bad['input_cards']['LF.L1']='incorrect'
    p.write_text(json.dumps({'schema':1,'split_protocol':'single_source_zero_shot','source_family_id':'family','cases':[bad]}))
    with pytest.raises(ValueError,match='Stale'):audit_manifest(p)

def test_no_action_and_solver_failure_are_not_false_transitions(encoder):
    from rl_poweradj.graph_environment import GraphPowerAdjustmentEnv
    from rl_poweradj.paths import ProjectPaths
    from rl_poweradj.psasp import SolverResult
    source=ProjectPaths.from_root().cases('train')[0]
    class Solved:
        def run(self,p):return SolverResult(True,0.,0)
    with GraphPowerAdjustmentEnv(encoder,Solved()) as env:
        s,info=env.reset(source,{'schema':1,'approved':True,'devices':[]})
        assert s is None and env.done and info['converged']
        with pytest.raises(RuntimeError):env.step(0)
    class Broken:
        def run(self,p):raise RuntimeError('license unavailable')
    with GraphPowerAdjustmentEnv(encoder,Broken()) as env:
        with pytest.raises(RuntimeError,match='license'):env.reset(source,{'schema':1,'approved':True,'devices':[]})
        assert env.workspace is None

def test_bitset_mask_matches_exact_configuration_history():
    from types import SimpleNamespace
    from rl_poweradj.graph_environment import GraphPowerAdjustmentEnv
    env=GraphPowerAdjustmentEnv.__new__(GraphPowerAdjustmentEnv)
    marks=[0,1,0,1,1]
    env.adapter=SimpleNamespace(lines=[{'Mark':v} for v in marks])
    env.candidates=[(SimpleNamespace(row=i),t) for i in range(5) for t in (0,1)]
    env.done=False
    changed=marks.copy();changed[1]=0
    history={tuple(marks),tuple(changed)}
    env.visited={sum(v<<i for i,v in enumerate(row)) for row in history}
    expected=[]
    for device,target in env.candidates:
        row=marks.copy();row[device.row]=target
        expected.append(marks[device.row]!=target and tuple(row) not in history)
    assert env.mask().tolist()==expected

def test_solver_input_mutation_rejected_and_workspace_closed(encoder):
    from rl_poweradj.graph_environment import GraphPowerAdjustmentEnv
    from rl_poweradj.paths import ProjectPaths
    from rl_poweradj.psasp import SolverResult
    source=ProjectPaths.from_root().cases('train')[0]
    class MutatingSolver:
        def run(self,path):
            file=path/'LF.L0';file.write_bytes(file.read_bytes()+b'\n')
            return SolverResult(True,0.,0)
    with GraphPowerAdjustmentEnv(encoder,MutatingSolver()) as env:
        with pytest.raises(RuntimeError,match='changed input cards'):
            env.reset(source,{'schema':1,'approved':True,'devices':[]})
        assert env.workspace is None

def test_single_source_training_gate_without_target_data(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import rl_poweradj.grid_manifest as module
    from rl_poweradj.run_store import file_hash
    monkeypatch.setattr(module,'input_card_hashes',lambda p:{'LF.L1':p.name})
    monkeypatch.setattr(module,'case_validation_key',lambda p,c:p.name)
    monkeypatch.setattr(module,'CaseAdapter',lambda p:SimpleNamespace(n=2000,authorize=lambda policy:[]))
    records=[]
    for split in ('train','validation'):
        evidence=tmp_path/(split+'.json')
        evidence.write_text(json.dumps({'converged':False,'input_cards':{'LF.L1':split},'solver_sha256':'test-solver'}))
        records.append({'case_id':split,'grid_family_id':'source','parent_case_id':split,
            'split':split,'source':split,'policy':{'approved':True},'input_cards':{'LF.L1':split},
            'fresh_label':'nonconverged','label_evidence':{'path':evidence.name,'sha256':file_hash(evidence)}})
    data={'schema':1,'split_protocol':'single_source_zero_shot','source_family_id':'source','cases':records}
    manifest=tmp_path/'manifest.json'
    def check():
        manifest.write_text(json.dumps(data));return module.audit_manifest(manifest,require_training=True)
    assert len(check()['cases'])==2  # No target grids or approvals required.
    for split in ('train','validation','test'):
        records[1]['split']=split;records[1]['grid_family_id']='target'
        with pytest.raises(ValueError,match='Target grid'):check()
    records[1]['split']='validation';records[1]['grid_family_id']='source'
    target={'case_id':'external','grid_family_id':'target','parent_case_id':'external',
            'split':'external_test','source':'external','policy':{'approved':False},'input_cards':{'LF.L1':'external'}}
    records.append(target)
    assert len(check()['cases'])==3  # Target execution approval is deferred to evaluation.
    monkeypatch.setattr(module,'CaseAdapter',lambda p:SimpleNamespace(n=3120 if p.name=='train' else 2000,authorize=lambda policy:[]))
    with pytest.raises(ValueError,match='2000-bus'):check()

def test_historical_labels_require_explicit_policy_and_bound_original(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import rl_poweradj.grid_manifest as module
    from rl_poweradj.run_store import file_hash
    monkeypatch.setattr(module,'input_card_hashes',lambda p:{'LF.L1':'same'})
    monkeypatch.setattr(module,'case_validation_key',lambda p,c:'same')
    monkeypatch.setattr(module,'CaseAdapter',lambda p:SimpleNamespace(n=2000,authorize=lambda policy:[]))
    monkeypatch.setattr(module,'case_input_fingerprint',lambda p:bytes.fromhex('ab'*32))
    old=tmp_path/'old.json';old.write_text(json.dumps({'records':[{'case':'case','first_ac_converged':False,'input_fingerprint':'ab'*32}]}))
    ev=tmp_path/'label.json'
    ev.write_text(json.dumps({'input_cards':{'LF.L1':'same'},'converged':False,'fresh_invocation':False,
        'solver_sha256':None,'label_source':'historical_manifest','historical_case':'case',
        'historical_manifest_path':str(old),'historical_manifest_sha256':file_hash(old)}))
    row={'case_id':'case','grid_family_id':'source','parent_case_id':'parent','split':'train','source':'case',
         'policy':{'approved':True},'input_cards':{'LF.L1':'same'},'flow_label':'nonconverged',
         'label_evidence':{'path':ev.name,'sha256':file_hash(ev)}}
    data={'schema':1,'source_family_id':'source','split_protocol':'single_source_zero_shot','cases':[row]}
    path=tmp_path/'manifest.json'
    def check():path.write_text(json.dumps(data));return module.audit_manifest(path)
    with pytest.raises(ValueError,match='Historical labels not allowed'):check()
    data['label_policy']='historical_allowed_reset_fresh'
    assert check()['cases'][0]['label_status']=='historical'
    row['fresh_label']='nonconverged'
    with pytest.raises(ValueError,match='advertised as fresh'):check()
    row.pop('fresh_label');old.write_text('{}')
    with pytest.raises(ValueError,match='source changed'):check()
