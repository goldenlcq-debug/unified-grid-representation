"""Fault-injected source-only lifecycle; no target learning or solver execution."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from rl_poweradj.graph_recipe import GraphRecipe
from rl_poweradj.grid_types import FrozenGraph,GraphObservation

def observation():
    rng=np.random.default_rng(17)
    graph=FrozenGraph('test',rng.normal(size=(2,192)).astype('f'),rng.normal(size=(1,192)).astype('f'),
                      rng.normal(size=(16,192)).astype('f'),np.array([[0],[0]],np.int64))
    return GraphObservation(graph,np.ones((2,8),'f'),np.array([[0,1,0,.1,0]],'f'),np.zeros(5,'f'),
                            np.array([0,0]),np.array([0,1]),np.array([False,True]),('device','device'))

@pytest.fixture(scope='module')
def real_encoder():
    from pathlib import Path
    from rl_poweradj.foundation import UnifiedFrozenEncoder
    return UnifiedFrozenEncoder.from_project(Path(__file__).resolve().parents[1],'cpu')

@pytest.fixture
def setup(tmp_path,monkeypatch,real_encoder):
    from contextlib import nullcontext
    import rl_poweradj.source_training as st
    from rl_poweradj.run_store import file_hash
    solver=tmp_path/'solver';solver.write_bytes(b'test')
    records=[]
    for split in ('train','validation','external_test'):
        evidence=tmp_path/(split+'.json');evidence.write_text(json.dumps({'solver_sha256':file_hash(solver)}))
        records.append({'case_id':split,'grid_family_id':'source' if split!='external_test' else 'target',
                        'parent_case_id':split,'split':split,'source':split,'policy':{},
                        'label_evidence':{'path':evidence.name},'input_cards':{}})
    monkeypatch.setattr(st,'audit_manifest',lambda *a,**k:{'cases':records,'digest':'fixed',
                        'source_family_id':'source','split_protocol':'single_source_zero_shot'})
    monkeypatch.setattr(st.UnifiedFrozenEncoder,'from_project',lambda *a:real_encoder)
    monkeypatch.setattr(st,'PSASPRunner',lambda *a,**k:None)
    monkeypatch.setattr(st,'SolverLock',lambda *a,**k:nullcontext())
    visited=[]
    class Env:
        def __init__(self,*args):self.done=True;self.steps=0
        def reset(self,source,policy,expected_inputs=None):
            assert source!='external_test';visited.append(source)
            self.done=False;self.steps=0
            return observation(),{'converged':False,'steps':0,'error':1.}
        def step(self,action):
            assert action==1
            self.steps=1;self.done=True;s=observation();s.mask[:]=False
            return s,499.,True,False,{'converged':True,'steps':1,'error':0.}
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(st,'GraphPowerAdjustmentEnv',Env)
    import rl_poweradj.graph_training as gt
    monkeypatch.setattr(gt,'GraphPowerAdjustmentEnv',Env)
    recipe=GraphRecipe(total_episodes=2,horizon=2,batch_size=1,warmup=1,replay_capacity=8,
        width=8,layers=1,checkpoint_interval=1,evaluation_interval=1,device='cpu',seed=7)
    return st,tmp_path,solver,recipe,visited

def run(setup,name,**kw):
    st,root,solver,recipe,_=setup
    directory=root/'artifacts'/'runs'/name
    result=st.train_source(root,root/'manifest.json',directory,solver,recipe,
                           acceptance=True,acceptance_cases=1,**kw)
    return directory,result

def checkpoint(directory):
    return json.loads((directory/'progress.json').read_text())['checkpoint']

def weights(directory):
    selected=json.loads((directory/'selected.json').read_text())
    return torch.load(selected['final_model'],weights_only=False)['online']

@pytest.mark.parametrize('boundary',[1,2])
def test_boundary_resume_identical_and_selection_source_only(setup,boundary):
    a,_=run(setup,'continuous')
    b,p=run(setup,'paused',pause_after_episode=boundary)
    assert p['episodes']==boundary and not (b/'selected.json').exists()
    _,resumed=run(setup,'paused',resume=checkpoint(b))
    assert resumed['episodes']==2
    for k,v in weights(a).items():assert torch.equal(v,weights(b)[k])
    selected=json.loads((b/'selected.json').read_text())
    assert selected['selection']=='source_validation_only' and selected['formal_training_started'] is False
    assert 'external_test' not in setup[-1]
    with pytest.raises(ValueError,match='Completed'):run(setup,'paused',resume=checkpoint(b))

def test_partial_validation_not_selected_and_resume_repeats(setup,monkeypatch):
    st=setup[0];original=st.evaluate_records
    def interrupted(*a,**k):raise st.GraphEvaluationPaused('test')
    monkeypatch.setattr(st,'evaluate_records',interrupted)
    directory,progress=run(setup,'partial')
    assert progress['validation_history']==[] and progress['best_model'] is None
    monkeypatch.setattr(st,'evaluate_records',original)
    _,progress=run(setup,'partial',resume=checkpoint(directory))
    assert [r['episode'] for r in progress['validation_history']]==[1,2]

def test_formal_gate_precedes_any_run_creation(setup):
    st,root,solver,recipe,_=setup
    run_dir=root/'artifacts'/'runs'/'formal'
    with pytest.raises(ValueError,match='readiness'):
        st.train_source(root,root/'manifest.json',run_dir,solver,recipe)
    assert not run_dir.exists()

def test_resume_after_final_file_written_before_publication(setup,monkeypatch):
    st=setup[0];original=st.atomic_json
    def fail_publication(path,value):
        if path.name=='selected.json':raise RuntimeError('Injected publication interruption')
        original(path,value)
    monkeypatch.setattr(st,'atomic_json',fail_publication)
    with pytest.raises(RuntimeError,match='Injected'):run(setup,'publish')
    directory=setup[1]/'artifacts'/'runs'/'publish'
    assert list((directory/'models').glob('final-*.pt')) and not (directory/'selected.json').exists()
    monkeypatch.setattr(st,'atomic_json',original)
    run(setup,'publish',resume=checkpoint(directory))
    assert (directory/'selected.json').exists()

def test_recipe_rejects_invalid_resource_and_device():
    with pytest.raises(ValueError):GraphRecipe(node_budget=0)
    with pytest.raises(ValueError):GraphRecipe(warmup=10000,replay_capacity=1)
    with pytest.raises(ValueError):GraphRecipe(device='auto')

def test_evaluation_rejects_acceptance_publication_before_loading_model(tmp_path):
    from rl_poweradj.graph_evaluation import evaluate_published
    publication=tmp_path/'selected.json'
    publication.write_text(json.dumps({'purpose':'acceptance_only','selection':'source_validation_only'}))
    with pytest.raises(ValueError,match='formally published'):
        evaluate_published(tmp_path,publication,tmp_path/'data',tmp_path/'solver',tmp_path/'result')
    assert not (tmp_path/'result').exists()

def test_retention_preserves_current_and_ignores_corrupt(tmp_path):
    from rl_poweradj.graph_release import retain_checkpoints
    from rl_poweradj.grid_types import GRAPH_SCHEMA
    from rl_poweradj.run_store import file_hash
    saved=[]
    for i in range(5):
        path=tmp_path/f'episode-{i}.pt';path.write_bytes(str(i).encode());saved.append(path)
        path.with_suffix('.pt.manifest.json').write_text(json.dumps({'schema':GRAPH_SCHEMA,'sha256':file_hash(path),'episodes':i}))
    corrupt=tmp_path/'episode-999.pt';corrupt.write_bytes(b'partial')
    removed=retain_checkpoints(tmp_path,2,saved[-1])
    assert len(removed)==3 and saved[-1].exists() and saved[-2].exists() and corrupt.exists()
