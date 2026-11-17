"""Published-model evaluation only; never optimizes, samples replay or selects."""
import json
from pathlib import Path
import hashlib
import numpy as np
from .foundation import UnifiedFrozenEncoder
from .grid_manifest import audit_manifest
from .graph_dqn import GraphDQN
from .graph_training import evaluate_records
from .psasp import PSASPRunner
from .run_store import atomic_json,file_hash

def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def evaluate_published(root,publication,manifest,solver,output,phase='external_test',device='cpu',resume=False):
    if phase not in ('test','external_test'):raise ValueError('Only final source/zero-shot test phases allowed')
    root=Path(root).resolve();publication=Path(publication).resolve();output=Path(output)
    if output.exists() and not resume:raise FileExistsError(output)
    if resume and not output.exists():raise FileNotFoundError(output)
    selected=json.loads(publication.read_text(encoding='utf-8'))
    if selected.get('purpose')!='formal_source_training' or selected.get('selection')!='source_validation_only':
        raise ValueError('Only a formally published source-selected model can enter final test')
    model=Path(selected['model']).resolve()
    if not model.is_relative_to(publication.parent/'models') or file_hash(model)!=selected['model_sha256']:
        raise ValueError('Published model changed or outside run')
    if selected['solver_sha256']!=file_hash(solver):raise ValueError('Evaluation solver differs from sealed solver')
    data=audit_manifest(manifest)
    if data['source_family_id']!=selected['source_family_id']:raise ValueError('Different source family declaration')
    records=[r for r in data['cases'] if r['split']==phase]
    if not records:raise ValueError('No cases for requested test phase')
    protocol={'publication_sha256':file_hash(publication),'model_sha256':file_hash(model),
              'manifest_digest':data['digest'],'solver_sha256':file_hash(solver),'phase':phase,
              'device':device,'horizon':selected['recipe']['horizon'],'zero_updates':True}
    completed=[]
    if resume:
        old=json.loads(output.read_text(encoding='utf-8'))
        if old['protocol']!=protocol:raise ValueError('Evaluation resume protocol changed')
        completed=old['cases']
        if [r['case_id'] for r in records[:len(completed)]]!=[r['case_id'] for r in completed]:
            raise ValueError('Result is not a prefix of this test set')
    encoder=UnifiedFrozenEncoder.from_project(root,device)
    agent,_,_,_=GraphDQN.load(model,encoder,device=device)
    agent.online.eval();agent.target.eval()
    before=state_hash(agent.online);updates=agent.updates
    payload={'status':'running','protocol':protocol,'cases':completed}
    def record(row):
        completed.append(row);atomic_json(output,payload)
    atomic_json(output,payload)
    try:
        evaluate_records(agent,encoder,PSASPRunner(solver),records[len(completed):],
                         protocol['horizon'],on_case=record)
        if state_hash(agent.online)!=before or agent.updates!=updates:
            raise RuntimeError('Evaluation changed the task network')
        encoder.verify_frozen()
        required=[r for r in completed if not r['initial_converged']]
        families={}
        for family in sorted({r['family'] for r in completed}):
            group=[r for r in required if r['family']==family]
            families[family]={'action_required':len(group),'successes':sum(r['converged'] for r in group),
                             'success_rate':sum(r['converged'] for r in group)/len(group) if group else None}
        rates=[r['success_rate'] for r in families.values() if r['success_rate'] is not None]
        payload.update(status='complete',action_required=len(required),families=families,
            micro_success_rate=sum(r['converged'] for r in required)/len(required) if required else None,
            macro_success_rate=float(np.mean(rates)) if rates else None,
            task_parameters_unchanged=True,encoder_frozen=True)
        atomic_json(output,payload)
    except BaseException as exc:
        payload.update(status='failed',error_type=type(exc).__name__,error=str(exc));atomic_json(output,payload);raise
    return payload
