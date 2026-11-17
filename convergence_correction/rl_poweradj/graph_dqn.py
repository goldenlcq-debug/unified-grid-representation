"""Variable-candidate Double DQN with self-contained, verified resume state."""
import copy
from dataclasses import dataclass
from pathlib import Path
import os
import random
import json
import uuid
import numpy as np
import torch
from .foundation import UnifiedFrozenEncoder
from .grid_types import GRAPH_SCHEMA
from .graph_q_network import CandidateGraphQ
from .run_store import atomic_json, file_hash

@dataclass(frozen=True)
class GraphTransition:
    state: object
    action: int
    reward: float
    next_state: object
    terminal: bool
    family: str
    source: str

    def validate(self):
        self.state.validate(); self.next_state.validate()
        if not 0 <= self.action < len(self.state.targets) or not self.state.mask[self.action]:
            raise ValueError('Replay contains invalid action')
        if not np.isfinite(self.reward): raise ValueError('Nonfinite reward')
        if self.state.device_keys != self.next_state.device_keys or not np.array_equal(self.state.targets,self.next_state.targets):
            raise ValueError('Candidate identity changed within transition')

class GraphReplay:
    def __init__(self, capacity=10000):
        if capacity < 1: raise ValueError('Positive capacity required')
        self.capacity,self.items,self.cursor = capacity,[],0
        self.graphs = {}

    def add(self, transition):
        transition.validate()
        # Detach mutable state arrays from the caller; intern immutable graph content.
        transition = copy.deepcopy(transition)
        for state in (transition.state,transition.next_state):
            graph = state.graph
            previous = self.graphs.setdefault(graph.key,graph)
            for a,b in zip((graph.nodes,graph.edges,graph.globals,graph.edge_index),
                           (previous.nodes,previous.edges,previous.globals,previous.edge_index)):
                if not np.array_equal(a,b): raise ValueError('Graph content-address collision')
            object.__setattr__(state,'graph',previous)
        if len(self.items)<self.capacity:self.items.append(transition)
        else:self.items[self.cursor] = transition
        self.cursor = (self.cursor+1)%self.capacity
        used = {s.graph.key for t in self.items for s in (t.state,t.next_state)}
        self.graphs = {k:v for k,v in self.graphs.items() if k in used}

    def sample(self,rng,count,node_budget=100000,edge_budget=200000,action_budget=100000):
        if count < 1: raise ValueError('Positive batch count required')
        selected=[]; used=np.zeros(3,dtype=np.int64)
        for index in rng.permutation(len(self.items)):
            t=self.items[int(index)]
            cost=np.array([len(t.state.nodes)+len(t.next_state.nodes),
                           len(t.state.edges)+len(t.next_state.edges),
                           len(t.state.targets)+len(t.next_state.targets)])
            limits=np.array([node_budget,edge_budget,action_budget])
            if (cost>limits).any():
                if not selected: raise MemoryError('Single transition exceeds declared resource budget')
                continue
            if (used+cost>limits).any():continue
            selected.append(t);used+=cost
            if len(selected)==count:break
        return selected

class GraphDQN:
    def __init__(self,encoder,device='cpu',width=128,layers=2,seed=0,
                 gamma=.99,learning_rate=3e-4,target_interval=100):
        if not isinstance(encoder,UnifiedFrozenEncoder):
            raise ValueError('Mandatory real unified frozen encoder')
        if not 0 <= gamma <= 1 or learning_rate <= 0 or target_interval < 1:
            raise ValueError('Invalid DQN settings')
        encoder.verify_frozen()
        torch.use_deterministic_algorithms(True)
        self.encoder=encoder
        self.contract=copy.deepcopy(encoder.graph_contract)
        torch.manual_seed(seed)
        self.rng=np.random.default_rng(seed)
        self.config=dict(width=width,layers=layers,gamma=gamma,
                         learning_rate=learning_rate,target_interval=target_interval)
        self.online=CandidateGraphQ(width,layers).to(device)
        self.target=copy.deepcopy(self.online).eval()
        for p in self.target.parameters():p.requires_grad_(False)
        self.optimizer=torch.optim.AdamW(self.online.parameters(),lr=learning_rate)
        self.updates=0
        self.audit={}
        self.training_cases=None  # Production installs the audited source-train allowlist.

    def guard(self):
        self.encoder.assert_runtime_frozen()
        ids={id(p) for g in self.optimizer.param_groups for p in g['params']}
        if ids!={id(p) for p in self.online.parameters()}:
            raise RuntimeError('Optimizer must contain exactly the task-head parameters')
        if ids & {id(p) for p in self.encoder.bundle.encoder.parameters()}:
            raise RuntimeError('Frozen encoder included in optimizer')

    def choose(self,state,epsilon=0.,exploration_mask=None):
        self.guard();state.validate()
        valid=np.flatnonzero(state.mask)
        if not len(valid):return None
        if not 0<=epsilon<=1:raise ValueError('Invalid epsilon')
        if exploration_mask is not None:
            preferred=np.asarray(exploration_mask)
            if preferred.shape!=state.mask.shape or preferred.dtype!=np.bool_:
                raise ValueError('Exploration mask must be a boolean candidate mask')
            narrowed=np.flatnonzero(state.mask & preferred)
            if len(narrowed):valid_for_exploration=narrowed
            else:valid_for_exploration=valid
        else:
            valid_for_exploration=valid
        if epsilon > 0 and self.rng.random()<epsilon:
            return int(self.rng.choice(valid_for_exploration))
        with torch.no_grad():
            q=self.online(state)
            if not torch.isfinite(q).all():raise RuntimeError('Nonfinite Q')
            mask=torch.tensor(state.mask,device=q.device)
            return int(q.masked_fill(~mask,-torch.inf).argmax())

    def learn(self,batch):
        if not batch:return None
        self.guard();self.optimizer.zero_grad(set_to_none=True)
        before=self.online.node[0].weight[:,:192].detach().clone()
        losses=[]
        for t in batch:
            t.validate()
            if self.training_cases is not None and (t.family,t.source) not in self.training_cases:
                raise ValueError('Transition is outside the audited source training split')
            predicted=self.online(t.state)[t.action]
            with torch.no_grad():
                expected=predicted.new_tensor(t.reward)
                if not t.terminal and t.next_state.mask.any():
                    q=self.online(t.next_state)
                    mask=torch.tensor(t.next_state.mask,device=q.device)
                    best=q.masked_fill(~mask,-torch.inf).argmax()
                    expected=expected+self.config['gamma']*self.target(t.next_state)[best]
            loss=torch.nn.functional.smooth_l1_loss(predicted,expected)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite TD loss')
            # Accumulate micrographs, avoiding all variable graphs resident in autograd.
            (loss/len(batch)).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),10.,error_if_nonfinite=True)
        norms={name:float(layer[0].weight.grad[:,:192].norm())
               for name,layer in [('node_token_gradient',self.online.node),('edge_token_gradient',self.online.edge)]}
        norms['global_token_gradient']=float(self.online.context[0].weight.grad[:,:3072].norm())
        self.optimizer.step();self.updates+=1
        self.audit={**norms,'node_token_update':float((self.online.node[0].weight[:,:192].detach()-before).norm()),
                    'encoder_gradients_absent':all(p.grad is None for p in self.encoder.bundle.encoder.parameters()),
                    'transitions':[{'family':t.family,'source':t.source,'action':int(t.action),
                        'device_key':t.state.device_keys[t.action],'target':int(t.state.targets[t.action])} for t in batch]}
        if self.updates%self.config['target_interval']==0:self.target.load_state_dict(self.online.state_dict())
        self.guard()
        return float(np.mean(losses))

    def save(self,path,replay,progress,run_contract):
        self.guard();self.encoder.verify_frozen()
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
        payload=dict(schema=GRAPH_SCHEMA,encoder=self.contract,config=self.config,
                     online=self.online.state_dict(),target=self.target.state_dict(),
                     optimizer=self.optimizer.state_dict(),updates=self.updates,audit=self.audit,
                     training_cases=sorted(self.training_cases) if self.training_cases is not None else None,
                     replay=replay,progress=progress,run_contract=run_contract,
                     rng=self.rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     python_rng=random.getstate(),numpy_rng=np.random.get_state())
        with temporary.open('xb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
        atomic_json(path.with_suffix('.pt.manifest.json'),{'sha256':file_hash(path),'schema':GRAPH_SCHEMA,
                    'episodes':progress.get('episodes',0),'updates':self.updates,'has_replay':bool(replay.items)})

    @classmethod
    def load(cls,path,encoder,device='cpu',expected_run_contract=None):
        path=Path(path)
        info=json.loads(path.with_suffix('.pt.manifest.json').read_text(encoding='utf-8'))
        if info.get('schema')!=GRAPH_SCHEMA or info.get('sha256')!=file_hash(path):
            raise ValueError('Wrong architecture or corrupt checkpoint')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['schema']!=GRAPH_SCHEMA or ck['encoder']!=encoder.graph_contract:
            raise ValueError('Frozen encoder/schema mismatch')
        if expected_run_contract is not None and ck['run_contract']!=expected_run_contract:
            raise ValueError('Run contract mismatch')
        for t in ck['replay'].items:t.validate()
        agent=cls(encoder,device=device,**ck['config'])
        agent.online.load_state_dict(ck['online']);agent.target.load_state_dict(ck['target'])
        agent.optimizer.load_state_dict(ck['optimizer'])
        for value in agent.optimizer.state.values():
            for key,tensor in value.items():
                if isinstance(tensor,torch.Tensor):value[key]=tensor.to(device)
        agent.updates=ck['updates'];agent.audit=ck['audit'];agent.rng.bit_generator.state=ck['rng']
        agent.training_cases=set(map(tuple,ck['training_cases'])) if ck.get('training_cases') is not None else None
        torch.set_rng_state(ck['torch_rng'])
        if ck['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        agent.guard()
        return agent,ck['replay'],ck['progress'],ck['run_contract']
