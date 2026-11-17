"""Explicit reproducible source-training recipe; no target-grid knobs."""
from dataclasses import dataclass,asdict
import json
from pathlib import Path

@dataclass(frozen=True)
class GraphRecipe:
    total_episodes: int = 2416
    horizon: int = 150
    batch_size: int = 16
    replay_capacity: int = 2048
    warmup: int = 64
    epsilon_decay: int = 10000
    evaluation_interval: int = 50
    checkpoint_interval: int = 10
    checkpoint_keep: int = 3
    width: int = 128
    layers: int = 2
    gamma: float = .99
    learning_rate: float = .0003
    target_interval: int = 100
    node_budget: int = 100000
    edge_budget: int = 200000
    action_budget: int = 100000
    device: str = 'cuda'
    seed: int = 0

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name in ('gamma','learning_rate','device'):continue
            if type(value) is not int or (value<0 if name=='seed' else value<1):
                raise ValueError(f'Invalid recipe field: {name}')
        if not 0<=self.gamma<=1 or not 0<self.learning_rate<1:
            raise ValueError('Invalid learning parameters')
        if self.device not in ('cpu','cuda'):raise ValueError('Explicit cpu or cuda required')
        if self.warmup>self.replay_capacity or self.batch_size>self.replay_capacity:
            raise ValueError('Replay capacity smaller than warmup/batch')

    def to_dict(self):return asdict(self)

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if set(data)!={'schema','training'} or data['schema']!=1:raise ValueError('Invalid recipe schema')
        return cls(**data['training'])
