"""Canonical production factory binding assets, action identity and solver."""
import torch

from .control import ControlPolicy
from .foundation import UnifiedFrozenEncoder, FoundationPowerAdjustmentEnv
from .model import DeviceRegistry
from .psasp import PSASPRunner


class ProductionRuntime:
    def __init__(self, paths, device='auto', registry=None, audit_dir=None):
        self.paths = paths
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device in (None, 'auto') else device
        self.registry = registry or DeviceRegistry.from_cases(
            paths.registry_cases('train'), ControlPolicy.load(paths.control_policy))
        if self.registry.observation_size != 17096 or self.registry.action_size != 302:
            raise ValueError('Production registry must be 17096 task features / 302 actions')
        self.encoder = UnifiedFrozenEncoder.from_project(paths.root, self.device)
        self.runner = PSASPRunner(paths.solver, audit_dir=audit_dir)

    def require_agent(self, agent):
        if agent.metadata.get('encoder') != self.encoder.contract:
            raise ValueError('Model differs from actual unified encoder contract')
        if agent.metadata.get('registry') != self.registry.to_dict():
            raise ValueError('Model registry differs from production runtime')
        if agent.observation_size != 20168 or agent.action_size != 302:
            raise ValueError('Model shape differs from production runtime')

    def environment(self, cases, max_steps=150, initial_perturbations=0, enable_qvio_reward=False):
        self.encoder.assert_runtime_frozen()
        return FoundationPowerAdjustmentEnv(
            paths=self.paths, cases=cases, registry=self.registry,
            foundation_encoder=self.encoder, runner=self.runner, max_steps=max_steps,
            initial_perturbations=initial_perturbations, enable_qvio_reward=enable_qvio_reward)
