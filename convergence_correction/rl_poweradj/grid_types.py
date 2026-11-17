"""Variable-size task schema. Names/IDs are locators, never neural features."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

GRAPH_SCHEMA = {'architecture': 'candidate_graph_q', 'schema': 1,
                'node_dynamic': 8, 'edge_dynamic': 5, 'global_dynamic': 5,
                'token_width': 192, 'global_tokens': 16,
                'reward': 'psasp_terminal_500_step_1', 'identity': 'signed_card_endpoints_circuit'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class FrozenGraph:
    key: str
    nodes: np.ndarray
    edges: np.ndarray
    globals: np.ndarray
    edge_index: np.ndarray

    def validate(self):
        n, m = len(self.nodes), len(self.edges)
        if n < 1 or self.nodes.shape != (n, 192) or self.edges.shape != (m, 192):
            raise ValueError('Invalid node/edge token shape')
        if self.globals.shape != (16, 192) or self.edge_index.shape != (2, m):
            raise ValueError('Invalid global tokens or edge index')
        if self.edge_index.dtype.kind not in 'iu':
            raise ValueError('Integer edge indices required')
        if m and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError('Graph endpoint out of range')
        if not all(np.isfinite(a).all() for a in (self.nodes, self.edges, self.globals)):
            raise ValueError('Nonfinite frozen features')

@dataclass(frozen=True)
class GraphObservation:
    graph: FrozenGraph
    nodes: np.ndarray
    edges: np.ndarray
    globals: np.ndarray
    candidate_edges: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    device_keys: tuple[str, ...]

    def validate(self):
        self.graph.validate()
        n, m, a = len(self.graph.nodes), len(self.graph.edges), len(self.targets)
        if self.nodes.shape != (n, 8) or self.edges.shape != (m, 5) or self.globals.shape != (5,):
            raise ValueError('Dynamic graph schema mismatch')
        if self.candidate_edges.shape != (a,) or self.mask.shape != (a,) or len(self.device_keys) != a:
            raise ValueError('Candidate schema mismatch')
        if self.mask.dtype != np.bool_ or self.candidate_edges.dtype.kind not in 'iu':
            raise ValueError('Invalid mask/index type')
        if a and (self.candidate_edges.min() < 0 or self.candidate_edges.max() >= m):
            raise ValueError('Candidate edge outside graph')
        if not np.isin(self.targets, [0, 1]).all():
            raise ValueError('Nonbinary action target')
        if not all(np.isfinite(v).all() for v in (self.nodes, self.edges, self.globals)):
            raise ValueError('Nonfinite dynamic features')
