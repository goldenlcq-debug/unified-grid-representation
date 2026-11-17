"""Isolated variable-grid environment. No implicit control authorization."""
import numpy as np
from .foundation import UnifiedFrozenEncoder
from .case_adapter import CaseAdapter
from .grid_types import GraphObservation
from .psasp import CaseWorkspace

class GraphPowerAdjustmentEnv:
    def __init__(self, encoder, runner, max_steps=150):
        if not isinstance(encoder, UnifiedFrozenEncoder):
            raise ValueError('Real unified frozen GridEncoder required')
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('Positive integer horizon required')
        encoder.verify_frozen()
        self.encoder, self.runner, self.max_steps = encoder, runner, max_steps
        self.workspace = None
        self.done = True

    def reset(self, source, policy, expected_inputs=None):
        self.close()
        self.workspace = CaseWorkspace(source)
        try:
            path = self.workspace.open()
            if expected_inputs is not None:
                from .data_preflight import input_card_hashes
                if input_card_hashes(path)!=expected_inputs:
                    raise ValueError('Copied source inputs differ from audited manifest')
            self.adapter = CaseAdapter(path)
            self.candidates = self.adapter.authorize(policy)
            self.steps = 0
            self.visited = {self.configuration()}
            self.result = self._solve()
            self.done = self.result.converged or not self.candidates
            if self.result.converged:
                # Already solved needs neither Q inference nor an action set.
                return None, self.info('initial_converged')
            if not self.mask().any():
                self.done = True
            self.graph = self.encoder.encode_graph(path)
            expected = np.array([[abs(int(r['I_No']))-1, abs(int(r['J_No']))-1]
                                 for r in self.adapter.rows], np.int64).reshape(-1,2).T
            if not np.array_equal(expected, self.graph.edge_index):
                raise ValueError('Frozen token/physical edge mapping mismatch')
            return self.observation(), self.info('no_authorized_action' if self.done else 'running')
        except BaseException:
            self.close()
            raise

    def configuration(self):
        # Exact integer bitset, not a probabilistic hash. A shunt candidate
        # changes one bit; avoid copying the entire grid for every candidate.
        return sum(int(row['Mark']) << i for i,row in enumerate(self.adapter.lines))

    def mask(self):
        state = self.configuration()
        result = []
        for d, target in self.candidates:
            current = int(self.adapter.lines[d.row]['Mark'])
            result.append(not self.done and current != target and (state ^ (1 << d.row)) not in self.visited)
        return np.asarray(result, bool)

    def observation(self):
        nodes, edges = self.adapter.dynamic()
        error = self.result.error
        finite = error is not None and np.isfinite(error)
        glob = np.array([np.log1p(abs(error)) if finite else 0., float(finite),
                         self.steps/self.max_steps, np.log1p(self.adapter.n),
                         np.log1p(len(self.adapter.rows))], np.float32)
        obs = GraphObservation(self.graph,nodes,edges,glob,
            np.array([d.edge for d,t in self.candidates],np.int64),
            np.array([t for d,t in self.candidates],np.int64),self.mask(),
            tuple(d.key for d,t in self.candidates))
        obs.validate()
        return obs

    def info(self, reason):
        return {'converged':self.result.converged,'error':self.result.error,
                'steps':self.steps,'reason':reason}

    def _solve(self):
        from .data_preflight import input_card_hashes
        before = input_card_hashes(self.adapter.path)
        result = self.runner.run(self.adapter.path)
        after = input_card_hashes(self.adapter.path)
        if after!=before:
            changed = sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
            raise RuntimeError('Solver changed input cards; refusing an unmodelled transition: '+', '.join(changed))
        return result

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode is already terminal')
        mask = self.mask()
        if not isinstance(action,(int,np.integer)) or not 0 <= action < len(mask) or not mask[action]:
            raise ValueError('Illegal action; solver not invoked')
        d,target = self.candidates[action]
        try:
            self.adapter.apply(d, target)
            self.result = self._solve()
        except BaseException:
            # The failed solve is NOT a negative learning transition.
            self.done = True
            raise
        self.steps += 1
        self.visited.add(self.configuration())
        exhausted = self.steps >= self.max_steps
        no_actions = not self.mask().any()
        self.done = self.result.converged or exhausted or no_actions
        reason = ('success' if self.result.converged else 'budget_exhausted' if exhausted
                  else 'no_legal_action' if no_actions else 'running')
        reward = 499. if self.result.converged else -1.
        info = {**self.info(reason),'device_key':d.key,'target_status':target}
        return self.observation(),reward,self.result.converged,(exhausted or no_actions) and not self.result.converged,info

    def close(self):
        if self.workspace is not None:
            self.workspace.close()
        self.workspace = None
        self.done = True

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
