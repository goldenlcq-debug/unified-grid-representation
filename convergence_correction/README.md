# Power-flow convergence correction

Sequential control of PSASP-confirmed nonconvergent cases by switching approved shunt elements. The frozen
encoder processes the input-card graph once per episode with operating-state fields marked missing; its
192-dimensional node and branch tokens and 16 global tokens stay fixed within the episode. A candidate-graph
Q network (width 128, two node-edge message-passing layers) scores every admissible shunt action and is
trained with Double DQN on a family-balanced replay buffer. Each action is followed by a PSASP power-flow
solution; a fresh convergence flag is the only success criterion.

## Contents

| Path | Purpose |
|---|---|
| `rl_poweradj/multigrid_cli.py` | command-line entry (train, evaluate, audit, inspect) |
| `rl_poweradj/multifamily_training.py`, `multifamily_replay.py`, `multifamily_sampler.py`, `multifamily_validation.py`, `multifamily_evaluation.py` | multi-grid training, family-balanced replay and sampling, validation and evaluation |
| `rl_poweradj/graph_q_network.py`, `graph_dqn.py` | candidate-graph Q network and Double DQN |
| `rl_poweradj/graph_environment.py`, `case_adapter.py`, `grid_types.py` | environment, legal-action mask, device binding and status edits |
| `rl_poweradj/foundation.py` | loading and freezing of the grid encoder |
| `rl_poweradj/psasp.py`, `solver_lock.py`, `run_store.py` | solver calls on isolated working copies, locking, atomic result writing |
| `rl_poweradj/evaluation_metrics.py` | evaluation summaries |
| `config/` | training recipes and approved-device policies |
| `tools/` | dataset manifests, split sealing, evaluation drivers for additional test sets |
| `tests/` | unit tests |
| `vendor/grid-foundation/gridfm/` | subset of the encoder library used by this task |

The package also contains modules from earlier stages of the project (fixed-length observation, other reward
definitions); the reported method is the variable-graph path listed above.
