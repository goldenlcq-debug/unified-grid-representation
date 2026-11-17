"""Compare uninterrupted versus resumed CUDA training using real PSASP calls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import torch

from rl_poweradj.paths import ProjectPaths, case_input_fingerprint
from rl_poweradj.runtime import ProductionRuntime
from rl_poweradj.dqn import train, DQNAgent
from rl_poweradj.readiness import code_manifest
from rl_poweradj.run_store import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    paths = ProjectPaths.from_root(root)
    cases = paths.reactive_cases('train')[:2]
    original = [case_input_fingerprint(c) for c in cases]
    runtime = ProductionRuntime(paths, 'cuda', audit_dir=output / 'solver')
    with runtime.environment(cases, max_steps=4) as env:
        continuous, _ = train(env, episodes=3, max_steps=4, batch_size=2, seed=13, device='cuda')
    with runtime.environment(cases, max_steps=4) as env:
        first, _ = train(env, episodes=2, max_steps=4, batch_size=2, seed=13, device='cuda')
    first.save(output / 'resume.pt', episode=2, replay=first.training_replay, metrics=first.training_metrics)
    resumed, replay, metrics, episode = DQNAgent.load_checkpoint(output / 'resume.pt', 'cuda')
    with runtime.environment(cases, max_steps=4) as env:
        resumed, _ = train(env, episodes=1, max_steps=4, batch_size=2, seed=13, device='cuda',
                           agent=resumed, replay=replay, metrics=metrics, start_episode=episode)
    differences = {name: float((value - resumed.online.state_dict()[name]).abs().max())
                   for name, value in continuous.online.state_dict().items()}
    runtime.encoder.verify_frozen()
    unchanged = original == [case_input_fingerprint(c) for c in cases]
    metrics_equal = continuous.training_metrics.to_dict() == resumed.training_metrics.to_dict()
    max_delta = max(differences.values())
    record = {'status': 'passed' if max_delta <= 1e-5 and metrics_equal and unchanged else 'failed',
              'max_parameter_delta': max_delta, 'metrics_equal': metrics_equal,
              'parameter_differences': differences, 'encoder_frozen': True, 'source_unchanged': unchanged,
              'solver_sha256': runtime.runner.executable_sha256,
              'production_code': {Path(k).name: v for k, v in code_manifest(root).items() if k.startswith('rl_poweradj/')},
              'continuous_metrics': continuous.training_metrics.to_dict(),
              'resumed_metrics': resumed.training_metrics.to_dict(),
              'device': 'cuda', 'seed': 13, 'horizon': 4, 'episodes': 3}
    atomic_json(output / 'acceptance.json', record)
    print({k: record[k] for k in ('status', 'max_parameter_delta', 'metrics_equal', 'source_unchanged')}, flush=True)
    if record['status'] != 'passed':
        raise RuntimeError('Real PSASP resume parity failed; inspect acceptance.json')


if __name__ == '__main__':
    main()
