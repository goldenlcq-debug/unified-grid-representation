"""Bounded real-solver training and resume acceptance, with immutable inputs."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from rl_poweradj.control import ControlPolicy
from rl_poweradj.dqn import DQNAgent, train
from rl_poweradj.foundation import UnifiedFrozenEncoder, FoundationPowerAdjustmentEnv
from rl_poweradj.model import DeviceRegistry
from rl_poweradj.paths import ProjectPaths, case_input_fingerprint
from rl_poweradj.psasp import PSASPRunner
from rl_poweradj.run_store import atomic_json, prepare_run, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--data-audit', type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError('Positive episode and step budgets required')
    paths = ProjectPaths.from_root()
    output = args.run_dir.resolve()
    allowed = paths.root / 'artifacts/runs'
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('Output must be an experiment beneath artifacts/runs')
    if output.exists():
        raise FileExistsError(output)
    cases = paths.reactive_cases('train')[:max(args.episodes, 2)]
    fingerprints = {str(p): case_input_fingerprint(p).hex() for p in cases}
    foundation = UnifiedFrozenEncoder.from_project(paths.root, args.device)
    registry = DeviceRegistry.from_cases(paths.registry_cases('train'), ControlPolicy.load(paths.control_policy))
    data_evidence = None
    if args.episodes >= 100 and args.data_audit is None:
        raise ValueError('100-episode pre-run requires full --data-audit')
    if args.data_audit:
        from rl_poweradj.data_preflight import verify_audit
        data_evidence = verify_audit(paths, args.data_audit, registry)
    run = prepare_run(output, {'encoder': foundation.contract, 'cases': fingerprints,
                      'max_steps': args.max_steps, 'episodes': args.episodes,
                      'batch_size': args.batch_size,
                      'data_preflight': data_evidence,
                      'acceptance_tool_sha256': file_hash(Path(__file__)),
                      'code_files': {p.name: file_hash(p) for p in sorted((paths.root / 'rl_poweradj').glob('*.py'))},
                      'solver_sha256': file_hash(paths.solver), 'purpose': 'bounded_acceptance'})
    runner = PSASPRunner(paths.solver, audit_dir=output / 'solver')
    started = time.monotonic()
    profile = {'peak_process_bytes': 0, 'cuda_peak_allocated_bytes': 0}
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    def progress(episode, metrics):
        if process is not None:
            memory = process.memory_info()
            profile['peak_process_bytes'] = max(profile['peak_process_bytes'], getattr(memory, 'peak_wset', memory.rss))
        if args.device.startswith('cuda'):
            profile['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        atomic_json(output / 'progress.json', {'completed': episode, 'updates': metrics.updates,
                    'transitions': metrics.transitions, 'successes': metrics.successes,
                    'elapsed_seconds': time.monotonic() - started, 'profile': profile})
        print(f'episode={episode} updates={metrics.updates} successes={metrics.successes}', flush=True)

    def environment():
        env = FoundationPowerAdjustmentEnv(paths=paths, cases=cases, registry=registry,
                   foundation_encoder=foundation, runner=runner, max_steps=args.max_steps,
                   enable_qvio_reward=False)
        env.run_identity = run
        return env

    try:
        with environment() as env:
            agent, _ = train(env, episodes=args.episodes, max_steps=args.max_steps,
                        batch_size=args.batch_size, seed=0, device=args.device, episode_callback=progress,
                        checkpoint_dir=output / 'checkpoints', checkpoint_interval=args.checkpoint_interval)
        first_updates = agent.updates
        agent.save(output / 'before-resume.pt', replay=agent.training_replay,
                   metrics=agent.training_metrics, episode=args.episodes)
        restored, replay, metrics, episode = DQNAgent.load_checkpoint(output / 'before-resume.pt', args.device)
        with environment() as env:
            restored, _ = train(env, episodes=1, max_steps=args.max_steps, batch_size=args.batch_size,
                        seed=0, device=args.device, agent=restored, replay=replay,
                        metrics=metrics, start_episode=episode, episode_callback=progress)
        foundation.verify_frozen()
        unchanged = all(case_input_fingerprint(Path(p)).hex() == h for p, h in fingerprints.items())
        assert unchanged
        assert restored.updates > first_updates
        assert restored.representation_audit['token_gradient_norm'] > 0
        assert restored.representation_audit['token_update_norm'] > 0
        assert all(p.grad is None and not p.requires_grad for p in foundation.bundle.encoder.parameters())
        restored.save(output / 'final.pt', replay=restored.training_replay,
                      metrics=restored.training_metrics, episode=args.episodes + 1)
        record = {'status': 'passed', 'purpose': 'real_training_resume_acceptance_not_formal_performance',
                  'seconds': time.monotonic() - started, 'device': args.device,
                  'observation_size': restored.observation_size, 'action_size': restored.action_size,
                  'metrics': restored.training_metrics.to_dict(),
                  'learning_audit': restored.representation_audit,
                  'encoder_frozen': True, 'source_unchanged': unchanged,
                  'solver_sha256': runner.executable_sha256, 'updates_before_resume': first_updates,
                  'batch_size': args.batch_size, 'profile': profile,
                  'initial_network_sha256': restored.metadata['initial_network_sha256']}
        atomic_json(output / 'acceptance.json', record)
        print({k: record[k] for k in ('status', 'seconds', 'encoder_frozen', 'source_unchanged')}, flush=True)
    except Exception as exc:
        atomic_json(output / 'acceptance.json', {'status': 'failed', 'exception': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
