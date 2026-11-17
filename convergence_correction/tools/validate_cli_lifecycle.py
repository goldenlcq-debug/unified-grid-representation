"""Exercise actual CUDA/PSASP CLI pause, resume and final checkpoint publication."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rl_poweradj.run_store import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-audit', type=Path, required=True)
    parser.add_argument('--pause-episode', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / 'run'
    base = [sys.executable, '-m', 'rl_poweradj.cli', 'train', '--run-dir', str(run),
            '--data-audit', str(args.data_audit.resolve()), '--acceptance-cases', '2',
            '--total-episodes', '2', '--max-steps', '4', '--batch-size', '2', '--device', 'cuda',
            '--disable-qvio-reward', '--checkpoint-interval', '10']
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    started = time.monotonic()
    with (output / 'first.log').open('wb') as log:
        child = subprocess.Popen(base, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        deadline = time.monotonic() + 240
        while not (run / 'run.json').exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if not (run / 'run.json').exists():
            raise RuntimeError(f'CLI startup failed or exceeded budget; PID={child.pid}, inspect first.log')
        if args.pause_episode == 2:
            while child.poll() is None and time.monotonic() < deadline:
                progress = run / 'progress.json'
                if progress.is_file() and json.loads(progress.read_text(encoding='utf-8')).get('completed_episodes', 0) >= 1:
                    break
                time.sleep(0.05)
        subprocess.run([sys.executable, '-m', 'rl_poweradj.cli', 'pause', '--run-dir', str(run)],
                       check=True, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        if child.wait(timeout=240) != 0:
            raise RuntimeError('First CLI phase failed; inspect first.log')
    paused = json.loads((run / 'progress.json').read_text(encoding='utf-8'))
    if paused['status'] != 'paused' or paused['completed_episodes'] != args.pause_episode:
        raise RuntimeError(f'Expected pause after first episode, got {paused}')
    with (output / 'resume.log').open('wb') as log:
        subprocess.run(base + ['--resume', 'latest'], check=True, timeout=240,
                       stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    final = json.loads((run / 'progress.json').read_text(encoding='utf-8'))
    if final['status'] != 'training_complete' or final['completed_episodes'] != 2:
        raise RuntimeError(f'Unexpected resumed result: {final}')
    from rl_poweradj.dqn import DQNAgent
    agent, replay, metrics, episode = DQNAgent.load_checkpoint(run / 'models/dqn-final.pt', 'cpu')
    if episode != 2 or not replay or not metrics.updates:
        raise RuntimeError('Resumed checkpoint lacks learning evidence')
    record = {'status': 'passed', 'purpose': 'cli_pause_resume_acceptance', 'seconds': time.monotonic() - started,
              'paused_episode': args.pause_episode, 'final_episode': episode, 'updates': metrics.updates,
              'transitions': metrics.transitions, 'full_validation_runs': len(metrics.full_validation_history),
              'token_learning': agent.representation_audit,
              'solver_sha256': agent.metadata['run_identity']['contract']['solver_sha256'],
              'production_code': agent.metadata['run_identity']['contract']['code_files'],
              'run_json_sha256': file_hash(run / 'run.json'), 'checkpoint_sha256': file_hash(run / 'models/dqn-final.pt')}
    atomic_json(output / 'acceptance.json', record)
    print(json.dumps({k: record[k] for k in ('status', 'paused_episode', 'final_episode', 'updates', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
