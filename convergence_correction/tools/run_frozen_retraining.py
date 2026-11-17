"""Plan or execute serial, fresh multi-seed training and held-out evaluations."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_commands(root, experiment_dir, config, data_audit, seeds, readiness):
    root, experiment_dir = Path(root).resolve(), Path(experiment_dir).resolve()
    allowed = root / 'artifacts/runs'
    if not experiment_dir.is_relative_to(allowed) or experiment_dir == allowed:
        raise ValueError('Experiment directory must be below artifacts/runs')
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    commands = []
    for seed in seeds:
        run = experiment_dir / f'seed-{seed}'
        if run.exists():
            raise FileExistsError(f'Fresh seed directory already exists: {run}')
        commands.append([sys.executable, '-m', 'rl_poweradj.cli', 'train',
                         '--config', str(Path(config).resolve()), '--run-dir', str(run),
                         '--data-audit', str(Path(data_audit).resolve()), '--seed', str(seed),
                         '--readiness', str(Path(readiness).resolve())])
        for phase in ('validation', 'test'):
            commands.append([sys.executable, '-m', 'rl_poweradj.cli', 'evaluate',
                             '--model', str(run / 'models/dqn-best.pt'), '--phase', phase,
                             '--output', str(run / f'results/{phase}.json'),
                             '--max-steps', '150', '--device', 'cuda', '--disable-qvio-reward'])
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/frozen-gridencoder-retrain.json'))
    parser.add_argument('--data-audit', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--execute', action='store_true', help='Execute serially; otherwise print plan only')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    recipe = config['training']
    if recipe.get('max_steps') != 150 or recipe.get('device') != 'cuda' or not recipe.get('disable_qvio_reward'):
        raise ValueError('Launcher requires the approved 150-step CUDA terminal-reward recipe')
    commands = build_commands(Path.cwd(), args.experiment_dir, args.config, args.data_audit, args.seeds, args.readiness)
    print(json.dumps({'execute': args.execute, 'commands': commands}, ensure_ascii=False, indent=2), flush=True)
    if args.execute:
        for command in commands:
            subprocess.run(command, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


if __name__ == '__main__':
    main()
