"""Strict JSON recipe application; explicit CLI conflicts fail instead of overriding."""
import json
from pathlib import Path


RECIPE_TYPES = {
    'total_episodes': int, 'max_steps': int, 'batch_size': int, 'seed': int,
    'perturbations': int, 'eval_perturbations': int, 'checkpoint_interval': int,
    'checkpoint_keep_last': int, 'epsilon_decay_steps': int,
    'evaluation_interval': int, 'evaluation_episodes': int,
    'full_evaluation_interval': int, 'workers': int, 'device': str,
    'disable_qvio_reward': bool,
}


def apply_recipe(args, argv):
    if not getattr(args, 'config', None):
        return args
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('schema_version') != 1 or set(config) != {'schema_version', 'training'}:
        raise ValueError('Invalid recipe schema')
    training = config['training']
    if not isinstance(training, dict) or set(training) - RECIPE_TYPES.keys():
        raise ValueError('Unknown recipe fields')
    flags = {token.split('=', 1)[0] for token in argv if token.startswith('--')}
    for key, value in training.items():
        if type(value) is not RECIPE_TYPES[key]:
            raise ValueError(f'Invalid type for recipe field {key}')
        flag = '--' + key.replace('_', '-')
        explicit = flag in flags or (key == 'disable_qvio_reward' and '--disable-lcq-reward' in flags)
        if explicit and getattr(args, key) != value:
            raise ValueError(f'CLI/recipe conflict: {key}')
        setattr(args, key, value)
    return args
