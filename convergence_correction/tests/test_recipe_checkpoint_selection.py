import json
from argparse import Namespace

import pytest

from rl_poweradj.recipe import apply_recipe
from rl_poweradj.checkpoints import select_latest
from rl_poweradj.run_store import atomic_json, file_hash


def test_recipe_conflicts_and_types_rejected(tmp_path):
    path = tmp_path / 'recipe.json'
    atomic_json(path, {'schema_version': 1, 'training': {'max_steps': 150}})
    args = Namespace(config=path, max_steps=100)
    with pytest.raises(ValueError, match='conflict'):
        apply_recipe(args, ['--max-steps', '100'])
    assert apply_recipe(args, []).max_steps == 150
    atomic_json(path, {'schema_version': 1, 'training': {'max_steps': True}})
    with pytest.raises(ValueError, match='type'):
        apply_recipe(args, [])
    atomic_json(path, {'schema_version': 1, 'training': {'unknown': 1}})
    with pytest.raises(ValueError, match='Unknown'):
        apply_recipe(args, [])


def test_latest_checkpoint_skips_corrupted_newer_file(tmp_path):
    for episode in (10, 20):
        path = tmp_path / f'episode-{episode}.pt'
        path.write_bytes(str(episode).encode())
        atomic_json(path.with_suffix('.pt.manifest.json'), {
            'has_replay': True, 'episode': episode, 'file_size': path.stat().st_size,
            'checkpoint_sha256': file_hash(path),
        })
    (tmp_path / 'episode-20.pt').write_bytes(b'broken')
    selected, rejected = select_latest(tmp_path)
    assert selected.name == 'episode-10.pt'
    assert len(rejected) == 1
    assert 'size' in rejected[0]['reason']
