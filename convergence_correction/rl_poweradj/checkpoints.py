"""Select the latest complete resumable checkpoint without loading corrupt payloads."""
import json
from pathlib import Path

from .run_store import file_hash


def select_latest(directory, extra_paths=()):
    directory = Path(directory)
    candidates = []
    rejected = []
    for path in list(directory.glob('*.pt')) + [Path(p) for p in extra_paths if Path(p).is_file()]:
        try:
            info = json.loads(path.with_suffix(path.suffix + '.manifest.json').read_text(encoding='utf-8'))
            if not info.get('has_replay'):
                raise ValueError('Not a resume checkpoint')
            if info['file_size'] != path.stat().st_size:
                raise ValueError('File size mismatch')
            if info['checkpoint_sha256'] != file_hash(path):
                raise ValueError('File hash mismatch')
            candidates.append((int(info['episode']), path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append({'path': str(path), 'reason': str(exc)})
    if not candidates:
        raise ValueError(f'No complete resumable checkpoint in {directory}; rejected={rejected}')
    selected = max(candidates, key=lambda item: (item[0], str(item[1])))[1]
    return selected, rejected
