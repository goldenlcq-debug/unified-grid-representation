"""Isolated, immutable experiment identity and atomic JSON records."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers/AV may briefly hold the destination without delete sharing.
    # Keep the old valid file until replacement succeeds; never delete it first.
    # Retry only recognized access/sharing/lock errors, and preserve the tmp file
    # for diagnosis if the bounded retry window is exhausted.
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 1.0))


def prepare_run(directory, contract, resume=False):
    directory = Path(directory).resolve()
    manifest = directory / 'run.json'
    if resume:
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['contract'] != contract:
            raise ValueError('Resume run contract mismatch')
        return saved
    if directory.exists():
        raise FileExistsError(f'Fresh run directory already exists: {directory}')
    directory.mkdir(parents=True, exist_ok=False)
    saved = {'schema_version': 1, 'run_id': uuid.uuid4().hex,
             'initialization': 'fresh', 'contract': contract}
    atomic_json(manifest, saved)
    return saved
