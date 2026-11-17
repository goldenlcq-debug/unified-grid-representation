"""Run ownership and explicit lifecycle records; never auto-resume a failed job."""
import json
from pathlib import Path
from .run_store import atomic_json
from .solver_lock import SolverLock


def set_status(directory, status, **fields):
    directory = Path(directory)
    progress = directory / 'progress.json'
    previous = json.loads(progress.read_text(encoding='utf-8')) if progress.exists() else {}
    atomic_json(progress, {**previous, **fields, 'status': status})


def request_pause(directory):
    directory = Path(directory).resolve()
    if not (directory / 'run.json').is_file():
        raise FileNotFoundError('No training run.json in requested directory')
    atomic_json(directory / 'pause.request', {'requested': True, 'boundary': 'episode'})


def run_lock(directory):
    import hashlib
    import tempfile
    key = hashlib.sha256(str(Path(directory).resolve()).casefold().encode()).hexdigest()
    return SolverLock(Path(tempfile.gettempdir()) / ('rl_poweradj_run_' + key + '.lock'))
