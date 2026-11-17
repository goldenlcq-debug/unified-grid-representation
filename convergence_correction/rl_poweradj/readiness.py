"""Content-bound release gate for formal training (acceptance runs stay separate)."""
import json
import sys
import platform
from importlib.metadata import version
from pathlib import Path
from .run_store import file_hash
from .foundation import verify_foundation_assets, default_foundation_paths, _source_digest, APPROVED_GRIDFM_SHA256


def code_manifest(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): file_hash(p)
            for folder in ('rl_poweradj', 'tools', 'tests') for p in sorted((root / folder).glob('*.py'))}


def runtime_identity():
    import torch
    return {'python': sys.version, 'platform': platform.platform(),
            'packages': {name: version(name) for name in ('torch', 'numpy', 'scipy', 'torch-geometric', 'psutil')},
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def verify_readiness(root, readiness_path, data_audit, solver, config):
    if readiness_path is None or config is None:
        raise ValueError('Formal training requires --readiness and --config; use explicit acceptance-cases for tests')
    root = Path(root).resolve()
    release = json.loads(Path(readiness_path).read_text(encoding='utf-8'))
    if release.get('status') != 'ready_for_training' or release.get('schema_version') != 1:
        raise ValueError('Training readiness is not passed')
    required = {'tests', 'cli_acceptance', 'parity', 'prerun', 'data_audit'}
    evidence = release.get('evidence', [])
    if len(evidence) != len(required) or {e.get('kind') for e in evidence} != required or release.get('test_count', 0) < 70:
        raise ValueError('Incomplete readiness evidence')
    expected = {'code': code_manifest(root), 'data_audit_sha256': file_hash(data_audit),
                'solver_sha256': file_hash(solver), 'recipe_sha256': file_hash(config),
                'runtime': runtime_identity()}
    for key, value in expected.items():
        if release.get(key) != value:
            raise ValueError(f'Stale readiness: {key} differs')
    assets, source = default_foundation_paths(root)
    if release.get('assets') != verify_foundation_assets(assets) or _source_digest(source) != APPROVED_GRIDFM_SHA256:
        raise ValueError('Readiness encoder assets/source differ')
    for item in evidence:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Readiness evidence must be inside project')
        if file_hash(path) != item['sha256']:
            raise ValueError('Readiness evidence changed')
    return {'path': str(Path(readiness_path).resolve()), 'sha256': file_hash(readiness_path)}
