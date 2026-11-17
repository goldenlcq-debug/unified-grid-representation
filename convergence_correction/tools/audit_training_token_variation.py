"""Read-only cross-parent token audit using training cases only."""
import argparse
from pathlib import Path
import numpy as np
from rl_poweradj.foundation import UnifiedFrozenEncoder
from rl_poweradj.paths import ProjectPaths
from rl_poweradj.run_store import atomic_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = ProjectPaths.from_root()
    encoder = UnifiedFrozenEncoder.from_project(paths.root, 'cpu')
    chosen = {}
    for case in paths.reactive_cases('train'):
        chosen.setdefault(case.name.rsplit('_', 1)[0], case)
        if len(chosen) == 10:
            break
    rows = np.stack([encoder.encode(case) for case in chosen.values()])
    payload = {'cases': [case.name for case in chosen.values()], 'parent_count': len(chosen),
               'max_difference_from_first': float(np.max(np.abs(rows - rows[0]))),
               'token_variance_mean': float(np.var(rows, axis=0).mean()),
               'encoder': encoder.contract,
               'note': 'Training-only structural diagnostic. STATE masking may make tokens invariant across operating points.'}
    atomic_json(args.output, payload)
    print({k: payload[k] for k in ('parent_count', 'max_difference_from_first', 'token_variance_mean')})
