"""Audit all training/validation/test inputs without invoking PSASP."""
import argparse
from pathlib import Path
from rl_poweradj.data_preflight import audit_dataset
from rl_poweradj.paths import ProjectPaths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_dataset(ProjectPaths.from_root(), args.output)
    print({k: result[k] for k in ('status', 'case_count', 'splits', 'parent_count')})
