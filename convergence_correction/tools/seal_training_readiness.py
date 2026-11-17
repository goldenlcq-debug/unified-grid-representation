"""Seal verified pretraining evidence; this tool never launches training."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from rl_poweradj.readiness import code_manifest, runtime_identity
from rl_poweradj.foundation import default_foundation_paths, verify_foundation_assets, _source_digest, APPROVED_GRIDFM_SHA256
from rl_poweradj.run_store import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--cli-acceptance', type=Path, required=True)
    parser.add_argument('--parity', type=Path, required=True)
    parser.add_argument('--prerun', type=Path, required=True)
    parser.add_argument('--data-audit', type=Path, required=True)
    parser.add_argument('--solver', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/frozen-gridencoder-retrain.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path.cwd().resolve()
    tests = ET.parse(args.tests).getroot().findall('.//testcase')
    if len(tests) < 70 or any(c.find('failure') is not None or c.find('error') is not None or c.find('skipped') is not None for c in tests):
        raise ValueError('Readiness requires a complete passing test run, with no skips')
    if not any(c.attrib.get('name') == 'test_solver_smoke_on_temporary_case' for c in tests):
        raise ValueError('Real PSASP smoke is missing from test evidence')
    cli, parity, prerun, audit = [json.loads(p.read_text(encoding='utf-8')) for p in
                                (args.cli_acceptance, args.parity, args.prerun, args.data_audit)]
    if any(p.get('status') != 'passed' for p in (cli, parity, prerun, audit)):
        raise ValueError('Acceptance/preflight evidence is not passed')
    code = code_manifest(root)
    production = {Path(k).name: v for k, v in code.items() if k.startswith('rl_poweradj/')}
    if cli['production_code'] != production or parity['production_code'] != production:
        raise ValueError('Real lifecycle/parity evidence is stale for current production code')
    solver_hash = file_hash(args.solver)
    if any(p['solver_sha256'] != solver_hash for p in (cli, parity, prerun)):
        raise ValueError('Solver identity differs across acceptance evidence')
    if prerun['metrics']['completed_episodes'] < 100 or not prerun['encoder_frozen'] or not prerun['source_unchanged']:
        raise ValueError('100-episode frozen-encoder acceptance incomplete')
    if not parity['source_unchanged'] or not parity['encoder_frozen'] or parity['max_parameter_delta'] > 1e-5:
        raise ValueError('Real resume parity was not verified')
    assets, source = default_foundation_paths(root)
    if _source_digest(source) != APPROVED_GRIDFM_SHA256:
        raise ValueError('Unapproved upstream source')
    evidence = []
    for kind, path in zip(('tests', 'cli_acceptance', 'parity', 'prerun', 'data_audit'),
                          (args.tests, args.cli_acceptance, args.parity, args.prerun, args.data_audit)):
        evidence.append({'kind': kind, 'path': path.resolve().relative_to(root).as_posix(), 'sha256': file_hash(path)})
    record = {'schema_version': 1, 'status': 'ready_for_training', 'formal_training_started': False,
              'code': code, 'assets': verify_foundation_assets(assets), 'evidence': evidence,
              'data_audit_sha256': file_hash(args.data_audit), 'solver_sha256': solver_hash,
              'recipe_sha256': file_hash(args.config), 'test_count': len(tests), 'runtime': runtime_identity()}
    atomic_json(args.output, record)
    print(json.dumps({'status': record['status'], 'tests': len(tests), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
