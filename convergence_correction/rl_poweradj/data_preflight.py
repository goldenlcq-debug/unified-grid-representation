"""Content-verified dataset audit with exact active-topology validation reuse."""
import hashlib
import json
import re
from pathlib import Path

from .control import ControlPolicy
from .input_validation import InputCaseValidator
from .model import DeviceRegistry, L2_FIELDS, read_table, is_shunt
from .paths import case_input_fingerprint
from .run_store import atomic_json, file_hash

# Superset of input cards opened by the approved GridSample and task-state adapters.
# Solver result cards are intentionally excluded from dataset identity.
INPUT_CARDS = ('LF.L0', 'LF.L1', 'LF.L2', 'LF.L3', 'LF.L5', 'LF.L6', 'LF.L7',
               'LF.NL4', 'LF.ZONE', 'LF.AREA', 'LF.PSTATION', 'LF.L10', 'LF.L20',
               'LF.BUS', 'ST.S0', 'ST.S1', 'ST.S5', 'ST.S6', 'ST.S11', 'DATALIB.DAT')


def verify_audit(paths, audit_path, registry):
    """Reject stale evidence by recomputing all input card hashes before training."""
    audit = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    if audit.get('schema_version') != 1 or audit.get('status') != 'passed' or audit.get('errors'):
        raise ValueError('Dataset preflight is not a successful audit')
    if audit['registry'] != registry.to_dict():
        raise ValueError('Dataset audit registry differs')
    records = audit['records']
    expected = [(s, c) for s in ('train', 'validation', 'test') for c in paths.reactive_cases(s)]
    if len(records) != len(expected):
        raise ValueError('Dataset audit count differs')
    for record, (split, case) in zip(records, expected):
        if (record['split'], record['case']) != (split, case.name):
            raise ValueError('Dataset audit ordering/catalog differs')
        if record['input_cards'] != input_card_hashes(case):
            raise ValueError(f'Inputs changed since preflight: {case}')
    return {'path': str(Path(audit_path).resolve()), 'sha256': file_hash(audit_path),
            'dataset_digest': audit['dataset_digest']}


def input_card_hashes(case):
    values = {}
    for name in INPUT_CARDS:
        for candidate in (Path(case) / name, Path(case) / (name + '.txt')):
            if candidate.is_file():
                values[candidate.name] = file_hash(candidate)
    return values


def audit_dataset(paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    policy = ControlPolicy.load(paths.control_policy)
    registry = DeviceRegistry.from_cases(paths.registry_cases('train'), policy)
    manifest = json.loads(paths.flow_manifest.read_text(encoding='utf-8'))
    labels = {r['case']: r for r in manifest['records']}
    validator = InputCaseValidator()
    cache = {}
    groups = {}
    fingerprints = {}
    records = []
    errors = []
    splits = {}
    total = sum(len(paths.cases(s)) for s in ('train', 'validation', 'test'))
    for split in ('train', 'validation', 'test'):
        cases = paths.reactive_cases(split)
        if len(cases) != len(paths.cases(split)):
            raise ValueError('Catalog contains cases outside reactive-recovery manifest')
        splits[split] = len(cases)
        for case in cases:
            fingerprint = case_input_fingerprint(case).hex()
            cards = input_card_hashes(case)
            row_errors = []
            label = labels[case.name]
            match = re.fullmatch(r'(sample_\d+)_\d+', case.name)
            if not match:
                raise ValueError(f'Unknown parent identity: {case.name}')
            parent = match.group(1)
            if label.get('parent_case') != parent or label.get('split') != split:
                row_errors.append('parent/split differs from manifest')
            if label.get('input_fingerprint') != fingerprint:
                row_errors.append('electrical input changed since label audit')
            if parent in groups and groups[parent] != split:
                row_errors.append('parent leakage')
            if fingerprint in fingerprints and fingerprints[fingerprint] != split:
                row_errors.append('input fingerprint leakage')
            groups[parent] = split
            fingerprints[fingerprint] = split
            rows = read_table(case / 'LF.L2', L2_FIELDS)
            shunts = [r for r in rows if is_shunt(r)]
            if len({r.identifier for r in shunts}) != len(shunts):
                row_errors.append('duplicate shunt ID')
            if set(registry.shunt_ids) != {r.identifier for r in shunts}:
                row_errors.append('shunt registry mismatch')
            if any(int(r['Mark']) not in (0, 1) for r in shunts):
                row_errors.append('nonbinary shunt Mark')
            # Active validator ignores shunt Mark. Reuse only when EVERY other
            # card and every other L2 field is identical, not merely parent ID.
            canonical = [list(r.values) for r in rows]
            for values, row in zip(canonical, rows):
                if is_shunt(row):
                    values[0] = 0
            active_key = hashlib.sha256(json.dumps({
                'cards': {k: v for k, v in cards.items() if k not in ('LF.L2', 'LF.L2.txt')},
                'l2': canonical,
            }, sort_keys=True).encode()).hexdigest()
            if active_key not in cache:
                validation = validator.validate(case)
                found_registry = DeviceRegistry.from_cases([case], policy)
                cache[active_key] = (validation.active_power_valid,
                                     list(validation.errors), found_registry == registry)
            active_valid, validation_errors, registry_equal = cache[active_key]
            if not active_valid:
                row_errors.extend(validation_errors)
            if not registry_equal:
                row_errors.append('full registry identity mismatch')
            if case_input_fingerprint(case).hex() != fingerprint:
                row_errors.append('case changed during audit')
            records.append({'case': case.name, 'split': split, 'parent': parent,
                            'input_fingerprint': fingerprint, 'input_cards': cards,
                            'active_validation_key': active_key, 'errors': row_errors})
            if row_errors:
                errors.append({'case': case.name, 'errors': row_errors})
            if len(records) % 100 == 0:
                atomic_json(output.with_suffix('.progress.json'), {
                    'completed': len(records), 'total': total, 'errors': len(errors)})
                print(f'data audit {len(records)}/{total}, errors={len(errors)}', flush=True)
    payload = {'schema_version': 1, 'status': 'passed' if not errors else 'failed',
               'splits': splits, 'case_count': len(records), 'parent_count': len(groups),
               'unique_electrical_inputs': len(fingerprints), 'errors': errors,
               'active_validation_computations': len(cache),
               'registry': registry.to_dict(), 'input_card_dependencies': INPUT_CARDS,
               'dataset_digest': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
               'records': records, 'psasp_recomputed': False}
    atomic_json(output, payload)
    if errors:
        raise ValueError(f'Dataset audit failed: {len(errors)} cases; see {output}')
    return payload
