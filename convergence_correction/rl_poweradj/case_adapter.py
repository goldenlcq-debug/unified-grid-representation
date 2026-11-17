"""Case-local physical addressing and byte-preserving authorized Mark edits."""
from dataclasses import dataclass
from pathlib import Path
import os
import re
import numpy as np
from .grid_types import digest
from .model import (read_table, read_generator_table, read_dc_plans, is_shunt,
                    L2_FIELDS, L3_FIELDS, L6_FIELDS)
from .psasp import read_records
from .input_validation import InputCaseValidator

@dataclass(frozen=True)
class Device:
    key: str
    row: int
    edge: int
    signature: str
    name: str

class CaseAdapter:
    def __init__(self, path):
        self.path = Path(path)
        self.n = len(read_records(self.path/'LF.L1'))
        if not self.n:
            raise ValueError('No buses')
        self.lines = read_table(self.path/'LF.L2', L2_FIELDS)
        # Explicit upstream-compatible L3 prefix: circuit ID precedes R1/X1.
        # Do not reuse the old flat-observation L3 schema (missing this column).
        self.transformers = read_table(self.path/'LF.L3',
            ('Mark','I_No','J_No','ID_No','R1','X1','Tk')) if (self.path/'LF.L3').exists() else []
        self.generators = read_generator_table(self.path/'LF.L5')
        self.loads = read_table(self.path/'LF.L6', L6_FIELDS)
        self.dc = read_dc_plans(self.path)
        self.rows = self.lines + self.transformers
        self.devices = {}
        keys = set()
        for i, row in enumerate(self.rows):
            card = 'LF.L2' if i < len(self.lines) else 'LF.L3'
            circuit = row['ID_No']
            key = f'{card}:{int(row["I_No"])}:{int(row["J_No"])}:{circuit}'
            if key in keys:
                raise ValueError(f'Ambiguous physical device key: {key}')
            keys.add(key)
            if int(row['Mark']) not in (0, 1):
                raise ValueError(f'Unsupported Mark: {key}')
            for field in ('I_No', 'J_No'):
                if not 1 <= abs(int(row[field])) <= self.n:
                    raise ValueError(f'Invalid bus endpoint: {key}')
            if card == 'LF.L2' and is_shunt(row):
                self.devices[key] = Device(key, i, i, digest(row.values[1:]), row.identifier)
        validation = InputCaseValidator().validate(self.path)
        if not validation.active_power_valid:
            raise ValueError('; '.join(validation.errors))
        self.validation = validation

    def proposal(self):
        """Discovery only. This is deliberately not an authorization."""
        return {'schema': 1, 'approved': False,
                'devices': [{'key':d.key, 'signature':d.signature, 'targets':[0, 1]}
                            for d in self.devices.values()]}

    def authorize(self, policy):
        if policy.get('schema') != 1 or policy.get('approved') is not True:
            raise ValueError('Explicit approved physical-device policy required')
        candidates = []
        seen = set()
        for item in policy['devices']:
            key = item['key']
            if key in seen or key not in self.devices:
                raise ValueError(f'Duplicate or unknown authorized device: {key}')
            seen.add(key)
            d = self.devices[key]
            if item['signature'] != d.signature:
                raise ValueError(f'Physical device signature changed: {key}')
            targets = item['targets']
            if not targets or any(type(t) is not int or t not in (0, 1) for t in targets) or len(set(targets)) != len(targets):
                raise ValueError('Invalid allowed targets')
            candidates.extend((d, t) for t in sorted(targets))
        return sorted(candidates, key=lambda x:(x[0].key, x[1]))

    def apply(self, device, target):
        if type(target) is not int or target not in (0, 1):
            raise ValueError('Invalid target')
        current = self.devices.get(device.key)
        if current != device:
            raise ValueError('Stale device locator')
        path = self.path/'LF.L2'
        contents = path.read_bytes().splitlines(keepends=True)
        physical = [i for i, line in enumerate(contents) if line.strip()]
        fresh = read_records(path)
        if len(fresh) != len(self.lines) or digest(fresh[device.row][1:]) != device.signature:
            raise ValueError('Source device changed before edit')
        i = physical[device.row]
        changed, count = re.subn(rb'^(\s*)[01](\s*,)',
                                lambda m:m[1]+str(target).encode()+m[2], contents[i], count=1)
        if count != 1:
            raise ValueError('Unsupported Mark lexical format')
        contents[i] = changed
        temporary = path.with_suffix('.L2.pending')
        with temporary.open('xb') as stream:
            stream.write(b''.join(contents)); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
        self.lines[device.row]['Mark'] = target

    def dynamic(self):
        nodes = np.zeros((self.n, 8), np.float32)
        for row in self.generators:
            if int(row['Mark']) == 1:
                i = abs(int(row['Bs_No']))-1
                nodes[i,0] += float(row['Pg'])
                nodes[i,3] += 1
                nodes[i,5] = max(nodes[i,5], float(int(row['Type']) == 0))
        for row in self.loads:
            if int(row['Mark']) == 1:
                i = abs(int(row['Bs_No']))-1
                nodes[i,1:3] += [float(row['Pl']), float(row['Ql'])]
                nodes[i,4] += 1
        for plan in self.dc:
            nodes[plan.from_bus,0] += plan.from_injection
            nodes[plan.to_bus,0] += plan.to_injection
        for island in self.validation.islands:
            nodes[list(island.buses),6] = island.energized
            nodes[list(island.buses),7] = island.has_online_reference
        edges = np.zeros((len(self.rows), 5), np.float32)
        for k, row in enumerate(self.rows):
            line = k < len(self.lines)
            edges[k] = [int(row['Mark']),float(line and is_shunt(row)),
                        float(row['R'] if line else row['R1']),
                        float(row['X'] if line else row['X1']),
                        float(row['B_half']) if line else 0.]
        return nodes, edges
