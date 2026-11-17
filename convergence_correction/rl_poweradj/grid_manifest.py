"""Single-source 2000-bus learning; disjoint parents and zero-shot target grids."""
import json
from pathlib import Path
import numpy as np
from .case_adapter import CaseAdapter
from .grid_types import digest
from .data_preflight import input_card_hashes
from .run_store import file_hash
from .psasp import read_records
from .paths import case_input_fingerprint

def case_validation_key(case,cards):
    rows=read_records(Path(case)/'LF.L2')
    canonical=[]
    for row in rows:
        if len(row)<4 or type(row[0]) not in (int,float) or row[0] not in (0,1):
            raise ValueError('Unsupported L2 row/Mark')
        row=list(row)
        if abs(int(row[1]))==abs(int(row[2])):row[0]=0
        canonical.append(row)
    return digest({'cards':{k:v for k,v in cards.items() if k!='LF.L2'},'l2':canonical})

def audit_manifest(path, require_training=False):
    path=Path(path).resolve()
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!=1 or data.get('split_protocol')!='single_source_zero_shot':
        raise ValueError('Explicit single_source_zero_shot manifest required; migrate old family_holdout manifests')
    source_family=data.get('source_family_id')
    label_policy=data.get('label_policy','fresh_required')
    if label_policy not in ('fresh_required','historical_allowed_reset_fresh'):raise ValueError('Unknown label policy')
    if not isinstance(source_family,str) or not source_family.strip():
        raise ValueError('Explicit source_family_id required')
    seen=set(); families={};parents={};fingerprints={};records=[];adapters={};historical_manifests={}
    for item in data['cases']:
        for field in ('case_id','grid_family_id','parent_case_id','split','source','policy','input_cards'):
            if field not in item:raise ValueError(f'Missing manifest field: {field}')
        identity=item['case_id'];family=item['grid_family_id'];split=item['split']
        if not identity or identity in seen or not family or not item['parent_case_id']:
            raise ValueError('Missing/duplicate provenance identity')
        if split not in ('train','validation','test','external_test','quarantine'):raise ValueError('Unknown split')
        if split in ('train','validation','test') and family!=source_family:
            raise ValueError('Target grid cannot enter source training/validation/test')
        if split=='external_test' and family==source_family:
            raise ValueError('External test must be an unseen target family')
        seen.add(identity)
        case=(path.parent/item['source']).resolve()
        cards=input_card_hashes(case)
        if not cards or cards!=item['input_cards']:raise ValueError(f'Stale input cards: {identity}')
        fp=digest(cards)
        if split!='quarantine':
            for key,mapping in (((family,item['parent_case_id']),parents),(fp,fingerprints)):
                if key in mapping and mapping[key]!=split:raise ValueError('Family/parent/input leakage')
                mapping[key]=split
            families.setdefault(family,[])
            if split not in families[family]:families[family].append(split)
        key=case_validation_key(case,cards)
        if key not in adapters:adapters[key]=CaseAdapter(case)
        adapter=adapters[key]
        if split in ('train','validation','test') and adapter.n!=2000:
            raise ValueError('Source learning/selection is restricted to the declared 2000-bus grid')
        policy=item['policy']
        if policy.get('approved') is True:adapter.authorize(policy)
        elif split in ('train','validation','test'):raise ValueError('Source cases require approved controls')
        label_status='unverified'
        if item.get('label_evidence'):
            evidence=item['label_evidence']
            label_path=(path.parent/evidence['path']).resolve()
            if file_hash(label_path)!=evidence['sha256']:raise ValueError('Changed solver-label evidence')
            label=json.loads(label_path.read_text(encoding='utf-8'))
            expected_label='converged' if label['converged'] else 'nonconverged'
            if label['input_cards']!=cards or item.get('flow_label',item.get('fresh_label'))!=expected_label:
                raise ValueError('Solver label does not match source inputs')
            historical=label.get('fresh_invocation') is False
            label_status='historical' if historical else 'recomputed'
            if historical:
                if 'fresh_label' in item:raise ValueError('Historical label may not be advertised as fresh')
                if label_policy!='historical_allowed_reset_fresh' or label.get('label_source')!='historical_manifest':
                    raise ValueError('Historical labels not allowed by this manifest')
                hist_path=Path(label['historical_manifest_path']).resolve()
                hist_identity=(hist_path,label['historical_manifest_sha256'])
                if hist_identity not in historical_manifests:
                    if file_hash(hist_path)!=label['historical_manifest_sha256']:raise ValueError('Historical label source changed')
                    historical_manifests[hist_identity]={r['case']:r for r in json.loads(hist_path.read_text(encoding='utf-8'))['records']}
                prior=historical_manifests[hist_identity][label['historical_case']]
                if bool(prior['first_ac_converged'])!=label['converged'] or prior['input_fingerprint']!=case_input_fingerprint(case).hex():
                    raise ValueError('Historical label does not match original electrical input')
            elif not label.get('solver_sha256'):raise ValueError('Unbound solver identity')
        records.append({**item,'source':str(case),'fingerprint':fp,'bus_count':adapter.n,'validation_key':key,
                        'flow_label':item.get('flow_label',item.get('fresh_label')),'label_status':label_status})
    if require_training:
        for split in ('train','validation'):
            if not any(r['split']==split for r in records):raise ValueError(f'Missing {split} cases')
        if not all(r.get('flow_label') in ('converged','nonconverged') and r.get('label_evidence')
                   for r in records if r['split'] in ('train','validation','test')):
            raise ValueError('Traceable solver-label evidence required before formal training')
        for split in ('train','validation'):
            if not any(r['split']==split and r.get('flow_label')=='nonconverged' for r in records):
                raise ValueError(f'No nonconverged cases in {split}')
    return {'cases':records,'digest':digest(data),'families':families,
            'source_family_id':source_family,'split_protocol':data['split_protocol'],'label_policy':label_policy}

class FamilySampler:
    def __init__(self,records,seed=0):
        self.groups={}
        for i,r in enumerate(records):
            if r['split']=='train':self.groups.setdefault(r['grid_family_id'],{}).setdefault(r['parent_case_id'],[]).append(i)
        if not self.groups:raise ValueError('No training families')
        if len(self.groups)!=1:raise ValueError('Only one source training family is allowed')
        self.families=sorted(self.groups)
        self.rng=np.random.default_rng(seed)

    def next(self):
        f=self.families[int(self.rng.integers(len(self.families)))]
        parents=sorted(self.groups[f]);p=parents[int(self.rng.integers(len(parents)))]
        children=self.groups[f][p]
        return children[int(self.rng.integers(len(children)))]

    def state_dict(self):return self.rng.bit_generator.state
    def load_state_dict(self,state):self.rng.bit_generator.state=state
