"""Content-bound release for the single-source candidate task head."""
import json
from pathlib import Path
from .readiness import code_manifest,runtime_identity
from .foundation import verify_foundation_assets,default_foundation_paths,_source_digest,APPROVED_GRIDFM_SHA256
from .run_store import file_hash
from .grid_types import GRAPH_SCHEMA

def release_identity(root,manifest,solver,recipe):
    assets,source=default_foundation_paths(root)
    if _source_digest(source)!=APPROVED_GRIDFM_SHA256:raise ValueError('Changed foundation source')
    return {'code':code_manifest(root),'runtime':runtime_identity(),'schema':GRAPH_SCHEMA,
            'manifest_sha256':file_hash(manifest),'solver_sha256':file_hash(solver),
            'recipe':recipe.to_dict(),'assets':verify_foundation_assets(assets),
            'protocol':'single_source_zero_shot'}

def verify_release(root,path,manifest,solver,recipe):
    if path is None:raise ValueError('Formal training requires candidate readiness')
    root=Path(root).resolve()
    release=json.loads(Path(path).read_text(encoding='utf-8'))
    if release.get('status')!='ready_for_training' or release.get('kind')!='candidate_source_release':
        raise ValueError('Not a released single-source candidate task head')
    if release.get('identity')!=release_identity(root,manifest,solver,recipe):
        raise ValueError('Stale candidate release identity')
    evidence=release.get('evidence',{})
    if set(evidence)!={'tests','data_audit','lifecycle','parity','prerun','resources'}:
        raise ValueError('Incomplete candidate release evidence')
    for item in evidence.values():
        file=(root/item['path']).resolve()
        if not file.is_relative_to(root) or file_hash(file)!=item['sha256']:
            raise ValueError('Changed release evidence')
    return {'path':str(Path(path).resolve()),'sha256':file_hash(path)}

def retain_checkpoints(directory,keep,current):
    """Remove only older verified checkpoints created inside this run directory."""
    directory=Path(directory).resolve();current=Path(current).resolve()
    if current.parent!=directory or keep<1:raise ValueError('Invalid retention scope')
    saved=[]
    for path in directory.glob('episode-*.pt'):
        try:
            info=json.loads(path.with_suffix('.pt.manifest.json').read_text())
            if info['schema']!=GRAPH_SCHEMA or info['sha256']!=file_hash(path):continue
            if path.resolve().parent!=directory:continue
            saved.append((int(info['episodes']),path))
        except (OSError,ValueError,KeyError,TypeError):continue
    current_episode=next((e for e,p in saved if p==current),None)
    if current_episode is None:raise ValueError('Current checkpoint failed integrity check')
    eligible=sorted(((e,p) for e,p in saved if e<=current_episode and p!=current),
                    key=lambda item:(item[0],item[1].stat().st_mtime_ns),reverse=True)
    removed=[]
    for _,path in eligible[max(0,keep-1):]:
        path.unlink();path.with_suffix('.pt.manifest.json').unlink();removed.append(path.name)
    return removed
