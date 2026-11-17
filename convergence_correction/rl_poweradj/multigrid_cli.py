"""Explicit cross-grid entry point; formal runs fail closed until release acceptance."""
import argparse
import json
from pathlib import Path
from .case_adapter import CaseAdapter
from .data_preflight import input_card_hashes
from .foundation import UnifiedFrozenEncoder
from .grid_manifest import audit_manifest
from .paths import discover_cases
from .run_store import atomic_json

def main(argv=None):
    p=argparse.ArgumentParser(description='Variable-grid frozen-encoder development tools')
    commands=p.add_subparsers(dest='command',required=True)
    inspect=commands.add_parser('inspect')
    inspect.add_argument('--cases-dir',required=True,type=Path)
    inspect.add_argument('--output',required=True,type=Path)
    inspect.add_argument('--device',default='cpu')
    audit=commands.add_parser('audit')
    audit.add_argument('--manifest',required=True,type=Path)
    audit.add_argument('--require-training',action='store_true')
    train=commands.add_parser('train')
    train.add_argument('--manifest',required=True,type=Path)
    train.add_argument('--run-dir',required=True,type=Path)
    train.add_argument('--solver',required=True,type=Path)
    train.add_argument('--config',required=True,type=Path)
    train.add_argument('--readiness',required=True,type=Path)
    train.add_argument('--resume',type=Path)
    source_accept=commands.add_parser('source-acceptance')
    source_accept.add_argument('--manifest',required=True,type=Path)
    source_accept.add_argument('--run-dir',required=True,type=Path)
    source_accept.add_argument('--solver',required=True,type=Path)
    source_accept.add_argument('--config',required=True,type=Path)
    source_accept.add_argument('--cases',required=True,type=int)
    source_accept.add_argument('--resume',type=Path)
    evaluation=commands.add_parser('evaluate')
    evaluation.add_argument('--publication',type=Path,required=True)
    evaluation.add_argument('--manifest',type=Path,required=True)
    evaluation.add_argument('--solver',type=Path,required=True)
    evaluation.add_argument('--output',type=Path,required=True)
    evaluation.add_argument('--phase',choices=['test','external_test'],default='external_test')
    evaluation.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    evaluation.add_argument('--resume',action='store_true')
    multifamily_evaluation=commands.add_parser('evaluate-multifamily')
    multifamily_evaluation.add_argument('--publication',type=Path,required=True)
    multifamily_evaluation.add_argument('--manifest',type=Path,required=True)
    multifamily_evaluation.add_argument('--solver',type=Path,required=True)
    multifamily_evaluation.add_argument('--output',type=Path,required=True)
    multifamily_evaluation.add_argument('--phase',choices=['seen_family_test','external_test'],required=True)
    multifamily_evaluation.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    multifamily_evaluation.add_argument('--resume',action='store_true')
    multifamily_evaluation.add_argument('--pause-request',type=Path)
    multifamily_train=commands.add_parser('train-multifamily')
    multifamily_train.add_argument('--manifest',type=Path,required=True)
    multifamily_train.add_argument('--run-dir',type=Path,required=True)
    multifamily_train.add_argument('--solver',type=Path,required=True)
    multifamily_train.add_argument('--config',type=Path,required=True)
    multifamily_train.add_argument('--readiness',type=Path,required=True)
    multifamily_train.add_argument('--resume',type=Path)
    multifamily_acceptance=commands.add_parser('multifamily-acceptance')
    multifamily_acceptance.add_argument('--manifest',type=Path,required=True)
    multifamily_acceptance.add_argument('--run-dir',type=Path,required=True)
    multifamily_acceptance.add_argument('--solver',type=Path,required=True)
    multifamily_acceptance.add_argument('--config',type=Path,required=True)
    multifamily_acceptance.add_argument('--cases-per-family',type=int,default=1)
    multifamily_acceptance.add_argument('--resume',type=Path)
    multifamily_acceptance.add_argument('--pause-after-episode',type=int)
    acceptance=commands.add_parser('acceptance-train')
    acceptance.add_argument('--manifest',required=True,type=Path)
    acceptance.add_argument('--run-dir',required=True,type=Path)
    acceptance.add_argument('--solver',required=True,type=Path)
    acceptance.add_argument('--total-episodes',type=int,default=3)
    acceptance.add_argument('--max-steps',type=int,default=5)
    acceptance.add_argument('--seed',type=int,default=0)
    acceptance.add_argument('--resume',type=Path)
    args=p.parse_args(argv)
    if args.command=='multifamily-acceptance':
        from .graph_recipe import GraphRecipe
        from .multifamily_training import train_multifamily
        train_multifamily(Path(__file__).resolve().parents[1],args.manifest,args.run_dir,args.solver,
                          GraphRecipe.load(args.config),resume=args.resume,acceptance=True,
                          cases_per_family=args.cases_per_family,
                          pause_after_episode=args.pause_after_episode)
        return 0
    if args.command=='train-multifamily':
        from .graph_recipe import GraphRecipe
        from .multifamily_training import train_multifamily
        train_multifamily(Path(__file__).resolve().parents[1],args.manifest,args.run_dir,args.solver,
                          GraphRecipe.load(args.config),readiness=args.readiness,resume=args.resume)
        return 0
    if args.command=='evaluate-multifamily':
        from .multifamily_evaluation import evaluate_multifamily_published
        evaluate_multifamily_published(Path(__file__).resolve().parents[1],args.publication,args.manifest,
                                       args.solver,args.output,phase=args.phase,device=args.device,
                                       resume=args.resume,pause_request=args.pause_request)
        return 0
    if args.command=='evaluate':
        from .graph_evaluation import evaluate_published
        evaluate_published(Path(__file__).resolve().parents[1],args.publication,args.manifest,args.solver,
                           args.output,args.phase,args.device,args.resume)
        return 0
    if args.command in ('train','source-acceptance'):
        from .source_training import train_source
        from .graph_recipe import GraphRecipe
        train_source(Path(__file__).resolve().parents[1],args.manifest,args.run_dir,args.solver,
                     GraphRecipe.load(args.config),readiness=getattr(args,'readiness',None),resume=args.resume,
                     acceptance=args.command=='source-acceptance',acceptance_cases=getattr(args,'cases',0))
        return 0
    if args.command=='acceptance-train':
        from .graph_training import acceptance_train
        acceptance_train(Path(__file__).resolve().parents[1],args.manifest,args.run_dir,
                         args.solver,args.total_episodes,args.max_steps,args.seed,args.resume)
        return 0
    if args.command=='audit':
        result=audit_manifest(args.manifest,args.require_training)
        print(json.dumps({'digest':result['digest'],'case_count':len(result['cases']),
                          'families':result['families']},ensure_ascii=False));return 0
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    encoder=UnifiedFrozenEncoder.from_project(root,args.device)
    rows=[]
    for case in discover_cases(args.cases_dir):
        before=input_card_hashes(case)
        adapter=CaseAdapter(case);graph=encoder.encode_graph(case)
        if before!=input_card_hashes(case):raise RuntimeError('Source changed during inspection')
        rows.append({'case_id':case.name,'source':str(case),'grid_family_id':'',
                     'parent_case_id':'','split':'quarantine','input_cards':before,
                     'policy':adapter.proposal(),'bus_count':adapter.n,
                     'physical_edges':len(graph.edges),'possible_actions':2*len(adapter.devices),
                     'encoder_graph_key':graph.key})
    encoder.verify_frozen()
    atomic_json(args.output,{'schema':1,'split_protocol':'single_source_zero_shot','source_family_id':'',
                            'purpose':'discovery_only_not_authorization','encoder':encoder.graph_contract,'cases':rows})
    print(json.dumps({'cases':len(rows),'output':str(args.output),'status':'quarantine'},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
