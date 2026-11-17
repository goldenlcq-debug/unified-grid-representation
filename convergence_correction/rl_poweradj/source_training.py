"""Shared production/acceptance lifecycle for single-source frozen graph RL."""
import copy
import json
from pathlib import Path
import tempfile
import time
import uuid
import torch
from .foundation import UnifiedFrozenEncoder
from .graph_environment import GraphPowerAdjustmentEnv
from .graph_dqn import GraphDQN,GraphReplay,GraphTransition
from .graph_training import evaluate_records,GraphEvaluationPaused
from .grid_manifest import audit_manifest,FamilySampler
from .graph_recipe import GraphRecipe
from .graph_release import verify_release,retain_checkpoints
from .grid_types import GRAPH_SCHEMA
from .readiness import code_manifest,runtime_identity
from .run_store import prepare_run,file_hash,atomic_json
from .lifecycle import run_lock,set_status
from .solver_lock import SolverLock
from .psasp import PSASPRunner

def train_source(root,manifest,run_dir,solver,recipe,*,readiness=None,resume=None,
                 acceptance=False,acceptance_cases=0,pause_after_episode=None):
    """Never consumes test/external data for updates or model selection.

    pause_after_episode is a deterministic lifecycle fault-injection hook for
    acceptance only. It does not alter the experiment/checkpoint contract.
    """
    root,run_dir=Path(root).resolve(),Path(run_dir).resolve()
    if not isinstance(recipe,GraphRecipe):raise ValueError('GraphRecipe required')
    if not run_dir.is_relative_to(root/'artifacts'/'runs'):raise ValueError('Run outside artifacts/runs')
    if acceptance:
        if recipe.total_episodes>100 or recipe.horizon>150 or not 1<=acceptance_cases<=100:
            raise ValueError('Acceptance bounded to 100 episodes/cases and 150 steps')
        release=None
    else:
        if acceptance_cases or pause_after_episode is not None:raise ValueError('Acceptance hooks forbidden in formal run')
        release=verify_release(root,readiness,manifest,solver,recipe)
    audited=audit_manifest(manifest,require_training=True)
    # A training invocation never admits target records, even as unused extras.
    records=[r for r in audited['cases'] if r['split'] in ('train','validation')]
    if acceptance:
        records=[r for split in ('train','validation')
                 for r in [x for x in audited['cases'] if x['split']==split][:acceptance_cases]]
    solver_hash=file_hash(solver)
    for r in records:
        label_path=Path(manifest).resolve().parent/r['label_evidence']['path']
        label=json.loads(label_path.read_text(encoding='utf-8'))
        if label.get('fresh_invocation') is False and audited.get('label_policy')=='historical_allowed_reset_fresh':
            continue  # Historical classification is explicit; env.reset always freshly solves.
        if label['solver_sha256']!=solver_hash:
            raise ValueError('Source labels were produced by a different solver')
    contract={'purpose':'acceptance_only' if acceptance else 'formal_source_training',
        'schema':GRAPH_SCHEMA,'protocol':audited['split_protocol'],'source_family_id':audited['source_family_id'],
        'dataset':audited['digest'],'recipe':recipe.to_dict(),'solver_sha256':solver_hash,
        'code':code_manifest(root),'runtime':runtime_identity(),'readiness':release,
        'accepted_case_ids':[r['case_id'] for r in records] if acceptance else None}
    sampler=FamilySampler(records,recipe.seed)
    validation=[r for r in records if r['split']=='validation']
    if not validation:raise ValueError('Source validation required')
    stop_path=run_dir/'pause.request'
    with SolverLock(Path(tempfile.gettempdir())/'rl_poweradj_training.lock'),run_lock(run_dir):
        prepare_run(run_dir,contract,resume is not None)
        if stop_path.exists():raise ValueError('Acknowledge/remove pause.request before explicit resume')
        if (run_dir/'selected.json').exists():raise ValueError('Completed run is immutable')
        started=time.monotonic()
        try:
            encoder=UnifiedFrozenEncoder.from_project(root,recipe.device)
            if recipe.device=='cuda':torch.cuda.reset_peak_memory_stats()
            if resume:
                checkpoint=Path(resume).resolve()
                if not checkpoint.is_relative_to(run_dir):raise ValueError('Checkpoint outside owning run')
                agent,replay,progress,_=GraphDQN.load(checkpoint,encoder,device=recipe.device,expected_run_contract=contract)
                if progress['transitions']>0 and not replay.items:
                    raise ValueError('Inference publication is not a resumable training checkpoint')
                if progress['updates']!=agent.updates:
                    raise ValueError('Checkpoint update counters differ')
                sampler.load_state_dict(progress['sampler'])
                if progress['episodes']>recipe.total_episodes:raise ValueError('Resume exceeds episode budget')
            else:
                agent=GraphDQN(encoder,recipe.device,recipe.width,recipe.layers,recipe.seed,
                               recipe.gamma,recipe.learning_rate,recipe.target_interval)
                replay=GraphReplay(recipe.replay_capacity)
                progress={'episodes':0,'transitions':0,'updates':0,'successes':0,'skipped_converged':0,
                    'lengths':[],'losses':[],'batch_sizes':[],'validation_history':[],
                    'validated_episode':-1,'best_rank':None,'best_model':None,'sampler':sampler.state_dict()}
            runner=PSASPRunner(solver,audit_dir=run_dir/'solver')
            agent.training_cases={(r['grid_family_id'],r['case_id']) for r in records if r['split']=='train'}
            def save_checkpoint():
                progress['sampler']=copy.deepcopy(sampler.state_dict())
                checkpoint=run_dir/'checkpoints'/f'episode-{progress["episodes"]}-{uuid.uuid4().hex[:8]}.pt'
                agent.save(checkpoint,replay,progress,contract)
                removed=retain_checkpoints(checkpoint.parent,recipe.checkpoint_keep,checkpoint)
                set_status(run_dir,'training',completed_episodes=progress['episodes'],
                           transitions=progress['transitions'],updates=agent.updates,
                           successes=progress['successes'],skipped_converged=progress['skipped_converged'],
                           checkpoint=str(checkpoint),removed_older_checkpoints=removed)
                return checkpoint
            def validate():
                result=evaluate_records(agent,encoder,runner,validation,recipe.horizon,stop_path.exists)
                progress['validation_history'].append({'episode':progress['episodes'],**result})
                progress['validated_episode']=progress['episodes']
                if result['action_required']:
                    rank=[result['micro_success_rate'],-result['mean_required_steps']]
                    if progress['best_rank'] is None or rank>progress['best_rank']:
                        model=run_dir/'models'/f'best-{progress["episodes"]}-{uuid.uuid4().hex[:8]}.pt'
                        agent.save(model,GraphReplay(1),progress,contract)
                        progress['best_rank']=rank;progress['best_model']=str(model)
                atomic_json(run_dir/'validation.json',result)
            set_status(run_dir,'training',purpose=contract['purpose'],pending_validation=False)
            if progress['episodes'] and progress['validated_episode']!=progress['episodes'] and (
                    progress['episodes']%recipe.evaluation_interval==0 or progress['episodes']==recipe.total_episodes):
                try:validate()
                except GraphEvaluationPaused:
                    save_checkpoint();set_status(run_dir,'paused',pending_validation=True);return progress
            with GraphPowerAdjustmentEnv(encoder,runner,recipe.horizon) as env:
                while progress['episodes']<recipe.total_episodes:
                    if stop_path.exists():
                        save_checkpoint();set_status(run_dir,'paused');return progress
                    record=records[sampler.next()]
                    state,info=env.reset(record['source'],record['policy'],expected_inputs=record['input_cards'])
                    if info['converged']:progress['skipped_converged']+=1
                    while not env.done:
                        epsilon=max(.05,1-.95*progress['transitions']/recipe.epsilon_decay)
                        action=agent.choose(state,epsilon)
                        if action is None:raise RuntimeError('Nonterminal environment has no action')
                        ns,reward,term,trunc,info=env.step(action)
                        with (run_dir/'transitions.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps({'episode':progress['episodes']+1,'source':record['case_id'],
                                'family':record['grid_family_id'],'step':env.steps,'action':int(action),
                                'device_key':state.device_keys[action],'target':int(state.targets[action]),
                                'reward':reward,'terminated':term,'truncated':trunc,'epsilon':epsilon,
                                'resume_checkpoint':str(resume) if resume else None},allow_nan=False)+'\n')
                        replay.add(GraphTransition(state,action,reward,ns,term or trunc,
                                                   record['grid_family_id'],record['case_id']))
                        progress['transitions']+=1
                        if len(replay.items)>=recipe.warmup:
                            batch=replay.sample(agent.rng,recipe.batch_size,recipe.node_budget,
                                                recipe.edge_budget,recipe.action_budget)
                            loss=agent.learn(batch)
                            if loss is None:raise RuntimeError('Empty batch under resource budget')
                            progress['losses'].append(loss);progress['batch_sizes'].append(len(batch))
                        if term:progress['successes']+=1
                        state=ns
                    progress['lengths'].append(env.steps);progress['episodes']+=1;progress['updates']=agent.updates
                    progress['sampler']=copy.deepcopy(sampler.state_dict())
                    paused=stop_path.exists() or (pause_after_episode is not None and progress['episodes']>=pause_after_episode)
                    if paused:
                        save_checkpoint();set_status(run_dir,'paused');return progress
                    if progress['episodes']%recipe.evaluation_interval==0 or progress['episodes']==recipe.total_episodes:
                        try:validate()
                        except GraphEvaluationPaused:
                            save_checkpoint();set_status(run_dir,'paused',pending_validation=True);return progress
                    if progress['episodes']%recipe.checkpoint_interval==0 or progress['episodes']==recipe.total_episodes:
                        save_checkpoint()
                    atomic_json(run_dir/'representation.json',agent.audit)
            # A final-boundary paused run must still perform selection on resume.
            if progress['validated_episode']!=progress['episodes']:
                try:validate()
                except GraphEvaluationPaused:
                    save_checkpoint();set_status(run_dir,'paused',pending_validation=True);return progress
            if progress['best_model'] is None:raise RuntimeError('No action-required validation cases for selection')
            encoder.verify_frozen()
            save_checkpoint()
            # Immutable unique payloads + selected.json as the commit point.
            # A partial/orphan payload from an interrupted publication never
            # blocks resume and is never mistaken for the committed model.
            final=run_dir/'models'/f'final-{uuid.uuid4().hex[:8]}.pt'
            agent.save(final,GraphReplay(1),progress,contract)
            selected={'schema':1,'purpose':contract['purpose'],'protocol':'single_source_zero_shot',
                'model':progress['best_model'],'model_sha256':file_hash(progress['best_model']),
                'final_model':str(final),'selection':'source_validation_only','best_rank':progress['best_rank'],
                'encoder':encoder.graph_contract,'solver_sha256':solver_hash,'recipe':recipe.to_dict(),
                'source_family_id':audited['source_family_id'],'formal_training_started':not acceptance}
            atomic_json(run_dir/'selected.json',selected)
            atomic_json(run_dir/'summary.json',{'status':'passed','purpose':contract['purpose'],
                'progress':progress,'representation':agent.audit,'encoder_frozen':True,
                'seconds_this_invocation':time.monotonic()-started,
                'cuda_peak_allocated':torch.cuda.max_memory_allocated() if recipe.device=='cuda' else None,
                'external_test_used':False,'formal_training_started':not acceptance})
            set_status(run_dir,'acceptance_complete' if acceptance else 'training_complete')
            return progress
        except BaseException as exc:
            set_status(run_dir,'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                       error_type=type(exc).__name__,error=str(exc))
            raise
