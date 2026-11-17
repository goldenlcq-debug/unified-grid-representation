"""Bounded single-source training/validation acceptance with boundary resume.

Formal release deliberately remains separate from this engineering harness.
"""
from pathlib import Path
import tempfile
import json
import uuid
import time
import copy
import numpy as np
import torch
from .foundation import UnifiedFrozenEncoder
from .graph_environment import GraphPowerAdjustmentEnv
from .graph_dqn import GraphDQN,GraphReplay,GraphTransition
from .grid_manifest import audit_manifest,FamilySampler
from .grid_types import GRAPH_SCHEMA
from .run_store import prepare_run,file_hash,atomic_json
from .lifecycle import run_lock,set_status
from .solver_lock import SolverLock
from .readiness import code_manifest,runtime_identity
from .graph_recipe import GraphRecipe
from .graph_release import verify_release,retain_checkpoints
from .psasp import PSASPRunner
from .evaluation_metrics import summarize_action_steps, STEP_METRIC_DEFINITIONS

class GraphEvaluationPaused(RuntimeError):
    pass

def evaluate_records(agent,encoder,runner,records,horizon,pause_requested=None,on_case=None):
    rows=[]
    with GraphPowerAdjustmentEnv(encoder,runner,horizon) as env:
        for record in records:
            if pause_requested and pause_requested():raise GraphEvaluationPaused('Incomplete validation is not eligible for selection')
            state,info=env.reset(record['source'],record['policy'],expected_inputs=record.get('input_cards'))
            initial=info['converged']
            while not env.done:
                action=agent.choose(state)
                if action is None:break
                state,_,_,_,info=env.step(action)
            rows.append({'case_id':record['case_id'],'family':record['grid_family_id'],
                         'initial_converged':initial,**info})
            if on_case:on_case(rows[-1])
    families=sorted({r['family'] for r in rows})
    rates=[]
    for f in families:
        required=[r for r in rows if r['family']==f and not r['initial_converged']]
        if required:rates.append(sum(r['converged'] for r in required)/len(required))
    required=[r for r in rows if not r['initial_converged']]
    metrics=summarize_action_steps(rows)
    return {'cases':rows,**metrics,
            'micro_success_rate':sum(r['converged'] for r in required)/len(required) if required else None,
            'macro_success_rate':sum(rates)/len(rates) if rates else None,
            # Preserve the old all-required population for existing ranking callers.
            'mean_required_steps':metrics['mean_all_required_steps'],
            'step_metric_definitions':dict(STEP_METRIC_DEFINITIONS),
            'families':{f:summarize_action_steps([r for r in rows if r['family']==f]) for f in families}}

def acceptance_train(root,manifest,run_dir,solver,total_episodes=3,horizon=5,seed=0,resume=None):
    if not 1<=total_episodes<=20 or not 1<=horizon<=20:
        raise ValueError('Acceptance only: maximum 20 episodes and 20 steps')
    root,run_dir=Path(root).resolve(),Path(run_dir).resolve()
    if not run_dir.is_relative_to(root/'artifacts'/'runs'):
        raise ValueError('Run must be under artifacts/runs')
    audit=audit_manifest(manifest)
    records=audit['cases']
    sampler=FamilySampler(records,seed)
    contract={'purpose':'acceptance_only','schema':GRAPH_SCHEMA,'dataset':audit['digest'],
              'split_protocol':audit['split_protocol'],'source_family_id':audit['source_family_id'],
              'solver_sha256':file_hash(solver),'code':code_manifest(root),
              'horizon':horizon,'seed':seed,'total_episodes':total_episodes,'device':'cpu'}
    lock=Path(tempfile.gettempdir())/'rl_poweradj_training.lock'
    with SolverLock(lock),run_lock(run_dir):
        prepare_run(run_dir,contract,resume is not None)
        try:
            encoder=UnifiedFrozenEncoder.from_project(root,'cpu')
            if resume:
                checkpoint=Path(resume).resolve()
                if not checkpoint.is_relative_to(run_dir):raise ValueError('Resume checkpoint outside run')
                agent,replay,progress,_=GraphDQN.load(checkpoint,encoder,expected_run_contract=contract)
                sampler.load_state_dict(progress['sampler'])
            else:
                agent=GraphDQN(encoder,seed=seed);replay=GraphReplay(1000)
                progress={'episodes':0,'transitions':0,'successes':0,'skipped_converged':0,'lengths':[],'losses':[]}
            runner=PSASPRunner(solver,audit_dir=run_dir/'solver')
            if (run_dir/'pause.request').exists():
                raise ValueError('Clear the acknowledged pause.request before explicit resume')
            set_status(run_dir,'running',purpose='acceptance_only')
            with GraphPowerAdjustmentEnv(encoder,runner,horizon) as env:
                while progress['episodes']<total_episodes:
                    record=records[sampler.next()]
                    state,info=env.reset(record['source'],record['policy'],expected_inputs=record.get('input_cards'))
                    if info['converged']:progress['skipped_converged']+=1
                    while not env.done:
                        action=agent.choose(state,epsilon=max(.05,1-progress['transitions']/10000))
                        if action is None:break
                        next_state,reward,term,trunc,info=env.step(action)
                        replay.add(GraphTransition(state,action,reward,next_state,term or trunc,
                                                   record['grid_family_id'],record['case_id']))
                        progress['transitions']+=1
                        loss=agent.learn(replay.sample(agent.rng,min(4,len(replay.items))))
                        progress['losses'].append(loss)
                        if term:progress['successes']+=1
                        state=next_state
                    progress['lengths'].append(env.steps);progress['episodes']+=1
                    progress['sampler']=sampler.state_dict()
                    checkpoint=run_dir/f'episode-{progress["episodes"]}.pt'
                    agent.save(checkpoint,replay,progress,contract)
                    set_status(run_dir,'running',completed_episodes=progress['episodes'],
                               transitions=progress['transitions'],checkpoint=str(checkpoint))
                    if (run_dir/'pause.request').exists():
                        set_status(run_dir,'paused');return progress
            validation=evaluate_records(agent,encoder,runner,
                [r for r in records if r['split']=='validation'],horizon)
            encoder.verify_frozen()
            atomic_json(run_dir/'acceptance.json',{'purpose':'engineering_not_formal',
                'progress':progress,'validation':validation,'representation':agent.audit})
            set_status(run_dir,'acceptance_complete')
            return progress
        except BaseException as exc:
            set_status(run_dir,'failed',error_type=type(exc).__name__,error=str(exc))
            raise
