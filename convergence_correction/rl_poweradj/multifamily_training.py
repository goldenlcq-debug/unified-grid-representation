"""Formal four-family training on a sealed train/validation manifest."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time
import uuid

import torch

from .foundation import UnifiedFrozenEncoder
from .graph_dqn import GraphDQN, GraphTransition
from .graph_environment import GraphPowerAdjustmentEnv
from .graph_evaluation import state_hash
from .graph_recipe import GraphRecipe
from .graph_release import retain_checkpoints
from .graph_training import GraphEvaluationPaused, evaluate_records
from .grid_types import GRAPH_SCHEMA
from .lifecycle import run_lock, set_status
from .multifamily_manifest import audit_multifamily_manifest
from .multifamily_release import verify_release
from .multifamily_replay import MultiFamilyReplay, sample_balanced
from .multifamily_sampler import MultiFamilySampler, TRAIN_FAMILIES
from .multifamily_validation import rank_validation
from .psasp import PSASPRunner
from .readiness import code_manifest, runtime_identity
from .run_store import atomic_json, file_hash, prepare_run
from .solver_lock import SolverLock


def train_multifamily(root, manifest, run_dir, solver, recipe: GraphRecipe, *,
                      readiness=None, resume=None, acceptance=False, cases_per_family=0,
                      pause_after_episode=None):
    root, manifest, run_dir, solver = (Path(x).resolve() for x in (root, manifest, run_dir, solver))
    if not isinstance(recipe, GraphRecipe):
        raise ValueError("GraphRecipe required")
    if not run_dir.is_relative_to(root / "artifacts/runs"):
        raise ValueError("Run outside artifacts/runs")
    if acceptance:
        if not (1 <= recipe.total_episodes <= 20 and 1 <= recipe.horizon <= 20
                and 1 <= cases_per_family <= 3):
            raise ValueError("Multi-family acceptance has bounded episode/case/step limits")
        if pause_after_episode is not None and not (1 <= pause_after_episode < recipe.total_episodes):
            raise ValueError("Invalid acceptance-only pause episode")
        release = None
    else:
        if cases_per_family or pause_after_episode is not None:
            raise ValueError("Acceptance sampling is forbidden in formal training")
        release = verify_release(root, readiness, manifest, solver, recipe)
    audited = audit_multifamily_manifest(manifest, verify_files=not acceptance)
    sealed = json.loads(manifest.read_text(encoding="utf-8"))
    if sealed["status"] != "sealed_train_validation":
        raise ValueError("Formal training cannot read a test or development manifest")
    records = audited["cases"]
    if acceptance:
        records = [row for family in TRAIN_FAMILIES for split in ("train", "validation")
                   for row in [r for r in records
                               if r["grid_family_id"] == family and r["split"] == split][:cases_per_family]]
    validation = [row for row in records if row["split"] == "validation"]
    if {row["grid_family_id"] for row in validation} != set(TRAIN_FAMILIES):
        raise ValueError("Validation does not cover every training family")
    sampler = MultiFamilySampler(records, recipe.seed)
    solver_hash = file_hash(solver)
    contract = {
        "purpose": "multifamily_acceptance_only" if acceptance else "formal_multifamily_training",
        "protocol": "multi_family_joint_training_v1",
        "schema": GRAPH_SCHEMA, "training_family_ids": list(TRAIN_FAMILIES),
        "training_bus_counts": [118, 300, 2000, 3120],
        "dataset_sha256": audited["digest"],
        "seen_test_sha256": sealed["seen_test_sha256"],
        "exploration_policy": "legal_currently_off_shunt_to_on_preferred_for_random_actions",
        "recipe": recipe.to_dict(), "solver_sha256": solver_hash,
        "code": code_manifest(root), "runtime": runtime_identity(), "readiness": release,
        "accepted_case_ids": [row["case_id"] for row in records] if acceptance else None,
    }
    stop = run_dir / "pause.request"
    lock = Path(tempfile.gettempdir()) / "rl_poweradj_multifamily_training.lock"
    with SolverLock(lock), run_lock(run_dir):
        prepare_run(run_dir, contract, resume is not None)
        if stop.exists():
            raise ValueError("Acknowledge pause.request before explicit resume")
        if (run_dir / "selected.json").exists():
            raise ValueError("Published run is immutable")
        started = time.monotonic()
        try:
            encoder = UnifiedFrozenEncoder.from_project(root, recipe.device)
            if recipe.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if resume is not None:
                checkpoint = Path(resume).resolve()
                if not checkpoint.is_relative_to(run_dir):
                    raise ValueError("Checkpoint outside owning run")
                agent, replay, progress, _ = GraphDQN.load(
                    checkpoint, encoder, device=recipe.device, expected_run_contract=contract)
                if not isinstance(replay, MultiFamilyReplay) or not replay.items:
                    raise ValueError("Multi-family resume requires a full replay checkpoint")
                if progress["updates"] != agent.updates:
                    raise ValueError("Checkpoint update counters differ")
                sampler.load_state_dict(progress["sampler"])
            else:
                agent = GraphDQN(encoder, device=recipe.device, width=recipe.width,
                                 layers=recipe.layers, seed=recipe.seed, gamma=recipe.gamma,
                                 learning_rate=recipe.learning_rate,
                                 target_interval=recipe.target_interval)
                replay = MultiFamilyReplay(recipe.replay_capacity)
                progress = {"episodes": 0, "transitions": 0, "updates": 0,
                            "successes": 0, "family_successes": {family: 0 for family in TRAIN_FAMILIES},
                            "lengths": [], "losses": [],
                            "batch_sizes": [], "validation_history": [],
                            "validated_episode": -1, "best_rank": None,
                            "best_model": None, "sampler": sampler.state_dict()}
            if progress["episodes"] > recipe.total_episodes:
                raise ValueError("Resume exceeds episode budget")
            agent.training_cases = {(row["grid_family_id"], row["case_id"])
                                    for row in records if row["split"] == "train"}
            runner = PSASPRunner(solver, audit_dir=run_dir / "solver")

            def checkpoint_save():
                progress["sampler"] = copy.deepcopy(sampler.state_dict())
                checkpoint = run_dir / "checkpoints" / f"episode-{progress['episodes']}-{uuid.uuid4().hex[:8]}.pt"
                agent.save(checkpoint, replay, progress, contract)
                removed = retain_checkpoints(checkpoint.parent, recipe.checkpoint_keep, checkpoint)
                set_status(run_dir, "training", completed_episodes=progress["episodes"],
                           transitions=progress["transitions"], updates=agent.updates,
                           successes=progress["successes"], sampled_families=sampler.counts,
                           checkpoint=str(checkpoint), removed_older_checkpoints=removed)
                return checkpoint

            def validate():
                before_parameters, before_updates = state_hash(agent.online), agent.updates
                result = evaluate_records(agent, encoder, runner, validation,
                                          recipe.horizon, stop.exists)
                if state_hash(agent.online) != before_parameters or agent.updates != before_updates:
                    raise RuntimeError("Validation modified the task network")
                encoder.verify_frozen()
                rank, family_summary = rank_validation(
                    result, horizon=recipe.horizon, episode=progress["episodes"])
                progress["validation_history"].append({"episode": progress["episodes"],
                                                       "rank": rank, "summary": family_summary})
                progress["validated_episode"] = progress["episodes"]
                if progress["best_rank"] is None or rank > tuple(progress["best_rank"]):
                    model = run_dir / "models" / f"best-{progress['episodes']}-{uuid.uuid4().hex[:8]}.pt"
                    agent.save(model, MultiFamilyReplay(4), progress, contract)
                    progress["best_rank"] = rank
                    progress["best_model"] = str(model)
                atomic_json(run_dir / "validation.json", result)
                atomic_json(run_dir / "validation-summary.json", family_summary)

            set_status(run_dir, "training", purpose=contract["purpose"], pending_validation=False)
            if progress["episodes"] and progress["validated_episode"] != progress["episodes"] and (
                    progress["episodes"] % recipe.evaluation_interval == 0
                    or progress["episodes"] == recipe.total_episodes):
                try:
                    validate()
                except GraphEvaluationPaused:
                    checkpoint_save()
                    set_status(run_dir, "paused", pending_validation=True)
                    return progress
            with GraphPowerAdjustmentEnv(encoder, runner, recipe.horizon) as env:
                while progress["episodes"] < recipe.total_episodes:
                    if stop.exists() or (pause_after_episode is not None
                                         and progress["episodes"] >= pause_after_episode):
                        checkpoint_save()
                        set_status(run_dir, "paused")
                        return progress
                    record = records[sampler.next()]
                    state, info = env.reset(record["source"], record["policy"],
                                            expected_inputs=record["input_cards"])
                    while not env.done:
                        epsilon = max(0.05, 1 - 0.95 * progress["transitions"] / recipe.epsilon_decay)
                        preferred = ((state.targets == 1)
                                     & (state.edges[state.candidate_edges, 0] == 0))
                        action = agent.choose(state, epsilon, exploration_mask=preferred)
                        if action is None:
                            raise RuntimeError("Nonterminal environment has no action")
                        next_state, reward, terminated, truncated, info = env.step(action)
                        with (run_dir / "transitions.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({
                                "episode": progress["episodes"] + 1, "source": record["case_id"],
                                "family": record["grid_family_id"], "step": env.steps,
                                "action": int(action), "device_key": state.device_keys[action],
                                "target": int(state.targets[action]), "reward": reward,
                                "terminated": terminated, "truncated": truncated,
                                "epsilon": epsilon,
                            }, allow_nan=False) + "\n")
                        replay.add(GraphTransition(state, action, reward, next_state,
                                                   terminated or truncated,
                                                   record["grid_family_id"], record["case_id"]))
                        progress["transitions"] += 1
                        if len(replay.items) >= recipe.warmup:
                            batch = sample_balanced(replay, agent.rng, recipe.batch_size,
                                                    node_budget=recipe.node_budget,
                                                    edge_budget=recipe.edge_budget,
                                                    action_budget=recipe.action_budget)
                            loss = agent.learn(batch)
                            if loss is None:
                                raise RuntimeError("Empty learning batch")
                            progress["losses"].append(loss)
                            progress["batch_sizes"].append(len(batch))
                        if terminated:
                            progress["successes"] += 1
                            progress["family_successes"][record["grid_family_id"]] += 1
                        state = next_state
                    progress["lengths"].append(env.steps)
                    progress["episodes"] += 1
                    progress["updates"] = agent.updates
                    progress["sampler"] = copy.deepcopy(sampler.state_dict())
                    set_status(run_dir, "training", completed_episodes=progress["episodes"],
                               transitions=progress["transitions"], updates=agent.updates,
                               successes=progress["successes"],
                               family_successes=progress["family_successes"],
                               sampled_families=sampler.counts,
                               last_case_id=record["case_id"],
                               last_family=record["grid_family_id"],
                               last_reason=info["reason"])
                    if stop.exists():
                        checkpoint_save()
                        set_status(run_dir, "paused")
                        return progress
                    if (progress["episodes"] % recipe.evaluation_interval == 0
                            or progress["episodes"] == recipe.total_episodes):
                        try:
                            validate()
                        except GraphEvaluationPaused:
                            checkpoint_save()
                            set_status(run_dir, "paused", pending_validation=True)
                            return progress
                    if (progress["episodes"] % recipe.checkpoint_interval == 0
                            or progress["episodes"] == recipe.total_episodes):
                        checkpoint_save()
                    atomic_json(run_dir / "representation.json", agent.audit)
            if progress["validated_episode"] != progress["episodes"]:
                validate()
            if progress["best_model"] is None:
                raise RuntimeError("No multi-family validation-selected model")
            encoder.verify_frozen()
            checkpoint_save()
            final = run_dir / "models" / f"final-{uuid.uuid4().hex[:8]}.pt"
            agent.save(final, MultiFamilyReplay(4), progress, contract)
            selected = {
                "schema": 2, "purpose": contract["purpose"],
                "protocol": "multi_family_joint_training_v1",
                "selection": "family_macro_validation_only",
                "model": progress["best_model"], "model_sha256": file_hash(progress["best_model"]),
                "final_model": str(final), "final_model_sha256": file_hash(final),
                "best_rank": progress["best_rank"],
                "training_family_ids": list(TRAIN_FAMILIES),
                "training_bus_counts": [118, 300, 2000, 3120],
                "encoder": encoder.graph_contract, "solver_sha256": solver_hash,
                "recipe": recipe.to_dict(), "manifest_sha256": audited["digest"],
                "seen_test_sha256": sealed["seen_test_sha256"],
                "exploration_policy": contract["exploration_policy"],
                "formal_training_started": not acceptance,
            }
            atomic_json(run_dir / "selected.json", selected)
            atomic_json(run_dir / "summary.json", {
                "status": "passed", "purpose": contract["purpose"], "progress": progress,
                "sampled_families": sampler.counts, "representation": agent.audit,
                "encoder_frozen": True, "external_test_used": False,
                "formal_training_started": not acceptance,
                "seconds_this_invocation": time.monotonic() - started,
                "cuda_peak_allocated": torch.cuda.max_memory_allocated()
                if recipe.device == "cuda" else None,
            })
            set_status(run_dir, "acceptance_complete" if acceptance else "training_complete")
            return progress
        except BaseException as exc:
            set_status(run_dir, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error_type=type(exc).__name__, error=str(exc))
            raise
