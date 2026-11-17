from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .control import ControlPolicy
from .environment import PowerAdjustmentEnv
from .foundation import FoundationPowerAdjustmentEnv, UnifiedFrozenEncoder
from .qvio_reward import QViolationRewardEnhancer
from .model import DeviceRegistry
from .paths import ProjectPaths, case_input_fingerprint
from .psasp import CaseWorkspace, PSASPRunner, read_records


def _effective_qvio_reward(paths: ProjectPaths, requested: bool) -> bool:
    if not requested:
        return False
    enhancer = QViolationRewardEnhancer(PSASPRunner(paths.solver))
    enhancer.require_compatible_solver()
    representative = paths.registry_cases("train")[0]
    if enhancer.is_applicable(representative):
        return True
    sys.stderr.write(
        "warning: Q-violation reward disabled: dataset has no active Type=-2/-3 "
        "generator; using PSASP terminal reward plus step cost\n"
    )
    return False


def _foundation_encoder(paths: ProjectPaths, device: str | None = None) -> UnifiedFrozenEncoder:
    if device is None or device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    return UnifiedFrozenEncoder.from_project(paths.root, device=device)


def inspect_command(paths: ProjectPaths) -> int:
    train_cases = paths.cases("train")
    validation_cases = paths.cases("validation") if paths.catalog_split else []
    test_cases = paths.cases("test")
    converged_cases = paths.cases("converged") if paths.catalog_split else []
    all_cases = paths.cases("all") if paths.catalog_split else train_cases + test_cases
    manifest = (
        json.loads(paths.flow_manifest.read_text(encoding="utf-8"))
        if paths.flow_manifest.is_file() else None
    )
    if manifest is not None and manifest.get("records"):
        unique_input_count = len({
            str(item["input_fingerprint"]) for item in manifest["records"]
        })
    else:
        unique_input_count = len({case_input_fingerprint(case) for case in all_cases})
    policy = ControlPolicy.load(paths.control_policy)
    registry_sources = paths.registry_cases("train")
    registry = DeviceRegistry.from_cases(registry_sources, policy)
    foundation = _foundation_encoder(paths, "cpu")
    qvio_enhancer = QViolationRewardEnhancer(PSASPRunner(paths.solver))
    payload = {
        "project_root": str(paths.root),
        "solver": str(paths.solver),
        "solver_exists": paths.solver.is_file(),
        "dataset_root": None if paths.dataset_root is None else str(paths.dataset_root),
        "catalog_split": paths.catalog_split,
        "train_case_count": len(train_cases),
        "validation_case_count": len(validation_cases),
        "test_case_count": len(test_cases),
        "converged_case_count": len(converged_cases),
        "nonconverged_case_count": len(train_cases) + len(validation_cases) + len(test_cases),
        "all_case_count": len(all_cases),
        "unique_input_point_count": unique_input_count,
        "bus_record_count": len(read_records(all_cases[0] / "LF.L1")),
        "train_case_example": str(train_cases[0]),
        "test_case_example": str(test_cases[0]),
        "base_observation_size": registry.observation_size,
        "foundation_observation_size": foundation.size,
        "observation_size": registry.observation_size + foundation.size,
        "encoder_contract": foundation.contract,
        "action_size": registry.action_size,
        "control_policy": str(paths.control_policy),
        "control_policy_sha256": registry.policy_digest,
        "registry_schema_version": registry.schema_version,
        "registered_generators": len(registry.generator_ids),
        "shunt_action_candidates": len(registry.shunt_ids),
        "observed_branch_disturbances": len(registry.branch_ids),
        "observed_transformers": len(registry.transformer_ids),
        "observed_dc_plans": len(registry.dc_ids),
        "formal_flow_manifest": str(paths.flow_manifest),
        "formal_flow_manifest_exists": paths.flow_manifest.is_file(),
        "reward_contract": qvio_enhancer.contract,
        "qvio_reward_applicable": qvio_enhancer.is_applicable(registry_sources[0]),
    }
    if manifest is not None:
        payload["formal_flow_manifest_schema_version"] = manifest.get("schema_version")
        payload["formal_flow_manifest_current"] = manifest.get("schema_version") == 2
        if payload["formal_flow_manifest_current"]:
            records = manifest.get("records", [])
            invalid = [item["case"] for item in records if not item.get("active_power_valid")]
            payload["active_invalid_case_count"] = len(invalid)
            payload["active_invalid_case_examples"] = invalid[:10]
            if not invalid:
                payload["reactive_train_case_count"] = len(paths.reactive_cases("train"))
                payload["reactive_validation_case_count"] = len(paths.reactive_cases("validation"))
                payload["reactive_test_case_count"] = len(paths.reactive_cases("test"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def smoke_command(paths: ProjectPaths, split: str) -> int:
    case = paths.cases(split)[0]
    try:
        with CaseWorkspace(case) as workspace:
            result = PSASPRunner(paths.solver).run(workspace)
    except (RuntimeError, TimeoutError) as exc:
        print(json.dumps({"case": str(case), "ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"case": str(case), "converged": result.converged, "error": result.error}, ensure_ascii=False))
    return 0


def gpu_check_command() -> int:
    try:
        import torch
    except ImportError:
        print(json.dumps({"ok": False, "error": "PyTorch is not installed"}))
        return 2
    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "error": "CUDA is unavailable", "torch": torch.__version__}))
        return 2
    value = torch.randn(512, 512, device="cuda", requires_grad=True)
    loss = (value @ value).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    payload = {
        "ok": bool(torch.isfinite(value.grad).all()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 2),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


def qvio_smoke_command(paths: ProjectPaths, args: argparse.Namespace) -> int:
    enhancer = QViolationRewardEnhancer(PSASPRunner(paths.solver))
    enhancer.require_compatible_solver()
    source = args.case
    if source is None:
        cases = paths.cases(args.split)
        if not 0 <= args.case_index < len(cases):
            raise ValueError(f"case-index must be in [0, {len(cases) - 1}]")
        source = cases[args.case_index]
    result = enhancer.evaluate(source)
    print(json.dumps({"case": str(Path(source).resolve()), **asdict(result)}, ensure_ascii=False, indent=2))
    return 0 if result.available else 2


def train_command(paths: ProjectPaths, args: argparse.Namespace) -> int:
    from .lifecycle import run_lock, set_status
    args._run_accepted = False
    with run_lock(args.run_dir):
        try:
            return _train_command(paths, args)
        except BaseException as exc:
            if args._run_accepted:
                set_status(args.run_dir, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                           exception_type=type(exc).__name__, error=str(exc),
                           recovery='Resume explicitly from the last complete checkpoint')
            raise


def _train_command(paths: ProjectPaths, args: argparse.Namespace) -> int:
    from .dqn import DQNAgent, evaluate, train, train_parallel
    from .run_store import prepare_run, atomic_json, file_hash
    from .runtime import ProductionRuntime

    run_dir = args.run_dir.resolve()
    allowed_root = (paths.root / 'artifacts' / 'runs').resolve()
    if not run_dir.is_relative_to(allowed_root) or run_dir == allowed_root:
        raise ValueError('run-dir must be an experiment directory beneath artifacts/runs')
    if args.resume is not None and str(args.resume) == 'latest':
        from .checkpoints import select_latest
        args.resume, rejected = select_latest(run_dir / 'models/checkpoints', [run_dir / 'models/dqn-final.pt'])
        atomic_json(run_dir / 'resume-selection.json', {'selected': str(args.resume), 'rejected': rejected})
    if not args.resume and run_dir.exists():
        raise FileExistsError(f'Fresh run already exists: {run_dir}')
    if args.resume and not args.resume.resolve().is_relative_to(run_dir):
        raise ValueError('Resume checkpoint must belong to this run directory')
    if args.total_episodes <= 0:
        raise ValueError('total-episodes must be positive')
    paths = replace(paths, model_dir=run_dir / 'models', result_dir=run_dir / 'results')
    enable_qvio_reward = _effective_qvio_reward(paths, not args.disable_qvio_reward)
    policy = ControlPolicy.load(paths.control_policy)
    registry = DeviceRegistry.from_cases(paths.registry_cases("train"), policy)
    train_cases = paths.reactive_cases("train")
    validation_cases = paths.reactive_cases("validation")
    test_cases = paths.reactive_cases("test")
    if args.acceptance_cases < 0:
        raise ValueError('acceptance-cases cannot be negative')
    if args.acceptance_cases:
        train_cases = train_cases[:args.acceptance_cases]
        validation_cases = validation_cases[:args.acceptance_cases]
        test_cases = test_cases[:args.acceptance_cases]
    agent = replay = metrics = None
    start_episode = 0
    selected_device = None if args.device == "auto" else args.device
    if args.workers != 1:
        raise ValueError("Unified frozen-encoder training currently requires --workers 1")
    runtime = ProductionRuntime(paths, args.device, registry=registry, audit_dir=run_dir / 'solver')
    foundation = runtime.encoder
    if args.preflight_only:
        import torch
        from .input_validation import InputCaseValidator
        case = train_cases[0]
        validation = InputCaseValidator().validate(case)
        if not validation.active_power_valid:
            raise ValueError('Representative case failed active power contract')
        tokens = foundation.encode(case)
        atomic_json(run_dir / 'preflight.json', {
            'status': 'encoder_preflight_passed', 'psasp_checked': False,
            'encoder': foundation.contract, 'observation_size': registry.observation_size + tokens.size,
            'action_size': registry.action_size, 'torch': torch.__version__,
            'token_norm': float(__import__('numpy').linalg.norm(tokens)),
            'note': 'Structural preflight only; not permission to launch formal training',
        })
        print(json.dumps({'preflight': str(run_dir / 'preflight.json')}, ensure_ascii=False))
        return 0
    if args.data_audit is None:
        raise ValueError('Training requires --data-audit from a successful full preflight')
    from .data_preflight import verify_audit
    data_evidence = verify_audit(paths, args.data_audit, registry)
    readiness_evidence = None
    if not args.acceptance_cases:
        from .readiness import verify_readiness
        readiness_evidence = verify_readiness(paths.root, args.readiness, args.data_audit, paths.solver, args.config)
    dataset = {}
    for split, cases in (('train', train_cases), ('validation', validation_cases), ('test', test_cases)):
        dataset[split] = [{'case': c.name, 'fingerprint': case_input_fingerprint(c).hex()} for c in cases]
    recipe = {key: getattr(args, key) for key in (
        'total_episodes', 'max_steps', 'batch_size', 'seed', 'perturbations',
        'eval_perturbations', 'epsilon_decay_steps', 'evaluation_interval',
        'evaluation_episodes', 'full_evaluation_interval', 'device', 'workers', 'acceptance_cases')}
    recipe['optimizer'] = {'kind': 'AdamW', 'lr': 3e-4, 'betas': [0.9, 0.999],
                           'eps': 1e-8, 'weight_decay': 0.01}
    recipe['algorithm'] = {'gamma': 0.99, 'target_interval': 250,
                          'replay_capacity': 20000, 'loss': 'SmoothL1', 'clip_norm': 10.0}
    recipe['deterministic_cuda'] = str(runtime.device).startswith('cuda')
    contract = {'encoder': foundation.contract, 'registry': registry.to_dict(),
                'readiness': readiness_evidence,
                'purpose': 'acceptance_only' if args.acceptance_cases else 'formal_training',
                'data_preflight': data_evidence,
                'dataset': dataset, 'recipe': recipe, 'qvio_enabled': enable_qvio_reward,
                'solver_sha256': file_hash(paths.solver),
                'code_files': {p.name: file_hash(p) for p in sorted((paths.root / 'rl_poweradj').glob('*.py'))}}
    run_identity = prepare_run(run_dir, contract, resume=bool(args.resume))
    args._run_accepted = True
    paths.ensure_outputs()
    if args.resume:
        if not args.resume.with_suffix(args.resume.suffix + '.manifest.json').is_file():
            raise ValueError('Resume requires a checkpoint integrity manifest')
        agent, replay, metrics, start_episode = DQNAgent.load_checkpoint(args.resume, selected_device)
        usage = agent.metadata.get("foundation_usage", {})
        if usage.get("inference_only"):
            raise ValueError("The zero-weight foundation bridge is inference-only and cannot resume training")
        if agent.metadata.get('run_identity') != run_identity:
            raise ValueError('Checkpoint run identity differs from run.json')
    remaining = args.total_episodes - start_episode
    if remaining < 0:
        raise ValueError('Requested total episodes are already completed')
    if args.resume:
        (run_dir / 'pause.request').unlink(missing_ok=True)
    from .lifecycle import set_status
    set_status(run_dir, 'resuming' if args.resume else 'starting', completed_episodes=start_episode)
    with (
        runtime.environment(
            cases=train_cases, max_steps=args.max_steps, initial_perturbations=args.perturbations,
            enable_qvio_reward=enable_qvio_reward,
        ) as env,
        runtime.environment(
            cases=validation_cases, max_steps=args.max_steps,
            initial_perturbations=args.eval_perturbations,
            enable_qvio_reward=enable_qvio_reward,
        ) as validation_env,
        runtime.environment(
            cases=test_cases, max_steps=args.max_steps,
            initial_perturbations=args.eval_perturbations,
            enable_qvio_reward=enable_qvio_reward,
        ) as test_env,
    ):
        env.run_identity = run_identity
        def progress(episode, current_metrics):
            atomic_json(run_dir / 'progress.json', {
                'status': 'running', 'episode': episode,
                'completed_episodes': current_metrics.completed_episodes,
                'transitions': current_metrics.transitions, 'updates': current_metrics.updates,
                'successes': current_metrics.successes,
                'skipped_converged': current_metrics.skipped_converged,
            })
        common = dict(
            episodes=remaining,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            device=selected_device,
            agent=agent,
            replay=replay,
            metrics=metrics,
            start_episode=start_episode,
            checkpoint_dir=paths.model_dir / "checkpoints",
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_keep_last=args.checkpoint_keep_last,
            epsilon_decay_steps=args.epsilon_decay_steps,
            episode_callback=progress,
            stop_requested=lambda: (run_dir / 'pause.request').exists(),
        )
        if remaining:
            agent, rewards = train(
                env, evaluation_env=validation_env,
                evaluation_interval=args.evaluation_interval,
                evaluation_episodes=args.evaluation_episodes,
                full_evaluation_interval=args.full_evaluation_interval,
                best_model_path=paths.model_dir / "dqn-best.pt", **common)
        else:
            # Pausing on the last training episode can leave final validation pending.
            from .dqn import _record_validation, EvaluationPaused
            from .training_contract import require_training_contract
            require_training_contract(env, agent, start_episode, replay, metrics)
            agent.training_metrics, agent.training_replay = metrics, replay
            agent.training_paused = False
            validation_env._pause_requested = lambda: (run_dir / 'pause.request').exists()
            try:
                _record_validation(agent, validation_env, metrics, start_episode, args.max_steps,
                                   args.evaluation_episodes, paths.model_dir / 'dqn-best.pt', full=True)
            except EvaluationPaused:
                agent.training_paused = True
            rewards = []
        if getattr(agent, 'training_paused', False):
            from .lifecycle import set_status
            set_status(run_dir, 'paused', completed_episodes=agent.training_metrics.completed_episodes)
            print(json.dumps({'status': 'paused', 'run_dir': str(run_dir)}, ensure_ascii=False))
            return 0
        final_path = paths.model_dir / "dqn-final.pt"
        agent.save(
            final_path, replay=agent.training_replay, metrics=agent.training_metrics,
            episode=args.total_episodes,
        )
        best_path = paths.model_dir / "dqn-best.pt"
    payload = {
        "selected_model": str(best_path),
        "final_model": str(final_path),
        "rewards": rewards,
        "training": agent.training_metrics.to_dict(),
        "final_test_evaluation": None,
        "test_status": "not_run_use_explicit_evaluation_after_model_selection",
    }
    result_path = paths.result_dir / "training-metrics.json"
    atomic_json(result_path, payload)
    atomic_json(run_dir / 'progress.json', {'status': 'training_complete',
                'completed_episodes': agent.training_metrics.completed_episodes})
    print(json.dumps({
        "selected_model": str(best_path),
        "final_model": str(final_path),
        "metrics": str(result_path),
    }, ensure_ascii=False))
    return 0


def evaluate_command(paths: ProjectPaths, args: argparse.Namespace) -> int:
    from .dqn import DQNAgent, evaluate
    from .runtime import ProductionRuntime

    if args.output and args.output.exists():
        raise FileExistsError('Evaluation output already exists')
    device = None if args.device == "auto" else args.device
    enable_qvio_reward = _effective_qvio_reward(paths, not args.disable_qvio_reward)
    agent = DQNAgent.load(args.model, device)
    registry_data = agent.metadata.get("registry")
    if registry_data is None:
        raise ValueError("Model checkpoint does not contain a device registry")
    registry = DeviceRegistry.from_dict(registry_data)
    runtime = ProductionRuntime(paths, args.device, registry=registry)
    runtime.require_agent(agent)
    foundation = runtime.encoder
    with runtime.environment(
        cases=paths.reactive_cases(args.phase), max_steps=args.max_steps,
        initial_perturbations=args.perturbations,
        enable_qvio_reward=enable_qvio_reward,
    ) as env:
        episodes = len(env.cases) if args.episodes is None else args.episodes
        result = evaluate(agent, env, episodes, args.max_steps)
    payload = asdict(result)
    if args.output:
        from .run_store import atomic_json, file_hash
        if args.output.exists():
            raise FileExistsError('Evaluation output already exists')
        payload = {'phase': args.phase, 'model_sha256': file_hash(args.model),
                   'solver_sha256': file_hash(paths.solver), 'encoder': foundation.contract,
                   'max_steps': args.max_steps, 'evaluation': payload}
        atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def recover_command(paths: ProjectPaths, args: argparse.Namespace) -> int:
    from .dqn import DQNAgent
    from .recovery import ConvergenceRecovery
    from .runtime import ProductionRuntime

    device = None if args.device == "auto" else args.device
    agent = DQNAgent.load(args.model, device)
    runtime = ProductionRuntime(paths, args.device, registry=DeviceRegistry.from_dict(agent.metadata['registry']))
    runtime.require_agent(agent)
    foundation = runtime.encoder
    source = args.case
    if source is None:
        cases = paths.cases(args.split)
        if not 0 <= args.case_index < len(cases):
            raise ValueError(
                f"case-index must be in [0, {len(cases) - 1}] for split {args.split}"
            )
        source = cases[args.case_index]
    result = ConvergenceRecovery(
        runtime.runner, agent, foundation_encoder=foundation
    ).run(
        source,
        output_case=args.output,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    payload = {
        "source_case": str(Path(source).resolve()),
        "active_power_valid": result.input_validation.active_power_valid,
        "active_power_errors": list(result.input_validation.errors),
        "active_power_islands": [asdict(item) for item in result.input_validation.islands],
        "initial_converged": result.initial_validation.converged,
        "initial_error": result.initial_validation.error,
        "final_converged": result.final_validation.converged,
        "final_error": result.final_validation.error,
        "steps": [asdict(step) for step in result.steps],
        "output_case": None if result.output_case is None else str(result.output_case),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def manifest_command(paths: ProjectPaths, args: argparse.Namespace) -> int:
    from .manifest import inspect_case_flow

    cases = paths.cases(args.split)
    runner = PSASPRunner(paths.solver)
    records = []
    for index, case in enumerate(cases, 1):
        record = inspect_case_flow(case, runner)
        records.append(record.to_dict())
        print(json.dumps({
            "progress": f"{index}/{len(cases)}",
            "case": case.name,
            "category": record.category,
            "first_ac_error": record.first_ac_error,
        }, ensure_ascii=False))
    payload = {"schema_version": 2, "records": records}
    output = (args.output or paths.flow_manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "records": len(records)}, ensure_ascii=False))
    return 0


def baseline_command(paths: ProjectPaths, args: argparse.Namespace) -> int:
    from .baseline import beam_search

    cases = paths.cases(args.split)
    if not 0 <= args.case_index < len(cases):
        raise ValueError(f"case-index must be in [0, {len(cases) - 1}]")
    case = cases[args.case_index]
    policy = ControlPolicy.load(paths.control_policy)
    registry = DeviceRegistry.from_cases(paths.registry_cases("train"), policy)

    def factory():
        return PowerAdjustmentEnv(
            paths=paths, cases=[case], registry=registry, max_steps=args.max_steps,
            enable_qvio_reward=False,
        )

    result = beam_search(
        factory, case_index=0, max_steps=args.max_steps, beam_width=args.beam_width
    )
    print(json.dumps({
        "case": str(case),
        "algorithm": "greedy" if args.beam_width == 1 else "beam_search",
        **asdict(result),
    }, ensure_ascii=False, indent=2))
    return 0 if result.converged else 1


def feature_audit_command(paths: ProjectPaths, args: argparse.Namespace) -> int:
    from .feature_audit import audit_observations

    policy = ControlPolicy.load(paths.control_policy)
    registry = DeviceRegistry.from_cases(paths.registry_cases("train"), policy)
    result = audit_observations(paths.cases(args.split), registry, paths.flow_manifest)
    output = (args.output or paths.result_dir / "feature-audit.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "case_count", "observation_size", "varying_feature_count",
            "constant_feature_count",
        )
    } | {"output": str(output)}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    incoming = sys.argv[1:] if argv is None else argv
    if incoming and incoming[0] == 'multigrid':
        from .multigrid_cli import main as multigrid_main
        return multigrid_main(incoming[1:])
    parser = argparse.ArgumentParser(prog="rl-poweradj")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Inspect paths, datasets, and fixed device registry")
    commands.add_parser("gpu-check", help="Run a CUDA forward/backward check")
    pause_parser = commands.add_parser('pause', help='Request an episode-boundary checkpoint and pause')
    pause_parser.add_argument('--run-dir', type=Path, required=True)
    data_preflight = commands.add_parser('preflight', help='Audit all split inputs without PSASP')
    data_preflight.add_argument('--output', type=Path, required=True)
    qvio_smoke = commands.add_parser(
        "qvio-smoke", help="Validate the PSASP-native Q-limit violation reward"
    )
    qvio_smoke.add_argument("--case", type=Path)
    qvio_smoke.add_argument(
        "--split", choices=("train", "validation", "test", "converged", "all"),
        default="train",
    )
    qvio_smoke.add_argument("--case-index", type=int, default=0)
    manifest_parser = commands.add_parser(
        "manifest", help="Validate active inputs and recompute unchanged first-AC classifications"
    )
    manifest_parser.add_argument(
        "--split", choices=("train", "validation", "test", "converged", "all"),
        default="all",
    )
    manifest_parser.add_argument("--output", type=Path)

    baseline_parser = commands.add_parser(
        "baseline", help="Run deterministic residual-ranked reactive search"
    )
    baseline_parser.add_argument(
        "--split", choices=("train", "validation", "test", "converged", "all"),
        default="test",
    )
    baseline_parser.add_argument("--case-index", type=int, default=0)
    baseline_parser.add_argument("--max-steps", type=int, default=5)
    baseline_parser.add_argument("--beam-width", type=int, default=1)
    audit_parser = commands.add_parser(
        "feature-audit", help="Audit observation schema 2 variance on a dataset split"
    )
    audit_parser.add_argument(
        "--split", choices=("train", "validation", "test", "converged", "all"),
        default="train",
    )
    audit_parser.add_argument("--output", type=Path)
    smoke = commands.add_parser("smoke", help="Run one isolated PSASP calculation")
    smoke.add_argument(
        "--split", choices=("train", "validation", "test", "converged", "all"),
        default="converged",
    )

    train_parser = commands.add_parser("train", help="Train DQN with checkpoints and evaluation")
    train_parser.add_argument('--run-dir', type=Path, required=True)
    train_parser.add_argument('--total-episodes', type=int, default=2416)
    train_parser.add_argument('--config', type=Path)
    train_parser.add_argument('--data-audit', type=Path)
    train_parser.add_argument('--readiness', type=Path)
    train_parser.add_argument('--acceptance-cases', type=int, default=0,
                              help='Explicit acceptance-only subset per split; zero uses full data')
    train_parser.add_argument('--preflight-only', action='store_true', help='Check encoder on representative input without PSASP or training')
    train_parser.add_argument("--max-steps", type=int, default=150)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--perturbations", type=int, default=0)
    train_parser.add_argument("--eval-perturbations", type=int, default=0)
    train_parser.add_argument("--checkpoint-interval", type=int, default=10)
    train_parser.add_argument(
        "--checkpoint-keep-last", type=int, default=3,
        help="Keep only the newest N periodic replay checkpoints; zero keeps all",
    )
    train_parser.add_argument(
        "--epsilon-decay-steps", type=int, default=10_000,
        help="Global environment-step horizon for epsilon decay; preserved on resume",
    )
    train_parser.add_argument("--evaluation-interval", type=int, default=10)
    train_parser.add_argument(
        "--evaluation-episodes", type=int, default=12,
        help="Rotating validation batch size",
    )
    train_parser.add_argument(
        "--full-evaluation-interval", type=int, default=50,
        help="Select the best checkpoint on the complete validation set at this interval",
    )
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--workers", type=int, default=1, help="Parallel isolated PSASP rollout workers")
    train_parser.add_argument(
        "--disable-qvio-reward", "--disable-lcq-reward",
        dest="disable_qvio_reward", action="store_true",
        help="A/B baseline: disable PSASP-native Q-limit violation dense reward",
    )

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate a DQN checkpoint")
    evaluate_parser.add_argument('--phase', choices=('validation', 'test'), default='test')
    evaluate_parser.add_argument('--output', type=Path)
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument(
        "--episodes", type=int,
        help="Number of final-test cases; defaults to the complete test set",
    )
    evaluate_parser.add_argument("--max-steps", type=int, default=150)
    evaluate_parser.add_argument("--perturbations", type=int, default=0)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument(
        "--disable-qvio-reward", "--disable-lcq-reward",
        dest="disable_qvio_reward", action="store_true",
    )

    recover = commands.add_parser(
        "recover", help="Recover PSASP convergence with DQN only and export the case"
    )
    recover.add_argument("--model", type=Path, required=True)
    recover.add_argument("--case", type=Path, help="Explicit source case directory")
    recover.add_argument(
        "--split", choices=("train", "validation", "test", "converged", "all"),
        default="test",
    )
    recover.add_argument("--case-index", type=int, default=0)
    recover.add_argument("--output", type=Path, required=True)
    recover.add_argument("--max-steps", type=int, default=100)
    recover.add_argument("--seed", type=int, default=0)
    recover.add_argument("--device", default="auto")

    raw_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_argv)
    paths = ProjectPaths.from_root()
    try:
        from .recipe import apply_recipe
        args = apply_recipe(args, raw_argv)
        if args.command == 'pause':
            from .lifecycle import request_pause
            request_pause(args.run_dir)
            print('Pause requested; wait for progress.json status=paused before resuming or closing')
            return 0
        if args.command == 'preflight':
            from .data_preflight import audit_dataset
            result = audit_dataset(paths, args.output)
            print(json.dumps({key: result[key] for key in ('status', 'case_count', 'splits')}, ensure_ascii=False))
            return 0
        if args.command == "inspect":
            return inspect_command(paths)
        if args.command == "gpu-check":
            return gpu_check_command()
        if args.command == "qvio-smoke":
            return qvio_smoke_command(paths, args)
        if args.command == "manifest":
            return manifest_command(paths, args)
        if args.command == "baseline":
            return baseline_command(paths, args)
        if args.command == "feature-audit":
            return feature_audit_command(paths, args)
        if args.command == "smoke":
            return smoke_command(paths, args.split)
        if args.command == "train":
            return train_command(paths, args)
        if args.command == "evaluate":
            return evaluate_command(paths, args)
        if args.command == "recover":
            return recover_command(paths, args)
        raise ValueError(f"Unknown command: {args.command}")
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
