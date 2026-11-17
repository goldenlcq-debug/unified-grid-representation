from __future__ import annotations

import random
import re
import os
import json
import uuid
import copy
import functools
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .model import DETERMINISTIC_OBSERVATION_METADATA
from .training_contract import require_training_contract
from .run_store import file_hash, atomic_json


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray
    source_case: str | None = None
    episode_step: int | None = None


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.capacity = int(capacity)
        self.data: deque[Transition] = deque(maxlen=self.capacity)

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, size: int) -> list[Transition]:
        return random.sample(self.data, size)

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "data": list(self.data)}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ReplayBuffer":
        buffer = cls(state["capacity"])
        buffer.data.extend(state["data"])
        return buffer

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class TrainMetrics:
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    transitions: int = 0
    updates: int = 0
    successes: int = 0
    skipped_converged: int = 0
    start_episode: int = 0
    completed_episodes: int = 0
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)
    full_validation_history: list[dict[str, Any]] = field(default_factory=list)
    validation_cursor: int = 0
    best_evaluation_score: float | None = None
    best_validation_rank: list[float] | None = None
    global_environment_steps: int = 0
    epsilon_decay_steps: int = 10_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaseEvaluation:
    case_index: int
    source_case: str
    success: bool
    reward: float
    steps: int
    final_error: float | None


@dataclass(frozen=True)
class EvaluationMetrics:
    episodes: int
    successes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    mean_final_error: float | None
    worst_final_error: float | None
    cases: tuple[CaseEvaluation, ...]


class EvaluationPaused(RuntimeError):
    """An incomplete validation must not enter model selection."""


class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        target_update_interval: int = 250,
        device: str | torch.device | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable to PyTorch")
        self.online = QNetwork(self.observation_size, self.action_size).to(self.device)
        self.target = QNetwork(self.observation_size, self.action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=learning_rate)
        self.loss = nn.SmoothL1Loss()
        self.updates = 0
        self.metadata = dict(metadata or {})
        self.training_metrics: TrainMetrics | None = None
        self.training_replay: ReplayBuffer | None = None

    def choose_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float = 0.0,
        rng: Any | None = None,
    ) -> int:
        if np.asarray(state).shape != (self.observation_size,):
            raise ValueError("State shape does not match the trained observation registry")
        if np.asarray(action_mask).shape != (self.action_size,):
            raise ValueError("Action mask shape does not match the trained device registry")
        valid = np.flatnonzero(action_mask)
        if len(valid) == 0:
            raise RuntimeError("Current state has no valid action")
        if epsilon > 0.0:
            generator = rng or random
            if generator.random() < epsilon:
                return int(generator.choice(valid))
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            values = self.online(state_tensor).masked_fill(~mask_tensor, -torch.inf)
        return int(values.argmax(dim=1).item())

    def learn(self, buffer: ReplayBuffer, batch_size: int = 64) -> float | None:
        if len(buffer) < batch_size:
            return None
        foundation = getattr(self, '_foundation', None)
        if foundation is not None:
            foundation.assert_runtime_frozen()
            encoder_ids = {id(p) for p in foundation.bundle.encoder.parameters()}
            optimizer_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
            if encoder_ids & optimizer_ids or optimizer_ids != {id(p) for p in self.online.parameters()}:
                raise RuntimeError('Optimizer parameter isolation violated')
            token_before = self.online.layers[0].weight[:, -3072:].detach().clone()
        batch = buffer.sample(batch_size)
        states = torch.as_tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([x.done for x in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([x.next_action_mask for x in batch]), dtype=torch.bool, device=self.device)
        predicted = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            online_next = self.online(next_states).masked_fill(~next_masks, -torch.inf)
            next_actions = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_next = target_next.masked_fill(~next_masks.any(dim=1), 0.0)
            expected = rewards + self.gamma * (1.0 - dones) * target_next
        loss = self.loss(predicted, expected)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite DQN loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if foundation is not None:
            token_gradient = self.online.layers[0].weight.grad[:, -3072:]
            token_grad_norm = float(token_gradient.norm().item())
            if not torch.isfinite(token_gradient).all():
                raise RuntimeError('Nonfinite token gradient')
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0, error_if_nonfinite=True)
        self.optimizer.step()
        if foundation is not None:
            self.representation_audit = {
                'token_gradient_norm': token_grad_norm,
                'token_update_norm': float((self.online.layers[0].weight[:, -3072:].detach() - token_before).norm().item()),
                'token_weight_norm': float(self.online.layers[0].weight[:, -3072:].detach().norm().item()),
                'encoder_gradients_absent': all(p.grad is None for p in foundation.bundle.encoder.parameters()),
                'update': self.updates + 1,
            }
            if self.updates == 0 or (self.updates + 1) % 250 == 0:
                from .representation_audit import token_sensitivity
                masks = torch.as_tensor(np.stack([x.next_action_mask for x in batch]), dtype=torch.bool, device=self.device)
                # Probe next-states using their own valid-action masks.
                usable = masks.any(dim=1)
                if usable.any():
                    self.token_sensitivity = token_sensitivity(self.online, next_states[usable][:16], masks[usable][:16])
                    indices = usable.nonzero().flatten().tolist()[:16]
                    self.token_sensitivity['transitions'] = [
                        {'case': getattr(batch[i], 'source_case', None),
                         'step': getattr(batch[i], 'episode_step', None), 'action': batch[i].action}
                        for i in indices]
            self.representation_audit['latest_sensitivity'] = getattr(self, 'token_sensitivity', None)
        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(
        self,
        path: Path | str,
        *,
        replay: ReplayBuffer | None = None,
        metrics: TrainMetrics | None = None,
        episode: int = 0,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        torch.save({
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "gamma": self.gamma,
            "target_update_interval": self.target_update_interval,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "metadata": self.metadata,
            "runtime_state": getattr(self, 'runtime_state', {}),
            "representation_audit": getattr(self, 'representation_audit', None),
            "token_sensitivity": getattr(self, 'token_sensitivity', None),
            "episode": int(episode),
            "metrics": None if metrics is None else metrics.to_dict(),
            "replay": None if replay is None else replay.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, temporary)
        with temporary.open('rb+') as stream:
            os.fsync(stream.fileno())
        checksum = file_hash(temporary)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + '.manifest.json'), {
            'schema_version': 1, 'checkpoint_sha256': checksum,
            'file_size': path.stat().st_size, 'episode': episode,
            'updates': self.updates, 'has_replay': replay is not None,
        })

    @classmethod
    def load_checkpoint(
        cls, path: Path | str, device: str | torch.device | None = None
    ) -> tuple["DQNAgent", ReplayBuffer | None, TrainMetrics | None, int]:
        path = Path(path)
        manifest_path = path.with_suffix(path.suffix + '.manifest.json')
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if 'file_size' in manifest and manifest['file_size'] != path.stat().st_size:
                raise ValueError('Checkpoint integrity file size mismatch')
            if manifest['checkpoint_sha256'].lower() != file_hash(path):
                raise ValueError('Checkpoint integrity hash mismatch')
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        if manifest_path.is_file():
            for key in ('episode', 'updates'):
                if key in manifest and manifest[key] != checkpoint.get(key):
                    raise ValueError(f'Checkpoint manifest/payload {key} mismatch')
            if 'has_replay' in manifest and manifest['has_replay'] != (checkpoint.get('replay') is not None):
                raise ValueError('Checkpoint manifest/payload replay mismatch')
        if checkpoint.get('metadata', {}).get('training_origin') == 'fresh_unified_gridencoder' and not manifest_path.is_file():
            raise ValueError('Unified training checkpoint requires integrity manifest')
        agent = cls(
            checkpoint["observation_size"], checkpoint["action_size"],
            gamma=checkpoint.get("gamma", 0.99),
            target_update_interval=checkpoint.get("target_update_interval", 250),
            device=device, metadata=checkpoint.get("metadata"),
        )
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in agent.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(agent.device)
        agent.updates = checkpoint["updates"]
        agent.runtime_state = checkpoint.get('runtime_state', {})
        if checkpoint.get('representation_audit') is not None:
            agent.representation_audit = checkpoint['representation_audit']
        if checkpoint.get('token_sensitivity') is not None:
            agent.token_sensitivity = checkpoint['token_sensitivity']
        replay_state = checkpoint.get("replay")
        replay = None if replay_state is None else ReplayBuffer.from_state_dict(replay_state)
        metrics_state = checkpoint.get("metrics")
        metrics = None if metrics_state is None else TrainMetrics(**metrics_state)
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
            cuda_states = checkpoint.get("cuda_random_state_all")
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        return agent, replay, metrics, int(checkpoint.get("episode", 0))

    @classmethod
    def load(cls, path: Path | str, device: str | torch.device | None = None) -> "DQNAgent":
        return cls.load_checkpoint(path, device)[0]


def _validate_contract(env, agent: DQNAgent, state: np.ndarray, info: dict[str, Any]) -> None:
    if np.asarray(state).shape != (agent.observation_size,):
        raise ValueError("A case changed the fixed observation contract")
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite observation')
    if env.action_size != agent.action_size or np.asarray(info["action_mask"]).shape != (agent.action_size,):
        raise ValueError("A case changed the fixed action contract")


def _environment_metadata(env) -> dict[str, Any]:
    if not hasattr(env, "registry"):
        return {}
    metadata: dict[str, Any] = {
        "registry": env.registry.to_dict(),
        "encoder": dict(
            env.encoder_contract
            if hasattr(env, "encoder_contract")
            else DETERMINISTIC_OBSERVATION_METADATA
        ),
    }
    if hasattr(env, "reward_contract"):
        metadata["reward"] = env.reward_contract
    return metadata


def _require_reward_preflight(env) -> None:
    if hasattr(env, "require_reward_preflight"):
        env.require_reward_preflight()


def _validate_metadata_contract(agent: DQNAgent, metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    if agent.metadata.get("registry") != metadata["registry"]:
        raise ValueError("Checkpoint device registry differs from the environment registry")
    if agent.metadata.get("encoder") != metadata["encoder"]:
        raise ValueError("Checkpoint observation contract differs from the environment contract")
    if "reward" in metadata and agent.metadata.get("reward") != metadata["reward"]:
        raise ValueError("Checkpoint reward contract differs from the environment reward contract")


def _environment_case_count(env) -> int | None:
    cases = getattr(env, "cases", None)
    return len(cases) if cases is not None and len(cases) else None


def _validation_score(result: EvaluationMetrics) -> tuple[float, float]:
    """Lexicographic action-required recovery rate, then mean action count."""
    required = [case for case in result.cases if not (case.success and case.steps == 0)]
    if not required:
        return (0.0, 0.0)  # No policy evidence from initially converged cases.
    return (sum(case.success for case in required) / len(required),
            -float(np.mean([case.steps for case in required])))


def epsilon_at_step(
    global_step: int,
    decay_steps: int,
    start: float = 1.0,
    end: float = 0.05,
) -> float:
    if decay_steps <= 0:
        raise ValueError("epsilon decay_steps must be positive")
    progress = min(max(global_step, 0) / decay_steps, 1.0)
    return float(start + progress * (end - start))


def _record_validation(
    agent: DQNAgent,
    env,
    metrics: TrainMetrics,
    episode: int,
    max_steps: int,
    evaluation_episodes: int,
    best_model_path: Path | str | None,
    *,
    full: bool,
) -> EvaluationMetrics:
    case_count = _environment_case_count(env)
    count = case_count if full and case_count is not None else evaluation_episodes
    if count <= 0:
        raise ValueError("Evaluation requires at least one episode")
    start_case = 0 if full else metrics.validation_cursor
    result = evaluate(agent, env, count, max_steps, start_case=start_case)
    record = {"episode": int(episode), "start_case": int(start_case), **asdict(result)}
    if full:
        metrics.full_validation_history.append(record)
        score = _validation_score(result)
        if metrics.best_validation_rank is None or score > tuple(metrics.best_validation_rank):
            metrics.best_validation_rank = list(score)
            if best_model_path is not None:
                agent.save(best_model_path, metrics=metrics, episode=episode)
    else:
        metrics.evaluation_history.append(record)
        modulo = case_count or max(evaluation_episodes, 1)
        metrics.validation_cursor = (start_case + count) % modulo
    return result


def _save_periodic_checkpoint(
    agent: DQNAgent,
    checkpoint_dir: Path | str,
    episode: int,
    replay: ReplayBuffer,
    metrics: TrainMetrics,
    keep_last: int,
) -> Path:
    directory = Path(checkpoint_dir)
    path = directory / f"dqn-episode-{episode}.pt"
    agent.save(path, replay=replay, metrics=metrics, episode=episode)
    if keep_last > 0:
        numbered = []
        for candidate in directory.glob("dqn-episode-*.pt"):
            match = re.fullmatch(r"dqn-episode-(\d+)\.pt", candidate.name)
            if match and int(match.group(1)) <= episode:
                try:
                    seal = json.loads(candidate.with_suffix(candidate.suffix + '.manifest.json').read_text(encoding='utf-8'))
                    valid = (seal.get('episode') == int(match.group(1)) and seal.get('has_replay') is True
                             and seal.get('file_size') == candidate.stat().st_size
                             and seal.get('checkpoint_sha256') == file_hash(candidate))
                    if valid:
                        numbered.append((int(match.group(1)), candidate))
                except (OSError, ValueError, TypeError):
                    continue  # Incomplete files never displace complete recovery points.
        numbered.sort()
        for _number, stale in numbered[:-keep_last]:
            if stale == path:
                continue
            stale.unlink()
            stale.with_suffix(stale.suffix + '.manifest.json').unlink(missing_ok=True)
    return path


def _single_production_training(function):
    @functools.wraps(function)
    def wrapped(env, *args, **kwargs):
        from .foundation import FoundationPowerAdjustmentEnv
        from .psasp import PSASPRunner
        from .solver_lock import SolverLock
        if isinstance(env, FoundationPowerAdjustmentEnv) and isinstance(env.runner, PSASPRunner):
            with SolverLock(Path(tempfile.gettempdir()) / 'rl_poweradj_training.lock'):
                return function(env, *args, **kwargs)
        return function(env, *args, **kwargs)
    return wrapped


@_single_production_training
def train(
    env,
    episodes: int = 100,
    max_steps: int = 100,
    batch_size: int = 64,
    seed: int = 0,
    device: str | torch.device | None = None,
    agent: DQNAgent | None = None,
    replay: ReplayBuffer | None = None,
    metrics: TrainMetrics | None = None,
    start_episode: int = 0,
    checkpoint_dir: Path | str | None = None,
    checkpoint_interval: int = 10,
    checkpoint_keep_last: int = 3,
    episode_callback: Callable[[int, TrainMetrics], None] | None = None,
    evaluation_env=None,
    evaluation_interval: int = 0,
    evaluation_episodes: int = 1,
    full_evaluation_interval: int = 0,
    best_model_path: Path | str | None = None,
    epsilon_decay_steps: int = 10_000,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[DQNAgent, list[float]]:
    require_training_contract(env, agent, start_episode, replay, metrics)
    if checkpoint_keep_last < 0:
        raise ValueError("checkpoint_keep_last cannot be negative")
    if start_episode == 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    _require_reward_preflight(env)
    reset_kwargs = {"seed": seed if start_episode == 0 else None}
    if start_episode > 0 and hasattr(env, '_rng'):
        saved_runtime = getattr(agent, 'runtime_state', {})
        if 'env_rng' not in saved_runtime:
            raise ValueError('Resume checkpoint has no environment RNG state')
        env._rng.bit_generator.state = copy.deepcopy(saved_runtime['env_rng'])
    case_count = _environment_case_count(env)
    if case_count is not None:
        reset_kwargs["case_index"] = start_episode % case_count
    state, info = env.reset(**reset_kwargs)
    metadata = _environment_metadata(env)
    if hasattr(env, 'foundation_encoder'):
        metadata['training_origin'] = 'fresh_unified_gridencoder'
        metadata['max_steps'] = env.max_steps
        if hasattr(env, 'run_identity'):
            metadata['run_identity'] = env.run_identity
    agent = agent or DQNAgent(len(state), env.action_size, device=device, metadata=metadata)
    if hasattr(env, 'foundation_encoder'):
        agent._foundation = env.foundation_encoder
        if start_episode == 0:
            from .representation_audit import network_digest
            agent.metadata['initial_network_sha256'] = network_digest(agent.online)
            agent.metadata['initial_token_weight_norm'] = float(agent.online.layers[0].weight[:, -3072:].detach().norm())
            if agent.metadata['initial_token_weight_norm'] == 0:
                raise ValueError('Fresh task head must have nonzero token weights')
    _validate_contract(env, agent, state, info)
    _validate_metadata_contract(agent, metadata)
    replay = replay or ReplayBuffer()
    metrics = metrics or TrainMetrics(
        start_episode=start_episode, epsilon_decay_steps=epsilon_decay_steps
    )
    if metrics.epsilon_decay_steps != epsilon_decay_steps:
        raise ValueError(
            "Resumed checkpoint uses a different epsilon_decay_steps contract"
        )
    initial_transitions = metrics.transitions
    initial_updates = agent.updates
    initial_reward_count = len(metrics.episode_rewards)
    paused = False
    if evaluation_env is not None:
        evaluation_env._pause_requested = stop_requested

    for local_episode in range(episodes):
        episode = start_episode + local_episode
        if local_episode:
            state, info = env.reset()
            _validate_contract(env, agent, state, info)
        total = 0.0
        length = 0
        source_case = info.get('source_case')
        if info["converged"]:
            metrics.skipped_converged += 1
            metrics.episode_rewards.append(0.0)
            metrics.episode_lengths.append(0)
        else:
            for _ in range(max_steps):
                epsilon = epsilon_at_step(
                    metrics.global_environment_steps, metrics.epsilon_decay_steps
                )
                action = agent.choose_action(state, info["action_mask"], epsilon)
                next_state, reward, terminated, truncated, next_info = env.step(action)
                _validate_contract(env, agent, next_state, next_info)
                done = terminated or truncated
                replay.add(Transition(state, action, reward, next_state, done, next_info["action_mask"], source_case, length + 1))
                metrics.transitions += 1
                metrics.global_environment_steps += 1
                loss = agent.learn(replay, batch_size)
                if loss is not None:
                    metrics.losses.append(loss)
                state, info = next_state, next_info
                total += reward
                length += 1
                if terminated:
                    metrics.successes += 1
                if done or not info["action_mask"].any():
                    break
            metrics.episode_rewards.append(float(total))
            metrics.episode_lengths.append(length)
        metrics.completed_episodes += 1
        metrics.updates = agent.updates
        completed = episode + 1
        paused = bool(stop_requested is not None and stop_requested())
        agent.runtime_state = {
            'env_rng': copy.deepcopy(env._rng.bit_generator.state) if hasattr(env, '_rng') else None,
            'next_case_index': getattr(env, 'current_case_index', None),
        }
        if hasattr(env, 'foundation_encoder') and checkpoint_dir:
            atomic_json(Path(checkpoint_dir).parent / 'representation_audit.json',
                        getattr(agent, 'representation_audit', {'updates': 0}))
            with (Path(checkpoint_dir).parent / 'representation_audit.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'episode': completed,
                                        **getattr(agent, 'representation_audit', {'updates': 0})}, allow_nan=False) + '\n')
        if hasattr(env, 'foundation_encoder') and (completed % checkpoint_interval == 0 if checkpoint_interval > 0 else False):
            env.foundation_encoder.verify_frozen()
        full_due = (
            not paused and evaluation_env is not None
            and (
                local_episode == episodes - 1
                or (full_evaluation_interval > 0 and completed % full_evaluation_interval == 0)
            )
        )
        if full_due:
            try:
                _record_validation(agent, evaluation_env, metrics, completed, max_steps,
                                   evaluation_episodes, best_model_path, full=True)
            except EvaluationPaused:
                paused = True
        elif (
            not paused and evaluation_env is not None
            and evaluation_interval > 0
            and completed % evaluation_interval == 0
        ):
            try:
                _record_validation(agent, evaluation_env, metrics, completed, max_steps,
                                   evaluation_episodes, best_model_path, full=False)
            except EvaluationPaused:
                paused = True
        if checkpoint_dir and checkpoint_interval > 0 and completed % checkpoint_interval == 0:
            _save_periodic_checkpoint(
                agent, checkpoint_dir, completed, replay, metrics,
                checkpoint_keep_last,
            )
        if episode_callback:
            episode_callback(completed, metrics)
        if paused:
            if checkpoint_dir is None:
                raise ValueError('Episode-boundary pause requires a checkpoint directory')
            if checkpoint_interval <= 0 or completed % checkpoint_interval != 0:
                _save_periodic_checkpoint(agent, checkpoint_dir, completed, replay, metrics, checkpoint_keep_last)
            break
    if metrics.transitions == initial_transitions and not paused:
        raise RuntimeError(
            "Training produced no transitions; use non-converged cases or enable initial perturbations"
        )
    if agent.updates == initial_updates and not paused:
        raise RuntimeError(
            f"Training collected transitions but performed no gradient update; "
            f"collect at least batch_size={batch_size} transitions"
        )
    agent.training_metrics = metrics
    agent.training_replay = replay
    agent.training_paused = paused
    if hasattr(env, 'foundation_encoder'):
        env.foundation_encoder.verify_frozen()
    return agent, metrics.episode_rewards[initial_reward_count:]


def evaluate(
    agent: DQNAgent,
    env,
    episodes: int,
    max_steps: int = 100,
    start_case: int = 0,
) -> EvaluationMetrics:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    _require_reward_preflight(env)
    _validate_metadata_contract(agent, _environment_metadata(env))
    if 'max_steps' in agent.metadata and agent.metadata['max_steps'] != max_steps:
        raise ValueError('Evaluation horizon differs from training contract')
    rewards: list[float] = []
    lengths: list[int] = []
    case_results: list[CaseEvaluation] = []
    successes = 0
    for episode in range(episodes):
        pause_requested = getattr(env, '_pause_requested', None)
        if pause_requested is not None and pause_requested():
            raise EvaluationPaused('Validation paused before next case; partial result is not eligible for selection')
        case_count = _environment_case_count(env)
        case_index = start_case + episode
        if case_count is not None:
            case_index %= case_count
        state, info = env.reset(case_index=case_index)
        _validate_contract(env, agent, state, info)
        source_case = str(info.get("source_case", case_index))
        total = 0.0
        length = 0
        success = bool(info["converged"])
        if success:
            successes += 1
        else:
            for _ in range(max_steps):
                action = agent.choose_action(state, info["action_mask"], epsilon=0.0)
                state, reward, terminated, truncated, info = env.step(action)
                total += reward
                length += 1
                if terminated:
                    successes += 1
                    success = True
                if terminated or truncated or not info["action_mask"].any():
                    break
        rewards.append(total)
        lengths.append(length)
        raw_error = info.get("error")
        final_error = None if raw_error is None else float(raw_error)
        case_results.append(CaseEvaluation(
            case_index=case_index,
            source_case=source_case,
            success=success,
            reward=float(total),
            steps=length,
            final_error=final_error,
        ))
    finite_errors = [case.final_error for case in case_results if case.final_error is not None]
    return EvaluationMetrics(
        episodes, successes, successes / max(episodes, 1),
        float(np.mean(rewards)), float(np.mean(lengths)),
        None if not finite_errors else float(np.mean(finite_errors)),
        None if not finite_errors else float(np.max(finite_errors)),
        tuple(case_results),
    )


def train_parallel(
    env_factory: Callable[[int], Any],
    episodes: int = 100,
    workers: int = 2,
    max_steps: int = 100,
    batch_size: int = 64,
    seed: int = 0,
    device: str | torch.device | None = None,
    agent: DQNAgent | None = None,
    replay: ReplayBuffer | None = None,
    metrics: TrainMetrics | None = None,
    start_episode: int = 0,
    checkpoint_dir: Path | str | None = None,
    checkpoint_interval: int = 10,
    checkpoint_keep_last: int = 3,
    evaluation_env=None,
    evaluation_interval: int = 0,
    evaluation_episodes: int = 1,
    full_evaluation_interval: int = 0,
    best_model_path: Path | str | None = None,
    epsilon_decay_steps: int = 10_000,
) -> tuple[DQNAgent, list[float]]:
    """Collect solver-bound episodes in isolated environments concurrently.

    Network weights remain frozen during each rollout batch and learning is
    serialized afterwards, avoiding concurrent optimizer mutation.
    """
    if workers < 2:
        raise ValueError("train_parallel requires at least two workers")
    if checkpoint_keep_last < 0:
        raise ValueError("checkpoint_keep_last cannot be negative")
    if start_episode == 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    probe = env_factory(0)
    try:
        require_training_contract(probe, agent, start_episode, replay, metrics)
        if hasattr(probe, 'foundation_encoder'):
            raise ValueError('Unified GridEncoder production training requires one worker')
        _require_reward_preflight(probe)
        state, info = probe.reset(seed=seed)
        metadata = _environment_metadata(probe)
        agent = agent or DQNAgent(len(state), probe.action_size, device=device, metadata=metadata)
        _validate_contract(probe, agent, state, info)
        _validate_metadata_contract(agent, metadata)
    finally:
        probe.close()
    replay = replay or ReplayBuffer()
    metrics = metrics or TrainMetrics(
        start_episode=start_episode, epsilon_decay_steps=epsilon_decay_steps
    )
    if metrics.epsilon_decay_steps != epsilon_decay_steps:
        raise ValueError(
            "Resumed checkpoint uses a different epsilon_decay_steps contract"
        )
    initial_transitions = metrics.transitions
    initial_updates = agent.updates

    def rollout(job: tuple[int, int]):
        episode, rollout_global_step = job
        env = env_factory(episode % workers)
        local_rng = np.random.default_rng(seed + episode)
        local_python_rng = random.Random(seed + episode)
        transitions: list[Transition] = []
        total = 0.0
        success = False
        try:
            reset_kwargs = {"seed": seed + episode}
            if hasattr(env, "cases") and len(env.cases):
                reset_kwargs["case_index"] = episode % len(env.cases)
            state, info = env.reset(**reset_kwargs)
            _validate_contract(env, agent, state, info)
            if info["converged"]:
                return transitions, total, 0, success, True
            for local_step in range(max_steps):
                epsilon = epsilon_at_step(
                    rollout_global_step + local_step, metrics.epsilon_decay_steps
                )
                valid = np.flatnonzero(info["action_mask"])
                if not len(valid):
                    break
                if local_rng.random() < epsilon:
                    action = int(local_rng.choice(valid))
                else:
                    action = agent.choose_action(
                        state, info["action_mask"], epsilon=0.0, rng=local_python_rng
                    )
                next_state, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                transitions.append(Transition(state, action, reward, next_state, done, next_info["action_mask"]))
                state, info = next_state, next_info
                total += reward
                if terminated:
                    success = True
                if done:
                    break
            return transitions, total, len(transitions), success, False
        finally:
            env.close()

    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="psasp-rollout") as pool:
        while completed < episodes:
            count = min(workers, episodes - completed)
            episode_numbers = [start_episode + completed + index for index in range(count)]
            jobs = [
                (episode, metrics.global_environment_steps + index * max_steps)
                for index, episode in enumerate(episode_numbers)
            ]
            results = list(pool.map(rollout, jobs))
            for transitions, reward, length, success, skipped in results:
                if skipped:
                    metrics.skipped_converged += 1
                for transition in transitions:
                    replay.add(transition)
                    metrics.transitions += 1
                    metrics.global_environment_steps += 1
                    loss = agent.learn(replay, batch_size)
                    if loss is not None:
                        metrics.losses.append(loss)
                metrics.episode_rewards.append(float(reward))
                metrics.episode_lengths.append(length)
                metrics.successes += int(success)
                metrics.completed_episodes += 1
            completed += count
            metrics.updates = agent.updates
            absolute_episode = start_episode + completed
            full_due = (
                evaluation_env is not None
                and (
                    completed == episodes
                    or (
                        full_evaluation_interval > 0
                        and absolute_episode % full_evaluation_interval < count
                    )
                )
            )
            if full_due:
                _record_validation(
                    agent, evaluation_env, metrics, absolute_episode, max_steps,
                    evaluation_episodes, best_model_path, full=True,
                )
            elif (
                evaluation_env is not None
                and evaluation_interval > 0
                and absolute_episode % evaluation_interval < count
            ):
                _record_validation(
                    agent, evaluation_env, metrics, absolute_episode, max_steps,
                    evaluation_episodes, best_model_path, full=False,
                )
            if (
                checkpoint_dir is not None
                and checkpoint_interval > 0
                and absolute_episode % checkpoint_interval < count
            ):
                _save_periodic_checkpoint(
                    agent, checkpoint_dir, absolute_episode, replay, metrics,
                    checkpoint_keep_last,
                )
    if metrics.transitions == initial_transitions:
        raise RuntimeError("Parallel training produced no transitions")
    if agent.updates == initial_updates:
        raise RuntimeError(f"Parallel training performed no update; collect at least {batch_size} transitions")
    agent.training_metrics = metrics
    agent.training_replay = replay
    return agent, metrics.episode_rewards[-episodes:]
