"""Production training gates shared by CLI and Python entry points."""


def require_training_contract(env, agent=None, start_episode=0, replay=None, metrics=None):
    from .environment import PowerAdjustmentEnv
    from .foundation import FoundationPowerAdjustmentEnv, UnifiedFrozenEncoder
    if agent is not None and agent.metadata.get('foundation_usage', {}).get('inference_only'):
        raise ValueError('Inference-only foundation bridge cannot train')
    if not isinstance(env, PowerAdjustmentEnv):
        # Generic mathematical test environments are not PSASP production environments.
        return
    if not isinstance(env, FoundationPowerAdjustmentEnv):
        raise ValueError('Production training requires unified frozen GridEncoder environment')
    if not isinstance(env.foundation_encoder, UnifiedFrozenEncoder):
        raise ValueError('Production training requires real UnifiedFrozenEncoder')
    env.foundation_encoder.verify_frozen()
    if env.observation_size != 20168 or env.action_size != 302:
        raise ValueError('Production training requires 20168 observation / 302 actions')
    if start_episode == 0:
        if agent is not None or replay is not None or metrics is not None:
            raise ValueError('Fresh training requires new Q network, optimizer and empty replay')
    else:
        if agent is None or replay is None or metrics is None:
            raise ValueError('Resume requires complete agent/replay/metrics')
        if agent.metadata.get('training_origin') != 'fresh_unified_gridencoder':
            raise ValueError('Resume requires a freshly trained unified GridEncoder task head')
        if not agent.metadata.get('initial_network_sha256') or agent.metadata.get('initial_token_weight_norm', 0) <= 0:
            raise ValueError('Resume requires fresh token-weight initialization evidence')
        if agent.metadata.get('encoder') != env.encoder_contract:
            raise ValueError('Resume encoder contract mismatch')
        if agent.metadata.get('max_steps') != env.max_steps:
            raise ValueError('Resume horizon mismatch')
        if agent.metadata.get('registry') != env.registry.to_dict():
            raise ValueError('Resume registry mismatch')
        if agent.metadata.get('reward') != env.reward_contract:
            raise ValueError('Resume reward mismatch')
        if metrics.completed_episodes != start_episode or agent.updates != metrics.updates:
            raise ValueError('Resume progress mismatch')
