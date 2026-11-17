"""Read-only audits of task-head dependency on the frozen representation."""
import hashlib
import torch


def network_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def token_sensitivity(network, states, masks, token_size=3072):
    baseline = network(states)
    zeroed = states.clone()
    zeroed[:, -token_size:] = 0
    permuted = states.clone()
    permuted[:, -token_size:] = states[:, -token_size:].roll(1, dims=0)
    actions = baseline.masked_fill(~masks, -torch.inf).argmax(1)
    result = {'samples': len(states), 'token_norm_mean': float(states[:, -token_size:].norm(dim=1).mean()),
              'token_variance_mean': float(states[:, -token_size:].var(dim=0, unbiased=False).mean())}
    for label, inputs in (('zero', zeroed), ('permuted', permuted)):
        values = network(inputs)
        result[label + '_q_max_abs'] = float((values - baseline).abs().max())
        result[label + '_action_changes'] = int((values.masked_fill(~masks, -torch.inf).argmax(1) != actions).sum())
    result['interpretation'] = 'Computational dependency diagnostic; not evidence of PSASP performance gain'
    return result
