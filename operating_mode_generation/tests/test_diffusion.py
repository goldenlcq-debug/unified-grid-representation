"""扩散数学件与去噪器单测(6.2/6.4;CPU 秒级)。"""
import torch

from gridfm.models.diffusion import (DenoiserV0, GaussianDiffusion,
                                     LatentRegressorV0, cosine_alpha_bar)
from gridfm.models.diffusion.denoiser import (LABEL_DIM, LABEL_MISSING_BIT,
                                              label_vec)

D, N_NODE, N_EDGE, B = 32, 10, 24, 3


def _fake_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    n, m = B * N_NODE, B * N_EDGE
    z = torch.randn(n, D, generator=g)
    ct = torch.randn(n, D, generator=g)
    ec = torch.randn(m, D, generator=g)
    ei1 = torch.randint(0, N_NODE, (2, N_EDGE), generator=g)
    ei = torch.cat([ei1 + k * N_NODE for k in range(B)], dim=1)
    t = torch.randint(1, 1001, (B,), generator=g)
    lab = torch.stack([label_vec(0), label_vec(2), label_vec(None)])
    npg = torch.full((B,), N_NODE, dtype=torch.long)
    return z, ct, ec, ei, t, lab, npg


def test_cosine_alpha_bar_monotone():
    ab = cosine_alpha_bar(1000)
    assert ab.shape == (1001,)
    assert abs(float(ab[0]) - 1.0) < 1e-6
    assert float(ab[-1]) < 1e-3
    assert bool((ab[1:] <= ab[:-1] + 1e-8).all())


def test_q_sample_limits():
    diff = GaussianDiffusion(1000)
    z0 = torch.randn(50, D)
    eps = torch.randn(50, D)
    z1 = diff.q_sample(z0, torch.full((50,), 1, dtype=torch.long), eps)
    assert float((z1 - z0).abs().mean()) < 0.1          # t=1 几乎无噪
    zT = diff.q_sample(z0, torch.full((50,), 1000, dtype=torch.long), eps)
    assert float((zT - eps).abs().mean()) < 0.1         # t=T 几乎纯噪


def test_ddim_oracle_recovers_z0():
    """ε 神谕(由 z0 反推的真 ε)下,确定性 DDIM 应精确还原 z0。"""
    diff = GaussianDiffusion(1000)
    torch.manual_seed(0)
    z0 = torch.randn(40, D)
    z_T = torch.randn(40, D)

    def oracle(z_t, t):
        ab = diff.alpha_bar[t]
        return (z_t - ab.sqrt() * z0) / (1 - ab).sqrt()

    out = diff.ddim_sample(oracle, z_T, steps=50)
    assert float((out - z0).abs().max()) < 1e-4


def test_ddim_clip_x0_bounds_and_oracle_safe():
    """静态阈值:①ε̂≡0 时 x0=z/√ᾱ 在高 t 爆炸(√ᾱ[T]=1e-4),clip 须把输出夹到
    界内;②神谕路径 + 宽 clip(z0 在 ±5)不改精确还原。"""
    diff = GaussianDiffusion(1000)
    torch.manual_seed(1)
    z_T = torch.randn(30, D)

    def zero_eps(z_t, t):
        return torch.zeros_like(z_t)          # x0=z/√ᾱ,高 t 爆炸

    out_noclip = diff.ddim_sample(zero_eps, z_T, steps=50, clip_x0=None)
    out_clip = diff.ddim_sample(zero_eps, z_T, steps=50, clip_x0=8.0)
    assert float(out_noclip.abs().max()) > 50            # 不夹:x0 爆炸
    assert float(out_clip.abs().max()) <= 8.0 + 1e-3     # 夹住

    z0 = torch.randn(30, D).clamp(-5, 5)
    def oracle(z_t, t):
        ab = diff.alpha_bar[t]
        return (z_t - ab.sqrt() * z0) / (1 - ab).sqrt()
    out_oc = diff.ddim_sample(oracle, z_T, steps=50, clip_x0=8.0)
    assert float((out_oc - z0).abs().max()) < 1e-3       # 宽 clip 不触真 z0


def test_denoiser_shapes_and_adaln_zero():
    torch.manual_seed(0)
    net = DenoiserV0(D, n_layers=2)
    z, ct, ec, ei, t, lab, npg = _fake_batch()
    out = net(z, ct, ec, ei, t, lab, npg)
    assert out.shape == (B * N_NODE, D)
    assert float(out.abs().max()) == 0.0        # adaLN-Zero:初始输出恒 0


def test_label_vec_and_cfg_paths_differ_after_training_step():
    torch.manual_seed(0)
    net = DenoiserV0(D, n_layers=2)
    z, ct, ec, ei, t, lab, npg = _fake_batch()
    # adaLN-Zero 下标签通路的梯度要经 head 二传,单步后仍为零 → 走 3 步
    for _ in range(3):
        net.zero_grad()
        loss = ((net(z, ct, ec, ei, t, lab, npg) - z) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p -= 0.1 * p.grad
    cond = torch.stack([label_vec(1)] * B)
    unc = torch.stack([label_vec(None)] * B)
    assert float(unc[0, LABEL_MISSING_BIT]) == 1.0 and float(unc[0].sum()) == 1.0
    assert cond.shape == (B, LABEL_DIM)
    e_c = net(z, ct, ec, ei, t, cond, npg)
    e_u = net(z, ct, ec, ei, t, unc, npg)
    assert float((e_c - e_u).abs().max()) > 0    # 条件通路生效


def test_regressor_same_params_and_shape():
    torch.manual_seed(0)
    net = DenoiserV0(D, n_layers=2)
    reg = LatentRegressorV0(D, n_layers=2)
    assert reg.num_params() == net.num_params()   # 单变量对照:同类同参数量
    _z, ct, ec, ei, _t, lab, npg = _fake_batch()
    out = reg(ct, ec, ei, lab, npg)
    assert out.shape == (B * N_NODE, D)
