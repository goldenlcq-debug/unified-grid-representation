"""DDPM 数学件(6.4 定死项:T=1000、cosine 噪声调度、ε 预测;采样 DDIM)。

约定:t ∈ {1..T} 为加噪步(t=0 = 干净数据);alpha_bar 定义在整数格点 0..T,
alpha_bar[0]=1。扩散状态 z = 标准化后的节点 token 场(6.3②:进扩散前按
latent_stats 逐通道标准化,解码前逆变换)。
"""
from __future__ import annotations

import math
from typing import Callable

import torch


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """[T+1] 单调递减,abar[0]=1(Nichol & Dhariwal cosine 调度)。"""
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).clamp(min=1e-8).float()


class GaussianDiffusion:
    def __init__(self, T: int = 1000) -> None:
        self.T = T
        self.alpha_bar = cosine_alpha_bar(T)

    def to(self, device) -> "GaussianDiffusion":
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, z0: torch.Tensor, t_node: torch.Tensor,
                 eps: torch.Tensor) -> torch.Tensor:
        """z0/eps [N,d];t_node [N](每节点已展开的整数步 1..T)。"""
        ab = self.alpha_bar[t_node][:, None]
        return ab.sqrt() * z0 + (1 - ab).sqrt() * eps

    @torch.no_grad()
    def ddim_sample(self, eps_fn: Callable[[torch.Tensor, int], torch.Tensor],
                    z_T: torch.Tensor, steps: int = 50,
                    clip_x0: float = None) -> torch.Tensor:
        """确定性 DDIM(eta=0)。eps_fn(z_t, t) → ε̂ [N,d](批内同一 t,由调用方
        负责按图展开与 CFG 组合);返回 z0。

        clip_x0(静态阈值,static thresholding):每步把 x0 预测夹到 [-clip_x0,
        clip_x0]。ε-参数化 DDIM 在高 t(cosine ᾱ[T]≈1e-8)的 x0 换算 = (z−√(1−ᾱ)ε)
        /√ᾱ 会把 ε 的微小误差放大 ~1e4 倍致数值爆炸;标准化潜码实测幅度 ≤~6.3,
        取 clip_x0=8 只斩爆炸不削真实结构(SD/Imagen 同类做法)。None=不夹(单测
        神谕路径用,ε 精确无爆炸)。"""
        taus = torch.linspace(self.T, 1, steps).round().long().tolist()
        z = z_T
        for i, tau in enumerate(taus):
            eps = eps_fn(z, int(tau))
            ab_t = self.alpha_bar[tau]
            z0 = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clip_x0 is not None:
                z0 = z0.clamp(-clip_x0, clip_x0)
            ab_n = self.alpha_bar[taus[i + 1]] if i + 1 < len(taus) \
                else self.alpha_bar[0]
            z = ab_n.sqrt() * z0 + (1 - ab_n).sqrt() * eps
        return z
