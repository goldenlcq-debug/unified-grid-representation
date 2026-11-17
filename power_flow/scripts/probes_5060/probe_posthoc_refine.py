"""对照实验:**后迭代**(post-hoc)vs **展开式**(unrolled,E12/E13 现行做法)。诊断用,不进判据。

区别:
- 展开式(现行):训练时就带着 K 步物理修正,梯度穿过它反传,步长 α/β 可学习 → 网络"知道"后面有修正,会偷懒(§3.27(a)①);
- 后迭代(本脚本):拿一个**训练时完全不知道有修正步**的检查点(E10 / E7),推理时才临时装上 K 步修正(α=β=1 固定)。

这直接回答:"训练时把求解器耦合进去,比训完再跑几步迭代,到底多赚了什么?"
用法:python scripts/probes_5060/probe_posthoc_refine.py --ckpt runs/pf_reasoner_5060_R1dcL_edge0_wrap_L_vh_s0/ckpt/best.pt --K 4
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import pf_reasoner as M  # noqa: E402
from eval_holdout_5060 import build_from_ckpt, load_holdout  # noqa: E402
from gridfm.reason.pf_physics import pf_phys_denorm  # noqa: E402

D = "../大规模算例语料交付-20260821/大规模算例语料交付-20260821"


def attach_refine(reasoner, K, qv=True, alpha=1.0, beta=1.0):
    """给一个训练时 refine_steps=0 的推理器**临时装上** K 步物理修正(固定步长,不训练)。"""
    reasoner.refine_steps = int(K)
    reasoner.refine_qv = bool(qv)
    reasoner.refine_alpha = nn.Parameter(torch.full((max(K, 1),), float(alpha)))
    if qv:
        reasoner.refine_beta = nn.Parameter(torch.full((max(K, 1),), float(beta)))
    for k, v in (("refine_cov", 0.99), ("refine_dth_cap", 1.0), ("refine_dv_cap", 0.5), ("refine_ls", False)):
        setattr(reasoner, k, v)
    reasoner.refine_stats = {}
    return reasoner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--K", default="0,1,2,4", help="临时装上的修正步数(逗号分隔;0 = 原样)")
    ap.add_argument("--extra-root", action="append", default=[f"{D}/gridfm-data-large", "../gridfm-data-large-ext"])
    ap.add_argument("--val-holdout", action="append",
                    default=[f"{D}/val留出清单-mp_activsg2000.json", "../gridfm-data-large-ext/val留出清单-mp_activsg10k.json"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--v1-dir", default=str(ROOT / "runs" / "encoder_v1"))
    ap.add_argument("--big", default=None, help="改评冻结外集家族(逗号分隔,如 mp_activsg25k,mp_activsg70k),每族前 --cap 个方式")
    ap.add_argument("--cap", type=int, default=3)
    a = ap.parse_args()
    if a.big:
        from gridfm.data.dataset import family_files
        from gridfm.schema.sample import GridSample
        groups = []
        for fam in a.big.split(","):
            ps = [q for q in family_files(Path(f"{D}/gridfm-data-large"), "transfer") if Path(q).parent.name == fam][: a.cap]
            rs = [GridSample.load_npz(q) for q in ps]
            M.attach_family_extra(rs, [f"{D}/gridfm-data-large"])
            groups.append((fam, rs))
    else:
        raws = load_holdout(a.extra_root, a.val_holdout)
        M.attach_family_extra(raws, list(a.extra_root))
        fams = sorted({s.family_id for s in raws})
        groups = [(f, [s for s in raws if s.family_id == f]) for f in fams] + [("留出44", raws)]
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    for ckpt in a.ckpt:
        enc, reasoner, cfg, norm, meta = build_from_ckpt(ckpt, a.v1_dir, dev, 0)
        dn = torch.from_numpy(pf_phys_denorm(norm)).to(dev) if meta["use_phys"] else None
        born = reasoner.refine_steps                       # 该检查点**训练时**的步数
        print(f"\n=== {ckpt}  step {meta['step']}  训练时 K={born}  (K>born 即后迭代:训练时没有、推理时临时装上)", flush=True)
        for K in [int(x) for x in a.K.split(",")]:
            attach_refine(reasoner, K, qv=True)
            for gname, grp in groups:
                gnorm = [copy.deepcopy(s) for s in grp]
                for s in gnorm:
                    norm.transform(s)
                t0 = time.time()
                mae = M.eval_pu_mae(enc, reasoner, gnorm, cfg, norm, dev, dev.type == "cuda", torch.float16, bs=a.bs,
                                    view=meta["view"], raws=grp if meta["use_raw"] else None, dn=dn, dc=meta["dc_mode"],
                                    phys=meta["use_phys"], edge=meta["theta_edge"], vhint=meta["v_hint"], edge_base=meta.get("edge_base", "acv"),
                                    vstat=meta["vstat"], negb=meta["negb"], shz=meta.get("shz", False))
                tag = "原样" if K == born else f"后迭代 K={K}"
                print(f"  {tag:<12}{gname:<16} n={len(grp):<4} θ {mae['theta']:.5f} rad ({mae['theta']*57.2958:.4f}°)  "
                      f"v {mae['v']:.6f}  |ΔPQ| {mae.get('phys_clean', float('nan')):.4f}  {time.time()-t0:.0f}s", flush=True)
        del enc, reasoner
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
