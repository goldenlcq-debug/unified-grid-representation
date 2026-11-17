"""E12 消融探针(诊断,不进判据):同一检查点上,把修正步的 α / β 置 0,看 θ⁰(势重建输出)与 V⁰ 本身的质量,
以区分"修正步把 θ 弄坏"与"边头学得慢"。在留出 44(分族 + 合计)与 --syn-val 个合成 val 方式上评。

用法(grid-foundation/ 下):
  python scripts/probes_5060/probe_refine_ablate.py --ckpt runs/<run>/ckpt/step_000250.pt --device cpu --syn-val 60
"""
import argparse
import copy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import pf_reasoner as M  # noqa: E402
from eval_holdout_5060 import build_from_ckpt, load_holdout  # noqa: E402
from gridfm.data.dataset import load_families  # noqa: E402
from gridfm.reason.pf_physics import pf_phys_denorm  # noqa: E402

D = "../大规模算例语料交付-20260821/大规模算例语料交付-20260821"
MODES = {"full": (None, None), "a0": (0.0, None), "b0": (None, 0.0), "a0b0": (0.0, 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--extra-root", action="append", default=[f"{D}/gridfm-data-large", "../gridfm-data-large-ext"])
    ap.add_argument("--val-holdout", action="append",
                    default=[f"{D}/val留出清单-mp_activsg2000.json", "../gridfm-data-large-ext/val留出清单-mp_activsg10k.json"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--syn-val", type=int, default=60)
    ap.add_argument("--modes", default="full,a0,b0,a0b0")
    ap.add_argument("--v1-dir", default=str(ROOT / "runs" / "encoder_v1"))
    a = ap.parse_args()
    raws = load_holdout(a.extra_root, a.val_holdout)
    M.attach_family_extra(raws, list(a.extra_root))
    fams = sorted({s.family_id for s in raws})
    groups = [(f, [s for s in raws if s.family_id == f]) for f in fams] + [("留出44", raws)]
    if a.syn_val:
        syn = load_families(Path("../gridfm-data"), "val")[: a.syn_val]
        groups.append((f"合成val({len(syn)})", syn))
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    for ckpt in a.ckpt:
        enc, reasoner, cfg, norm, meta = build_from_ckpt(ckpt, a.v1_dir, dev, 0)
        dn = torch.from_numpy(pf_phys_denorm(norm)).to(dev) if meta["use_phys"] else None
        alpha0 = reasoner.refine_alpha.detach().clone() if hasattr(reasoner, "refine_alpha") else None
        beta0 = reasoner.refine_beta.detach().clone() if hasattr(reasoner, "refine_beta") else None
        print(f"\n=== {ckpt}  step {meta['step']}  α={None if alpha0 is None else [round(float(x), 3) for x in alpha0]} "
              f"β={None if beta0 is None else [round(float(x), 3) for x in beta0]}")
        for mode in a.modes.split(","):
            av, bv = MODES[mode]
            with torch.no_grad():
                if alpha0 is not None:
                    reasoner.refine_alpha.copy_(alpha0 if av is None else torch.full_like(alpha0, av))
                if beta0 is not None:
                    reasoner.refine_beta.copy_(beta0 if bv is None else torch.full_like(beta0, bv))
            for gname, grp in groups:
                gnorm = [copy.deepcopy(s) for s in grp]
                for s in gnorm:
                    norm.transform(s)
                mae = M.eval_pu_mae(enc, reasoner, gnorm, cfg, norm, dev, dev.type == "cuda", torch.float16, bs=a.bs,
                                    view=meta["view"], raws=grp if meta["use_raw"] else None, dn=dn, dc=meta["dc_mode"],
                                    phys=meta["use_phys"], edge=meta["theta_edge"], vhint=meta["v_hint"], vstat=meta["vstat"], edge_base=meta.get("edge_base", "acv"),
                                    negb=meta["negb"], shz=meta.get("shz", False))
                print(f"{mode:<6}{gname:<16} n={len(grp):<4} θ {mae['theta']:.4f}  v {mae['v']:.5f}  "
                      f"|ΔPQ|clean {mae.get('phys_clean', float('nan')):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
