"""对拍:推理器 `_refine`(批路径、稀疏 LU、refine_free、0.5 因子)从带网损 DC 起点 + 神谕 V 走 K 步,是否与 numpy 探针(probe_newton_big v2,sign 口径)
在同一 25k / 70k 方式上给出同样的 θ 误差序列。若不一致 → 推理器大图路径有 bug;若一致 → 模型 θ⁰/V⁰ 的差异才是 E12 步子比探针弱的原因。
用法:python scripts/probes_5060/probe_refine_vs_probe.py --fams mp_activsg25k --cap 2 --steps 3
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import pf_reasoner as M  # noqa: E402
from probe_newton_big import run as probe_run  # noqa: E402
from gridfm.data.dataset import family_files  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.pf_physics import dc_theta_lossy, pf_phys_denorm  # noqa: E402
from gridfm.reason.pf_task import PF_FIELDS, PFReasoner  # noqa: E402
from gridfm.reason.pf_theta_edge import EDGE_SIDE_DIM  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.train.pretrain import TrainConfig  # noqa: E402

D = Path("../大规模算例语料交付-20260821/大规模算例语料交付-20260821/gridfm-data-large")
TH, VC = NODE_TABLE.offset_of("theta"), NODE_TABLE.offset_of("v")
K_TH = PF_FIELDS.index("theta")


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", default="mp_activsg25k")
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    norm = Normalizer.load(ROOT / "runs" / "encoder_v1" / "normalizer.npz")
    vck = torch.load(ROOT / "runs" / "encoder_v1" / "v1.pt", map_location="cpu", weights_only=False)
    from dataclasses import fields as dc_fields
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in vck["config"].items() if k in allowed})
    vstat = (float(norm.node_mean[VC]), float(norm.node_std[VC]))
    dn = torch.from_numpy(pf_phys_denorm(norm))
    for fam in a.fams.split(","):
        ps = [p for p in family_files(D, "transfer") if Path(p).parent.name == fam][: a.cap]
        raws = [GridSample.load_npz(p) for p in ps]
        M.attach_family_extra(raws, [D])
        z = np.load(D / "families" / "transfer" / fam / "family_extra.npz")
        tap, shift = z["branch_tap"].astype(np.float64), z["branch_shift_rad"].astype(np.float64)
        for i, raw in enumerate(raws):
            # numpy 探针(神谕 V,sign 口径,不带 Q–V)
            pr = probe_run(raw, tap, shift, "oracle", steps=a.steps, qv=False, negb="sign")
            seq_np = [pr["hist"][0]["th_wrap"]] + [h.get("th_wrap_a1", float("nan")) for h in pr["hist"][1:]]
            # 推理器 _refine:θ⁰ = 带网损 DC,pred = 真值 z,α 全 1,K = steps,不带 Q–V
            s = copy.deepcopy(raw); norm.transform(s)
            # 注:E12 期探针,配方硬编码(dc="lossy",无 shz/edge_base)—— 与 E15 起的主线口径本就不同源,
            # 只用于当年那批检查点的横向对照,不随新开关同步(2026-08-24 E19 审查确认,故意不改)。
            d = M._pf_data(s, cfg, view="group", raw=raw, dc="lossy", phys=True, edge=True, vhint=True, vstat=vstat)
            b = Batch.from_data_list([d])
            torch.manual_seed(0)
            r = PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=b.side.shape[1], gnn_layers=1, theta_head=True,
                           edge_side_dim=EDGE_SIDE_DIM, theta_edge=True, theta_wrap=True, v_hint=True, refine_steps=a.steps)
            phys = dict(index=b.phys_index, g=b.phys_g, b=b.phys_b, denorm=dn, target=b.target, loss_mask=b.loss_mask,
                        eqp_clean=b.eqp_clean, eqq_clean=b.eqq_clean, eqq_cand=b.eqq_cand, edge_b=b.edge_b, refine_free=b.refine_free)
            cols = [NODE_TABLE.offset_of(f) for f in PF_FIELDS]
            pred0 = b.target[:, cols].clone().float()
            dcl = dc_theta_lossy(raw)
            th0 = torch.from_numpy(dcl[:, 0].astype(np.float32))
            tt = raw.node_feat[:, TH].astype(np.float64)
            m = b.loss_mask[:, TH].numpy() & b.reach_mask.numpy()
            seq_r = []
            with torch.no_grad():
                for k in range(a.steps + 1):
                    r.refine_steps = k
                    r.refine_alpha = torch.nn.Parameter(torch.ones(max(k, 1)))
                    th, _ = (th0, None) if k == 0 else r._refine(th0, pred0, K_TH, phys, b.edge_index, b.batch, b.ref_mask, b.reach_mask, 1)
                    e = float(np.abs(wrap(th.numpy().astype(np.float64)[m] - tt[m])).mean())
                    seq_r.append(e)
                    if k:
                        st = r.refine_stats
                        print(f"   K={k}: 闸 covP {st['cov_skip_p']} divP {st['div_skip_p']} max|Δθ| {st['max_dth']:.3f}")
            print(f"{fam} #{i}  探针(numpy) θ: {['%.4f' % x for x in seq_np]}   推理器 _refine θ: {['%.4f' % x for x in seq_r]}"
                  f"   覆盖率P {pr['cov_p']:.4f} refine_free {int(b.refine_free.sum())}/{int(b.reach_mask.sum())}", flush=True)


if __name__ == "__main__":
    main()
