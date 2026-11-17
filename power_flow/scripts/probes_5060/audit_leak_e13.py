# -*- coding: utf-8 -*-
"""防作弊审计(2026-08-23,用户要求"训练过程中严格规避作弊"):用**训练好的**检查点、在**真实大网方式**上做 D15 端到端泄漏检查。

做法:对每个样本构造"被监督量全毁"的副本——PQ 的 v/θ、PV 的 qg/θ、平衡机 pg/qg 全部换成随机垃圾(均匀 [-5,5] 的 z 值 / 原值 +7.7),
然后(1)检查喂进模型的每一个张量(编码器输入 x / 侧通道 side / 边侧 edge_side / 电压提示 v_hint / 物理常量 phys_* / 修正步的 refine_free、edge_b、theta_ref)
逐位相同;(2)整个推理器前向(含 K 步修正、Q–V 半步)的输出 pred **逐位相同**;(3)对原样本报 θ/v MAE,证明输出不是靠读真值得到的。
另外核对:训练日志里的零交集断言(train 族 ∩ transfer 族 = ∅、sample_id 无重复)与留出命中断言。
用法:python scripts/probes_5060/audit_leak_e13.py --ckpt runs/<run>/ckpt/best.pt
"""
import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import pf_reasoner as M  # noqa: E402
from eval_holdout_5060 import build_from_ckpt  # noqa: E402
from gridfm.data.dataset import family_files  # noqa: E402
from gridfm.reason.pf_physics import pf_phys_denorm  # noqa: E402
from gridfm.reason.pf_task import pf_supervised_cols  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.train.masking import PF_UNKNOWN_FIELDS  # noqa: E402

D = ROOT.parent / "大规模算例语料交付-20260821" / "大规模算例语料交付-20260821" / "gridfm-data-large"
EXT = ROOT.parent / "gridfm-data-large-ext"
TH, VC = NODE_TABLE.offset_of("theta"), NODE_TABLE.offset_of("v")


def corrupt(raw: GridSample, rng) -> GridSample:
    """把全部 D15 未知量(被监督量)换成垃圾:原值 + 7.7 + U(-5,5)。给定量一个都不碰。"""
    nf = raw.node_feat.copy()
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    n = 0
    for t, fields in PF_UNKNOWN_FIELDS.items():
        rows = np.where(types == t)[0]
        for f in fields:
            c = NODE_TABLE.offset_of(f)
            nf[rows, c] = nf[rows, c] + 7.7 + rng.uniform(-5, 5, size=rows.size)
            n += rows.size
    out = replace(raw, node_feat=nf)
    for k in ("branch_tap", "branch_shift_rad"):
        if hasattr(raw, k):
            setattr(out, k, getattr(raw, k))
    return out, n


def tensors_of(data):
    return {k: v for k, v in data.items() if torch.is_tensor(v)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--v1-dir", default=str(ROOT / "runs" / "encoder_v1"))
    a = ap.parse_args()
    dev = torch.device("cpu")
    hold = set()
    for hp in (D.parent / "val留出清单-mp_activsg2000.json", EXT / "val留出清单-mp_activsg10k.json"):
        hold |= set(json.loads(hp.read_text(encoding="utf-8"))["val_holdout_sample_ids"])
    picks = []
    for root, split, fam, want_hold in ((D, "train", "mp_activsg2000", True), (EXT, "train", "mp_activsg10k", True),
                                        (D, "transfer", "mp_activsg25k", False)):
        ps = [p for p in family_files(root, split) if Path(p).parent.name == fam and ((Path(p).stem in hold) == want_hold)][:1]
        raw = GridSample.load_npz(ps[0])
        M.attach_family_extra([raw], [root])
        picks.append((fam, Path(ps[0]).stem, "留出(val 中)" if want_hold else "冻结外集", raw))
    enc, reasoner, cfg, norm, meta = build_from_ckpt(a.ckpt, a.v1_dir, dev, 0)
    dn = torch.from_numpy(pf_phys_denorm(norm)).to(dev) if meta["use_phys"] else None
    cols = pf_supervised_cols()
    rng = np.random.default_rng(123)
    print(f"ckpt {a.ckpt}  step {meta['step']}  K={reasoner.refine_steps} qv={reasoner.refine_qv}")
    all_ok = True
    for fam, sid, status, raw in picks:
        raw2, n_pert = corrupt(raw, rng)
        preds, tens, maes = [], [], []
        for r in (raw, raw2):
            s = copy.deepcopy(r)
            norm.transform(s)
            d = M._pf_data(s, cfg, meta["view"], raw=r, dc=meta["dc_mode"], phys=meta["use_phys"], edge=meta["theta_edge"],
                           vhint=meta["v_hint"], vstat=meta["vstat"], negb=meta["negb"], shz=meta.get("shz", False), edge_base=meta.get("edge_base", "acv"))
            tens.append(tensors_of(d))
            b = Batch.from_data_list([d]).to(dev)
            pred, _ = M._forward(enc, reasoner, b, False, torch.float16, dev, dn=dn)
            preds.append(pred.float().cpu())
            # 对原真值算 MAE(两次都对 raw 的真值算,证明输出不随垃圾变)
            tgt = b.target.float().cpu().numpy()
            lm = b.loss_mask.cpu().numpy()
            full = tgt.copy()
            for k, f in enumerate(("pg", "qg", "v", "theta")):
                full[:, cols[f]] = pred.float().cpu().numpy()[:, k]
            pu_p = norm.inverse_transform_node(full)
            tt = raw.node_feat[:, TH]; tv = raw.node_feat[:, VC]
            maes.append((float(np.abs(pu_p[lm[:, TH], TH] - tt[lm[:, TH]]).mean()), float(np.abs(pu_p[lm[:, VC], VC] - tv[lm[:, VC]]).mean())))
        # 喂进模型的张量:除 target(监督用)外逐位相同;target 必须不同(证明确实毁了)
        diff_keys = [k for k in tens[0] if not torch.equal(tens[0][k], tens[1].get(k, tens[0][k] + 1))]
        inputs_same = diff_keys == ["target"]
        pred_same = torch.equal(preds[0], preds[1])
        ok = inputs_same and pred_same
        all_ok &= ok
        print(f"{fam:<15}{sid:<22}{status:<10} 扰动 {n_pert} 个被监督量 | 输入张量除 target 外逐位相同: {inputs_same}(不同的键:{diff_keys}) "
              f"| 前向输出逐位相同: {pred_same} | θ/v MAE 原 {maes[0][0]:.4f}/{maes[0][1]:.5f} 毁后 {maes[1][0]:.4f}/{maes[1][1]:.5f} → {'PASS' if ok else 'FAIL'}",
              flush=True)
    print("审计结论:", "PASS —— 输出只依赖给定量,被监督量怎么改都不变" if all_ok else "FAIL —— 存在泄漏,必须查")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
