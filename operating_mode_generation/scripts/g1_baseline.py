"""G1-基线(M6/T6.1;方案 6.6 第一档、6.7):数据空间确定性回归对照。

固定拓扑(118)上,小型消息传递网络输入**骨架特征(拓扑/静态组)+ 方式标签
(load_factor)**,直接回归方式变量(pl/ql/pg/v_set)——**不经过编码器/解码器**
(6.7 依赖口径:仅 P0/G0)。生成方式写回 51 卡经 SolverRunner 试算,报告"同标签下
回归输出的裸收敛率 vs 真实方式"(平均化失败演示;**结果如实报告**,若本库单模态
扰动下失败不复现,如实写明并注记多模态方式分布是复现前提)。该报告是 G1-扩散立项依据。

红线自查(2.4):逐节点共享头(无固定 N 输出);无按编号查表的可学习嵌入(输入全为
物理/静态特征);消息沿真实边;无注意力。标签侧通道 = 广播标量拼接,属任务条件输入。

PSASP 窗口纪律:LF 试算是收敛二值(7.3 豁免类),但**不与包 A 的串行复核窗同机同时段**
——脚本开跑试算前检查 runs/packA_v2_logs/chain_status.json,处于 requeue 阶段即拒绝
(--skip-trial 先出拟合报告,稍后 --trial-only 补试算)。

  python scripts/g1_baseline.py --root E:/AI-2/统一计算大模型/gridfm-data [--skip-trial]
  python scripts/g1_baseline.py --root ... --trial-only     # 补试算(读已存预测)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from gridfm.data.dataset import load_library  # noqa: E402
from gridfm.fsutil import robust_copytree, robust_rmtree  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.schema.fields import BRANCH_TABLE, NODE_TABLE  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TARGETS = ("pl", "ql", "pg", "v_set")            # 方式变量(采样器扰动的量)
STATIC_COLS = ("is_null", "has_gen", "has_load", "vbase",
               "pmax", "pmin", "qmax", "qmin")   # 骨架特征(拓扑/静态组,无状态量)


def _chain_stage() -> str:
    p = REPO / "runs" / "packA_v2_logs" / "chain_status.json"
    if not p.exists():
        return "absent"
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        return f"{s.get('stage')}:{s.get('state')}"
    except Exception:  # noqa: BLE001
        return "unreadable"


def _trial_window_ok(stage: str) -> bool:
    """LF 试算窗口:包 A 空闲/暂停/完链,或处于并行豁免的 labels_w2 阶段;
    串行复核(requeue_w1)窗一律避开。"""
    if stage in ("absent", "unreadable"):
        return True
    return ("requeue" not in stage) or ("PAUSED" in stage) or ("complete" in stage)


class _MP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, h, e, ei):
        src, dst = ei[0], ei[1]
        x = self.ln(h)
        m = self.msg(torch.cat([x[src], x[dst], e], dim=-1))
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        return h + self.upd(torch.cat([x, agg], dim=-1))


class G1Net(nn.Module):
    """骨架特征 + 标签 → 方式变量;逐节点共享头(红线 3)。~0.1M 参数。"""

    def __init__(self, node_in: int, edge_in: int, d: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.enc_n = nn.Linear(node_in, d)
        self.enc_e = nn.Linear(edge_in, d)
        self.mp = nn.ModuleList(_MP(d) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, len(TARGETS)))

    def forward(self, x, e_attr, ei):
        h = self.enc_n(x)
        e = self.enc_e(e_attr)
        for mp in self.mp:
            h = mp(h, e, ei)
        return self.head(h)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_arrays(samples, lf_by_sid):
    """静态特征恒定(固定拓扑)→ 只有标签列随样本变;目标堆叠 [S,N,4]。"""
    s0 = samples[0]
    tcol = NODE_TABLE.offset_of("node_type")
    stat = [s0.node_feat[:, tcol:tcol + 3]]
    stat += [s0.node_feat[:, [NODE_TABLE.offset_of(c)]] for c in STATIC_COLS]
    stat = np.concatenate(stat, axis=1).astype(np.float32)          # [N, 11]
    mu, sd = stat.mean(0, keepdims=True), stat.std(0, keepdims=True) + 1e-6
    stat = (stat - mu) / sd
    ei = np.concatenate([s0.edge_index, s0.edge_index[::-1]], axis=1)
    br = [s0.branch_feat[:, [BRANCH_TABLE.offset_of(c)]] for c in ("r", "x", "b_half")]
    br = np.log1p(np.abs(np.concatenate(br, axis=1))).astype(np.float32)
    br = np.concatenate([br, br], axis=0)
    br = (br - br.mean(0, keepdims=True)) / (br.std(0, keepdims=True) + 1e-6)

    types = np.argmax(s0.node_feat[:, tcol:tcol + 3], axis=1)
    has_gen = s0.node_feat[:, NODE_TABLE.offset_of("has_gen")] > 0.5
    has_load = s0.node_feat[:, NODE_TABLE.offset_of("has_load")] > 0.5
    mask = np.zeros((s0.n_nodes, len(TARGETS)), dtype=bool)
    mask[:, 0] = has_load
    mask[:, 1] = has_load
    mask[:, 2] = has_gen & (types != 2)     # 平衡机 pg 由潮流决定,不监督(采样器同款不动)
    mask[:, 3] = has_gen & (types != 2)

    y = np.stack([np.stack([s.node_feat[:, NODE_TABLE.offset_of(f)] for f in TARGETS],
                           axis=1) for s in samples]).astype(np.float32)   # [S,N,4]
    lf = np.array([lf_by_sid[s.sample_id] for s in samples], dtype=np.float32)
    return stat, ei, br, mask, y, lf


def writeback_and_trial(pred_pu, base_dir: Path, work: Path, solver):
    """节点聚合预测 → 51 卡(118 全母线单记录,写回精确)→ LF 试算。"""
    robust_copytree(base_dir, work)
    case = PsaspCase(work)
    for ld in case.loads():
        row = ld.bs_no - 1
        ld.pl = float(pred_pu[row, 0])
        ld.ql = float(pred_pu[row, 1])
    for g in case.gens():
        if g.type_code == 0:
            continue                          # 平衡机不动(采样器同款)
        row = g.bs_no - 1
        newp = float(pred_pu[row, 2])
        if g.pmax > 0:
            newp = min(newp, g.pmax)
        g.pg = max(newp, g.pmin)
        g.v0 = float(pred_pu[row, 3])
    case.save_inplace()
    pf = solver.run_powerflow(work)
    vmin = vmax = None
    if pf.bus_voltage:
        vs = [b.v_pu for b in pf.bus_voltage if abs(b.v_pu) < 3.0]
        if vs:
            vmin, vmax = min(vs), max(vs)
    return bool(pf.converged), vmin, vmax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lib-name", default="lib118",
                    help="方式库目录名(lib118 = 单模态基线;lib118_diverse = 裁决④多模态库)")
    ap.add_argument("--base", default=str(REPO / "tests" / "fixtures" / "ieee118_golden"))
    ap.add_argument("--out", default="runs/g1_baseline")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-every", type=int, default=5,
                    help="按 load_factor 排序每 N 个取 1 个进测试集(覆盖全量程)")
    ap.add_argument("--skip-trial", action="store_true", help="只训练+拟合报告,不跑 LF 试算")
    ap.add_argument("--trial-only", action="store_true", help="读已存预测补 LF 试算")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    import csv
    lib = Path(a.root) / a.lib_name
    lf_by_sid, solvable = {}, {}
    with open(lib / "manifest.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lf_by_sid[r["sample_id"]] = float(r["load_factor"])
            solvable[r["sample_id"]] = r["solvable"] == "1"
    samples = [s for s in load_library(lib) if solvable.get(s.sample_id)]
    samples.sort(key=lambda s: lf_by_sid[s.sample_id])
    test_idx = set(range(0, len(samples), a.test_every))
    train_s = [s for i, s in enumerate(samples) if i not in test_idx]
    test_s = [s for i, s in enumerate(samples) if i in test_idx]
    print(f"[g1] lib118 可解 {len(samples)}:train {len(train_s)} / test {len(test_s)}"
          f"(load_factor 每 {a.test_every} 取 1,覆盖全量程)")

    stat, ei, br, mask, y_all, lf_all = build_arrays(samples, lf_by_sid)
    tr = np.array([i for i in range(len(samples)) if i not in test_idx])
    te = np.array(sorted(test_idx))
    y_mu = y_all[tr].reshape(-1, len(TARGETS)).mean(0)
    y_sd = y_all[tr].reshape(-1, len(TARGETS)).std(0) + 1e-6

    dev = torch.device("cpu")
    x_stat = torch.from_numpy(stat)
    e_attr = torch.from_numpy(br)
    eit = torch.from_numpy(ei.astype(np.int64))
    m_t = torch.from_numpy(mask)
    y_t = torch.from_numpy((y_all - y_mu) / y_sd)
    lf_t = torch.from_numpy(lf_all)
    net = G1Net(node_in=stat.shape[1] + 1, edge_in=br.shape[1], d=a.d, layers=a.layers)
    print(f"[g1] 参数 {net.num_params()/1e3:.0f}k;确定性回归,无编码器/解码器(6.7)")

    if not a.trial_only:
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        rng = np.random.default_rng(a.seed)
        for step in range(a.steps):
            i = int(rng.choice(tr))
            x = torch.cat([x_stat, lf_t[i].expand(x_stat.shape[0], 1)], dim=1)
            pred = net(x, e_attr, eit)
            loss = ((pred - y_t[i])[m_t] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % max(1, a.steps // 10) == 0:
                print(f"[g1] step {step+1}/{a.steps} mse={float(loss):.5f}", flush=True)
        torch.save({"state": net.state_dict(), "y_mu": y_mu, "y_sd": y_sd,
                    "config": vars(a)}, out / "g1_net.pt")
    else:
        ck = torch.load(out / "g1_net.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state"])
        y_mu, y_sd = ck["y_mu"], ck["y_sd"]

    # ---- 拟合口径:test 标幺 MAE(仅监督位置)----
    net.eval()
    preds = {}
    mae = {f: [] for f in TARGETS}
    with torch.no_grad():
        for i in te:
            x = torch.cat([x_stat, lf_t[i].expand(x_stat.shape[0], 1)], dim=1)
            p = (net(x, e_attr, eit).numpy() * y_sd + y_mu).astype(np.float32)
            preds[samples[i].sample_id] = p
            for k, f in enumerate(TARGETS):
                m = mask[:, k]
                mae[f].append(np.abs(p[m, k] - y_all[i][m, k]))
    mae = {f: float(np.concatenate(v).mean()) for f, v in mae.items()}
    print("[g1] test 标幺 MAE: " + " ".join(f"{f}={mae[f]:.4f}" for f in TARGETS))
    np.savez(out / "g1_preds_test.npz",
             **{sid: p for sid, p in preds.items()})

    # ---- 平均化指纹(多模态库的判别性诊断):回归调度落在模式之间还是模式之上 ----
    # 逐 test 例:预测逐机出力向量到"同档任一真实方式"的最近 L1 距离(nearest);
    # 对照 = 同档真实方式两两 L1 的中位(inter,模式间典型距离)。
    # nearest ≈ 0 → 回归命中某个真实模式;nearest ~ inter/2 → 落在模式之间(平均化)。
    gen_m = mask[:, 2]
    fp_nearest, fp_inter = [], []
    by_lf = {}
    for i in range(len(samples)):
        by_lf.setdefault(float(lf_all[i]), []).append(y_all[i][gen_m, 2])
    for i in te:
        lv = float(lf_all[i])
        peers = [v for j, v in enumerate(by_lf[lv])]
        pred_pg = preds[samples[i].sample_id][gen_m, 2]
        fp_nearest.append(min(float(np.abs(pred_pg - p).sum()) for p in peers))
    for lv, group in by_lf.items():
        for i2 in range(0, len(group) - 1, 2):
            fp_inter.append(float(np.abs(group[i2] - group[i2 + 1]).sum()))
    fingerprint = {
        "pred_to_nearest_true_mode_L1": round(float(np.mean(fp_nearest)), 4),
        "inter_mode_L1_median": round(float(np.median(fp_inter)), 4) if fp_inter else None,
        "note": "nearest≈0=命中某真实模式;nearest 与 inter 同量级=落在模式之间(平均化指纹)",
    }
    print(f"[g1] 平均化指纹:pred→最近真实模式 L1={fingerprint['pred_to_nearest_true_mode_L1']}"
          f" / 模式间典型 L1={fingerprint['inter_mode_L1_median']}")

    rep = {"lib": a.lib_name, "n_train": len(train_s), "n_test": len(test_s),
           "params": net.num_params(),
           "steps": a.steps, "mae_pu_test": mae, "fingerprint": fingerprint,
           "label": "load_factor(lib118=连续因子;lib118_diverse=离散档,同档多方式)"}

    # ---- 平均化失败演示:写回 + LF 试算(窗口纪律)----
    if a.skip_trial:
        rep["trial"] = "skipped(--skip-trial;稍后 --trial-only 补)"
    else:
        stage = _chain_stage()
        if not _trial_window_ok(stage):
            print(f"[g1] 包 A 处于串行复核窗({stage}),按纪律不跑 LF 试算;"
                  f"稍后 --trial-only 补")
            rep["trial"] = f"deferred(chain={stage})"
        else:
            from gridfm.solver.psasp import PsaspSolver
            solver = PsaspSolver()
            base = Path(a.base)
            rows = []
            t0 = time.time()
            for i in te:
                sid = samples[i].sample_id
                work = out / "work" / f"pred_{sid}"
                conv, vmin, vmax = writeback_and_trial(preds[sid], base, work, solver)
                rows.append({"sid": sid, "kind": "pred", "lf": float(lf_all[i]),
                             "converged": conv, "vmin": vmin, "vmax": vmax})
                robust_rmtree(work)
                # 对照:同标签真实方式(留存算例重解,应 100% 收敛)
                case_dir = lib / "cases" / sid
                if case_dir.is_dir():
                    work2 = out / "work" / f"real_{sid}"
                    robust_copytree(case_dir, work2)
                    pf = solver.run_powerflow(work2)
                    rows.append({"sid": sid, "kind": "real", "lf": float(lf_all[i]),
                                 "converged": bool(pf.converged)})
                    robust_rmtree(work2)
            pred_rows = [r for r in rows if r["kind"] == "pred"]
            real_rows = [r for r in rows if r["kind"] == "real"]
            conv_pred = sum(r["converged"] for r in pred_rows)
            conv_real = sum(r["converged"] for r in real_rows)
            vband = sum(1 for r in pred_rows
                        if r["converged"] and r["vmin"] is not None and r["vmin"] >= 0.85)
            rep["trial"] = {
                "n": len(pred_rows),
                "pred_converged": conv_pred,
                "pred_conv_rate": round(conv_pred / max(len(pred_rows), 1), 4),
                "pred_conv_and_vband085": vband,
                "real_converged": conv_real,
                "real_conv_rate": round(conv_real / max(len(real_rows), 1), 4),
                "elapsed_s": round(time.time() - t0, 1),
                "note": "vband085 = 收敛且 vmin≥0.85(与方式库电压带验收同门槛)",
            }
            print(f"[g1] 试算:回归方式收敛 {conv_pred}/{len(pred_rows)}"
                  f"(电压带内 {vband});真实方式 {conv_real}/{len(real_rows)}")
            nearest = fingerprint["pred_to_nearest_true_mode_L1"]
            inter = fingerprint["inter_mode_L1_median"] or 0.0
            fp_positive = inter > 0 and nearest > 0.25 * inter
            if conv_pred < conv_real and conv_pred / max(len(pred_rows), 1) < 0.9:
                rep["conclusion"] = ("平均化失败复现:回归方式裸收敛率显著低于真实方式"
                                     "(6.6 第一档成立,'不收敛'口径)")
            elif fp_positive:
                rep["conclusion"] = (
                    "平均化指纹阳性但收敛未失败:回归输出到最近真实模式的距离"
                    f"({nearest:.1f})与模式间典型距离({inter:.1f})同量级——回归给出"
                    "'哪个模式都不是'的平均调度,一对多平均化确凿;但该平均调度在本网架上"
                    "物理可解(平衡机兜底/电压支撑充足),'必不收敛'分句不成立。"
                    "G1-扩散立项依据建议改挂'模式保真/语义'口径(6.6 文本修订需评审)")
            else:
                rep["conclusion"] = (
                    "平均化失败未复现且指纹阴性:本库扰动单模态,条件均值≈缩放基态方式;"
                    "失败演示需多模态方式分布(多调度方案/多开机组合)")
            print(f"[g1] 结论:{rep['conclusion']}")

    (out / "g1_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"[g1] 报告 {out / 'g1_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
