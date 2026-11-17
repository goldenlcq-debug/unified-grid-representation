"""探针 B v2(2026-08-22 第 3 轮,交接 §3.2 末段):E12 上训练前的离线可行性探针,25k / 70k 各 3 个方式,零训练、只读。

与 v1(probe_newton_big_v1_normaleq.py,法方程 AᵀA,万节点级条件数平方后发散)的区别,全部按预注册 §15.2 的修正步口径:
  * **直接解** B′·Δθ = ΔP / V(B′ = 在运支路 1/x 的拉普拉斯,平衡机 Dirichlet;scipy 直接稀疏 LU,不用法方程);
  * 右端 = 只在"干净 P 方程"行上的失配 ΔP 除以 V(FDPF 约定),无效行置 0;
  * 覆盖率闸:干净 P 方程在自由节点上的覆盖率 < 99% → 该样本跳过(这里只打印覆盖率);
  * 发散闸:max|Δθ| > 1 rad → 本步跳过;
  * V 三版:神谕 V(上界)/ 神谕 + 0.002 pu 噪声(E10 量级)/ 零训练 Q–V 提示 v̂(模型起步时手里真有的);
  * 可选 Q–V 半步(--qv,预览 rf2qv 臂):解 B″·ΔV = ΔQ / V(B″ = −Im Y 的 PQ 子块),看 25k 的 v 能从提示 0.00167 压到哪。
期望(交接 §3.2):从带网损 DC 起点 1~2 步到 ~0.014(25k)/ ~0.03 rad(70k)量级;探针不过就别烧 GPU。
用法(grid-foundation/ 下):python scripts/probes_5060/probe_newton_big.py [--qv] [--cap 3] [--steps 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, '.')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
from gridfm.data.dataset import family_files  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.reason.pf_physics import dc_theta_lossy, pf_phys_arrays, qv_hint  # noqa: E402
from gridfm.reason.pf_theta_edge import pf_edge_susceptance, pf_theta_ref  # noqa: E402
from gridfm.train.masking import _pf_typed_mask  # noqa: E402

TH, VC = NODE_TABLE.offset_of("theta"), NODE_TABLE.offset_of("v")
noff = NODE_TABLE.offset_of
D = Path("../大规模算例语料交付-20260821/大规模算例语料交付-20260821/gridfm-data-large")


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def run(raw, tap, shift, vmode, steps=3, qv=False, rng=None, negb="zero"):
    n = raw.n_nodes
    arr = pf_phys_arrays(raw, tap=tap, shift=shift)
    idx, gy, by = arr["index"], arr["g"].astype(np.float64), arr["b"].astype(np.float64)
    eqp, eqq = arr["eqp_clean"], arr["eqq_clean"]
    _mf, _tg, lm = _pf_typed_mask(raw.node_feat, raw.node_missing)
    sup_t, sup_v = lm[:, TH], lm[:, VC]
    d = dc_theta_lossy(raw)
    _tr, ref, reach, _av = pf_theta_ref(raw, d[:, :2])
    th = d[:, 0].astype(np.float64).copy()
    tt = raw.node_feat[:, TH].astype(np.float64)
    tv = raw.node_feat[:, VC].astype(np.float64)
    types = np.argmax(raw.node_feat[:, noff("node_type"):noff("node_type") + 3], axis=1)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    pq = (types == 0) & ~null
    if vmode == "oracle":
        vv = tv.copy()
    elif vmode == "noise":
        vv = tv + rng.normal(0, 0.002, size=n)
        vv[~pq] = tv[~pq]                                   # 只有 PQ 的 v 是预测量
    elif vmode == "qvhint":
        vh = qv_hint(raw, tap=tap, shift=shift, dc=d)
        vv = tv.copy()
        vv[pq] = np.where(vh[pq, 1] > 0.5, vh[pq, 0], 1.0)
    else:
        raise ValueError(vmode)
    p_given = (raw.node_feat[:, noff("pg")] - raw.node_feat[:, noff("pl")]).astype(np.float64)
    q_given = (raw.node_feat[:, noff("qg")] - raw.node_feat[:, noff("ql")]).astype(np.float64)
    # B′:在运支路 1/x 的拉普拉斯(与 pf_edge_susceptance 同源,串补 x<0 置 0)
    from gridfm.reason.pf_theta_edge import use_edges
    from scipy.sparse.csgraph import connected_components
    from gridfm.schema.fields import BRANCH_TABLE
    xb = raw.branch_feat[:, BRANCH_TABLE.offset_of("x")].astype(np.float64)
    use = use_edges(raw)
    i, j = raw.edge_index[0].astype(np.int64), raw.edge_index[1].astype(np.int64)
    with np.errstate(divide="ignore"):
        bfull = np.where(use, 1.0 / np.where(np.abs(xb) > 1e-9, xb, 1.0), 0.0)
    if negb == "zero":          # 现行 pf_edge_susceptance:串补置 0(孤块按 Dirichlet 跳过)
        bb = np.where(bfull > 0, bfull, 0.0)
    elif negb == "sign":        # FDPF 标准:带符号 1/x
        bb = bfull
    elif negb == "abs":
        bb = np.abs(bfull)
    else:
        raise ValueError(negb)
    k = bb != 0
    i, j, bb = i[k], j[k], bb[k]
    L = sp.coo_matrix((np.r_[bb, bb, -bb, -bb], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()
    # 自由节点 = 经 b≠0 支路与某参考角连通的非参考节点(孤块 → Dirichlet,Δθ=0)
    A = sp.coo_matrix((np.ones(i.size * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    _nc, comp = connected_components(A, directed=False)
    has_ref = np.bincount(comp[ref], minlength=_nc) > 0
    free = has_ref[comp] & reach & ~ref
    fidx = np.where(free)[0]
    n_orphan = int((reach & ~ref & ~free).sum())
    cov = float((eqp & free).sum() / max(free.sum(), 1))
    t0 = time.time()
    lu = splu(L[fidx][:, fidx].tocsc())
    t_lu = time.time() - t0
    lu_q, qidx, cov_q = None, None, float("nan")
    if qv:
        qidx = np.where(eqq)[0]
        cov_q = float(eqq.sum() / max(pq.sum(), 1))
        Bpp = sp.coo_matrix((-by, (idx[0], idx[1])), shape=(n, n)).tocsr()
        t0 = time.time()
        lu_q = splu(Bpp[qidx][:, qidx].tocsc() + 1e-8 * sp.identity(qidx.size, format="csc"))
        t_lu += time.time() - t0

    def mismatch(th_, vv_):
        vre, vim = vv_ * np.cos(th_), vv_ * np.sin(th_)
        ire = np.zeros(n)
        iim = np.zeros(n)
        np.add.at(ire, idx[0], gy * vre[idx[1]] - by * vim[idx[1]])
        np.add.at(iim, idx[0], gy * vim[idx[1]] + by * vre[idx[1]])
        return p_given - (vre * ire + vim * iim), q_given - (vim * ire - vre * iim)

    mt = sup_t & reach
    mv = sup_v & reach

    def e_th(th_):
        return float(np.abs(th_[mt] - tt[mt]).mean()), float(np.abs(wrap(th_[mt] - tt[mt])).mean())

    def e_v(v_):
        return float(np.abs(v_[mv] - tv[mv]).mean())

    r0, w0 = e_th(th)
    hist = [dict(step=0, th_raw=r0, th_wrap=w0, v=e_v(vv),
                 dp=float(np.abs(mismatch(th, vv)[0][eqp & free]).mean()))]
    skipped = 0
    for s in range(1, steps + 1):
        dp, _ = mismatch(th, vv)
        rhs = np.where(eqp, dp / np.maximum(vv, 0.5), 0.0)
        dth = np.zeros(n)
        dth[fidx] = lu.solve(rhs[fidx])
        mx = float(np.abs(dth).max())
        if not np.isfinite(mx) or mx > 1.0:
            skipped += 1
            rec = dict(step=s, skipped=True, max_dth=mx)
        else:
            cands = {a: th + a * dth for a in (0.5, 1.0)}
            errs = {a: e_th(c)[1] for a, c in cands.items()}
            best_a = min(errs, key=errs.get)
            th = cands[1.0]                              # 主线按 α=1(FDPF 标准步)推进
            ra, wa = e_th(th)
            rec = dict(step=s, max_dth=mx, th_raw_a1=ra, th_wrap_a1=wa, best_alpha=best_a, th_wrap_best=errs[best_a])
        if qv and lu_q is not None:
            _, dq = mismatch(th, vv)
            dv = lu_q.solve(dq[qidx] / np.maximum(vv[qidx], 0.5))
            mxv = float(np.abs(dv).max())
            if np.isfinite(mxv) and mxv <= 0.5:
                vv = vv.copy()
                vv[qidx] = np.clip(vv[qidx] + dv, 0.5, 1.5)
                rec["v_after_qv"] = e_v(vv)
            else:
                rec["qv_skipped"] = True
                rec["max_dv"] = mxv
        rec["dp_after"] = float(np.abs(mismatch(th, vv)[0][eqp & free]).mean())
        hist.append(rec)
    return dict(n=n, cov_p=cov, cov_q=cov_q, n_free=int(free.sum()), n_orphan=n_orphan, t_lu=t_lu, skipped=skipped, hist=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--fams", default="mp_activsg25k,mp_activsg70k")
    ap.add_argument("--vmodes", default="oracle,noise,qvhint")
    ap.add_argument("--qv", action="store_true", help="每步后附加一次 Q–V 半步(预览 rf2qv 的电压)")
    ap.add_argument("--negb", default="zero,sign,abs", help="串补 x<0 在 B′ 里的口径:zero(置 0,孤块 Dirichlet)/sign(带符号)/abs")
    ap.add_argument("--out", default="runs/diag_5060/probe_newton_big_v2.json")
    a = ap.parse_args()
    allrec = {}
    for fam in a.fams.split(","):
        z = np.load(D / "families" / "transfer" / fam / "family_extra.npz")
        tap, shift = z["branch_tap"].astype(np.float64), z["branch_shift_rad"].astype(np.float64)
        ps = [p for p in family_files(D, "transfer") if Path(p).parent.name == fam][:a.cap]
        raws = [GridSample.load_npz(p) for p in ps]
        for negb in a.negb.split(","):
          for vmode in a.vmodes.split(","):
            rng = np.random.default_rng(1)
            rs = [run(r, tap, shift, vmode, steps=a.steps, qv=a.qv, rng=rng, negb=negb) for r in raws]
            allrec[f"{fam}/{negb}/{vmode}"] = rs
            cov = np.mean([r["cov_p"] for r in rs])
            tlu = np.mean([r["t_lu"] for r in rs])
            h0 = [r["hist"][0] for r in rs]
            line = [f"{fam:<14}{negb:<5}{vmode:<7} 孤块 {np.mean([r['n_orphan'] for r in rs]):.0f} 覆盖率P {cov:.4f}"
                    + (f" Q {np.mean([r['cov_q'] for r in rs]):.4f}" if a.qv else "")
                    + f" LU {tlu:.1f}s | 起点 θ raw/wrap {np.mean([h['th_raw'] for h in h0]):.4f}/"
                      f"{np.mean([h['th_wrap'] for h in h0]):.4f} v {np.mean([h['v'] for h in h0]):.5f}"]
            for s in range(1, a.steps + 1):
                hs = [r["hist"][s] for r in rs]
                ok = [h for h in hs if not h.get("skipped")]
                if not ok:
                    line.append(f" | {s}步 全部跳过(max|Δθ| {np.mean([h['max_dth'] for h in hs]):.2f})")
                    continue
                wa = np.mean([h["th_wrap_a1"] for h in ok])
                seg = (f" | {s}步 α=1 raw/wrap {np.mean([h['th_raw_a1'] for h in ok]):.4f}/{wa:.4f}"
                       f"({np.degrees(wa):.2f}°) 最好α {np.mean([h['th_wrap_best'] for h in ok]):.4f}"
                       f" max|Δθ| {np.mean([h['max_dth'] for h in ok]):.2f}")
                if len(ok) < len(hs):
                    seg += f" 跳过 {len(hs) - len(ok)}/{len(hs)}"
                if a.qv:
                    vq = [h.get("v_after_qv") for h in hs if h.get("v_after_qv") is not None]
                    seg += f" v→{np.mean(vq):.5f}" if vq else " v:QV跳过"
                line.append(seg)
            print("".join(line), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(allrec, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[probe] 写 {a.out}")


if __name__ == "__main__":
    main()
