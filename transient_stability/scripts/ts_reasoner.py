"""判稳推理器 v0 训练(交接 15 版主线;方案 5.2 稳定判别/8.6;D8/D11/D13)。

编码器冻结(v1 或随机 = 5.3 组1/组2),只训推理器。输入 = 已解方式标准档案
(判稳无"未知量置缺失"——解后状态是故障前初值)+ 事件 token(故障支路 token +
[K/100,Ts,Te,dur],ts_task 模块 docstring);**泄漏红线 = 暂态产物不得入输入**
(单测锁定)。四头 = 判稳二值/类型三 multi-label/CCT(删失铰链)/机群 multi-hot。

划分(D14):cepri36 按方式留出 [内](**同拓扑豁免,报告如实标注**);ieee39 整族
[外],永不进训练。评估按 8.6 双报,失稳召回主指标;v0 = 管线贯通 + 冒烟,
**不设死门禁**(门禁校准待 v2 粮仓,8.9:阈值只准依据 val/冒烟定)。

  E:/anaconda3/envs/grid_gpu/python.exe scripts/ts_reasoner.py \
      --root E:/AI-2/统一计算大模型/gridfm-data --device cuda [--encoder v1|random]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from gridfm.data.dataset import load_library  # noqa: E402
from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.psasp_io.stana import CRITERIA_VERSION  # noqa: E402
from gridfm.psasp_io.modegate import ts_mode_ok  # noqa: E402
from gridfm.reason.ts_task import (  # noqa: E402
    CCT_T_MAX, EVENT_PARAM_DIM, TS_SUBTYPES, TS_TYPE_FROM_SUB, TSReasoner,
    average_precision, build_ts_example, fault_edge_pointer, gen_node_mask,
    join_key, ts_loss,
)
from gridfm.reason.ts_encoder_io import (  # noqa: E402
    UNFREEZE_CHOICES, apply_unfreeze, encoder_drift, encoder_fingerprint,
    train_mode_sensitive,
)
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

# 训练/评估库与角色(D14:ieee39 整族 [外],永不入训)。
# 每族可挂多个标签批(v1 电池 + v2/v3 扩产电池)——**只纳已收口批**,收口门见 _closed_out()。
# **2026-07-16 拓扑维度 1→2**:ieee118 成为**第二个训练家族**(收轮裁决 1)。此前训练
# 粮仓在结构上只有 cepri36 一个拓扑 → 模型从未见过拓扑变化,[外] 判别力 1.010±0.015
# (恒定预测,零跨族能力),且加数据无用(1,800→7,800 例纹丝不动)。
LIBS = {"cepri36": ("lib_cepri36", ["ts_labels_cepri36", "ts_labels_cepri36_v2",
                                    "ts_labels_cepri36_k50"]),
        "ieee118": ("lib118", ["ts_labels_ieee118_v1", "ts_labels_ieee118_v2"]),
        "ieee39": ("lib_ieee39", ["ts_labels_ieee39", "ts_labels_ieee39_v2"])}
# [外] 家族(D14 铁律:整族只准评估,永不入训)。其余 LIBS 家族 = 训练家族。
OUT_FAMILIES = ("ieee39",)
# **跨库批**(2026-08-01 OOD 定向修补):标签目录在本仓 runs/、样本库非族默认库的训练批。
# 为什么需要:diverse 方式(ieee118_3xxxx)样本在 lib118_diverse,而 LIBS 每族只挂一个库;
# 且新批产在本仓 runs/ 而历史批在归档 labels_root —— 两者都用覆盖项解决,不动 LIBS 结构。
# 批目录不存在时整条 no-op(历史行为逐位不变)。sid 域(3xxxx vs 0xxxx)天然不撞。
_REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_OVERRIDES = {
    "ts_labels_ieee118div_train": {
        "family": "ieee118",
        "lib": "lib118_diverse",
        "labels_dir": _REPO_ROOT / "runs" / "ts_labels_ieee118div_train",
        "holdout_exempt": True,   # 新批方式全进训练侧,[内] 留出与 dose_b 逐位不变
    },
}
# 定向补测批(BATTERY_V4;2026-07-17 B2)——**只评估、永不入训的代码级锁**(与
# resolve_out_families 同理):默认装载(only_batches=None)永不包含;训练入口显式点名
# 一律拒绝(main 里 assert);唯一合法消费者 = ts_eval_batch 等评估器显式 only_batches 点名。
# 动因:这些批只打 v5 划分的留出方式,入训 = 留出方式泄漏进训练集,[内] 评估全部作废。
# 共享冻结基座指纹(blake2b(file,16),非 MD5;与 ts_screen.py:60-61 同源同值)
FP_V1 = "ccf4260f2a8f5db5b6650a33b17d7a0f"
FP_NM = "12281dd483206efe09fdf19475472ebc"


def _blake16(p) -> str:
    return hashlib.blake2b(Path(p).read_bytes(), digest_size=16).hexdigest()


EVAL_ONLY_BATCHES = {
    "ts_labels_cepri36_sub25": "cepri36", "ts_labels_cepri36_sub75": "cepri36",
    "ts_labels_cepri36_vover0": "cepri36", "ts_labels_cepri36_vover50": "cepri36",
}


def assert_no_eval_only_batches(batch_names) -> None:
    """训练入口锁:--batches 点名定向补测批一律响亮拒绝。"""
    bad = sorted(set(batch_names or []) & set(EVAL_ONLY_BATCHES))
    if bad:
        raise SystemExit(f"[ts] 定向补测批只准评估、永不入训(代码级锁):{bad}")


def _batch_allowed(lab_name: str, only_batches) -> bool:
    """装载准入:评估专用批必须被显式点名;普通批按白名单(None=全收)。"""
    if lab_name in EVAL_ONLY_BATCHES:
        return bool(only_batches) and lab_name in only_batches
    return not only_batches or lab_name in only_batches


def resolve_out_families(names) -> tuple:
    """--out-families 校验(2026-07-17 双外族实验):去重保序;**ieee39 必须在列**
    (D14 铁律"ieee39 整族永不入训"——任何配置都不许把它挪回训练);家族名须在 LIBS。
    默认 = OUT_FAMILIES,行为与单外族时代逐位一致。"""
    out = tuple(dict.fromkeys(names))
    if "ieee39" not in out:
        raise SystemExit("[ts] ✋ --out-families 必须包含 ieee39(D14 铁律:整族永不入训)")
    unknown = [n for n in out if n not in LIBS]
    if unknown:
        raise SystemExit(f"[ts] ✋ 未知家族 {unknown}(合法:{sorted(LIBS)})")
    if len(out) >= len(LIBS):
        raise SystemExit("[ts] ✋ 外族不能覆盖全部家族(至少留一个训练家族)")
    return out


def family_seed(fam: str) -> int:
    """家族名 → **稳定**种子分量(逐族独立留出用)。

    **不得用内置 hash()**:Python 对 str 的 hash 逐进程随机化(PYTHONHASHSEED)→ 同
    --seed 在不同进程会切出**不同的留出集** → 续训检查点校验必崩,更糟的是重跑时
    上一轮的留出方式会漂进 train(静默泄漏)。
    """
    return int.from_bytes(hashlib.sha256(fam.encode("utf-8")).digest()[:4], "big")


def mode_holdout_by_family(examples, train_fams, holdout_frac: float, split_seed: int):
    """逐族按方式留出 → {family: [sample_id]}(D14;2026-07-25 抽出为可测函数)。

    **每族独立 rng([split_seed, family_seed(fam)])**:否则两族的留出选择被同一序列耦合。
    split_seed 与训练种子解耦(--holdout-seed):同一 split_seed 下多种子共享留出集,
    这是做无泄漏 ensemble 的前提(此前各种子留出互异,跨种子平均没有干净验证口径)。
    """
    held_by_fam = {}
    for fam in train_fams:
        # holdout_exempt(2026-08-01 跨库批):豁免例不入留出候选池 → 整批落训练侧,
        # 且非豁免池与豁免批加入前**逐位相同** → held_ids 不因新批而漂移(Q2 口径基石)
        ids = sorted({x["sample_id"] for x in examples
                      if x["family"] == fam and not x.get("holdout_exempt")})
        if not ids:
            continue
        rng_f = np.random.default_rng([split_seed, family_seed(fam)])
        n_held = max(1, int(len(ids) * holdout_frac))
        held_by_fam[fam] = sorted(rng_f.permutation(ids)[:n_held].tolist())
    return held_by_fam


def _read_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _closed_out(lab: Path):
    """收口门(2026-07-15;**同日对抗审查后重写**):在产/半截/未重判批一律拒纳。

    **重写动因(审查 2 条 critical)**:
    - 旧门**从不读 labels.jsonl**,判据是"summary.n == len(labels_v1)"——两个都是
      历史快照,正常流水线里天然相等。结果**时序反转**:批正在产(labels.jsonl 在长、
      summary 未重写)门反而**开**,放行过时的 labels_v1;批产完未重判门才关。
      它挡住 ieee39_v2 纯属侥幸(该批从未 relabel,被"无 labels_v1"拦下)。
    - 旧门只看 audit 的 n_failed,而 ts_audit 对**缺归档行**记 skip 不记 fail:
      "体检 0 行、跳过 1500 行"同样是 n_failed=0 全绿 → 门以为体检过了,其实一行没查。

    **新判据(全满才收;一律以 labels.jsonl 的实际内容为锚)**:
    ①labels_v1.jsonl 在位;
    ②**tag 集合逐一对齐**:labels_v1 的 tag 集 == labels.jsonl 的 tag 集(内容绑定,
      不是行数——requeue 原地截断后行数会复原,行数比不出陈旧);
    ③summary.n == labels.jsonl 行数(生产确已正常收尾;进程被杀/熔断时 summary 不写);
    ④audit:n_failed==0 **且 n_audited == n_feasible_rows 且 n_skipped_missing_archive==0**
      (真正体检过每一行,不是"全跳过也算绿");
    ⑤**判据版本**:labels_v1 每行 criteria_version 必须 == CRITERIA_VERSION(判据行为
      或 schema 变更后,旧批必须重判才准入训——防"v0 冒充 v1"与"缺字段静默退化")。
    coherency.jsonl 缺席不拦(仅机群头无标签),但显式报出。
    返回 (ok, 原因, rows, coh)。
    """
    if not (lab / "labels_v1.jsonl").exists():
        return False, "无 labels_v1.jsonl(未 relabel 定稿)", [], {}
    rows = _read_jsonl(lab / "labels_v1.jsonl")
    raw = _read_jsonl(lab / "labels.jsonl")
    if not raw:
        return False, "无 labels.jsonl(生产产物缺失)", [], {}
    t_v1 = {r.get("tag") for r in rows}
    t_raw = {r.get("tag") for r in raw}
    if t_v1 != t_raw:
        return False, (f"labels_v1 与 labels.jsonl 的 tag 集不一致"
                       f"(v1 {len(t_v1)} / 生产 {len(t_raw)};缺判 {len(t_raw - t_v1)}、"
                       f"多出 {len(t_v1 - t_raw)})→ 批在产或产后未重判"), [], {}
    sm = lab / "labels_summary.json"
    if not sm.exists():
        return False, "无 labels_summary.json(生产未正常收尾)", [], {}
    n = json.loads(sm.read_text(encoding="utf-8")).get("n")
    if n != len(raw):
        return False, f"summary.n={n} ≠ labels.jsonl 行数 {len(raw)}(生产未收尾/在产)", [], {}
    au = lab / "audit_report.json"
    if not au.exists():
        return False, "无 audit_report.json(未体检)", [], {}
    a = json.loads(au.read_text(encoding="utf-8"))
    nf, n_aud = a.get("n_failed", -1), a.get("n_audited", -1)
    n_feas, n_skip = a.get("n_feasible_rows", -1), a.get("n_skipped_missing_archive", -1)
    if nf != 0:
        return False, f"体检 n_failed={nf}(未全绿)", [], {}
    if n_skip != 0:
        return False, f"体检跳过 {n_skip} 行(缺归档)——skip 不计入 n_failed,不得当全绿", [], {}
    if n_aud != n_feas or n_aud <= 0:
        return False, (f"体检覆盖不全:n_audited={n_aud} / n_feasible_rows={n_feas}"
                       f"(0 行体检也是 n_failed=0)"), [], {}
    # **t0 覆盖门**(2026-07-14 对抗审查 major):ts_audit 的 --lib 是可选参数,缺了它
    # 每行只记 t0_skipped='no_lp1'、**不记 fail 也不记 skip** → n_failed=0、
    # n_audited==n_feasible_rows、n_skipped=0 全绿,而 t0_checked=0。t0 是**唯一**拿
    # 轨迹去对独立事实源(LF.LP1 潮流解)的检查(其余锚点都是 STOUT↔STANA 同源互对,
    # 同源错误对它们不可见)。丢了它 = 又一个"0 行体检也算全绿"。
    # 老四批实测 t0_checked == n_audited(1500/6000/300/10200),加此门不误伤。
    n_t0 = a.get("t0_checked", 0)
    n_no_t2 = a.get("no_type2_curves", 0)
    if n_t0 + n_no_t2 < n_aud:
        return False, (f"t0 初值校验覆盖不全:t0_checked={n_t0}(+无 Type2 曲线 {n_no_t2})"
                       f" < n_audited={n_aud} —— ts_audit 缺 --lib?轨迹未对过独立事实源"
                       f"(LF.LP1),不得当全绿"), [], {}
    # 版本门只管**入训行**(feasible 且非 timeout);infeasible 行走 relabel 直通、
    # 标 v0_passthrough,下游本就丢弃(build_ts_example),不参与判据版本核对。
    live = [r for r in rows if r.get("feasible") and not r.get("timeout")]
    bad_ver = {r.get("criteria_version") for r in live} - {CRITERIA_VERSION}
    if bad_ver:
        return False, (f"入训行判据版本不符:{sorted(bad_ver)} ≠ {CRITERIA_VERSION}"
                       f"(判据/schema 已变更或存在缺归档直通行,须 ts_relabel 重判"
                       f"/补档后方可入训)"), [], {}
    coh = {r["tag"]: r for r in _read_jsonl(lab / "coherency.jsonl")}
    return True, "ok", rows, coh


def _load_superseded(lab: Path) -> set:
    """读批目录内 `superseded_sids.json` → 作废方式的 sample_id 集合;不存在 = 空集。

    档2(W1.3):`ts_labels_ieee118_v1` 里被 v2 重产的方式(运行点已抬压变更),其 v1 标签
    作废。批内历史文件一字不动(留档纪律),仅在装载时凭此清单剔除对应行并计数。
    文件格式:sample_id 列表 `["ieee118_00004", ...]`,或 `{"sample_ids": [...]}`。
    无该文件的批 → 空集,行为与补丁前逐位一致(向后兼容)。
    """
    f = lab / "superseded_sids.json"
    if not f.is_file():
        return set()
    data = json.loads(f.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("sample_ids", data.get("superseded_sids", []))
    return set(data)


def _cct_quality_ok(row) -> bool:
    """CCT 双验质量门(B3):verify.ok=False 拒绝(Te=cct 稳/Te=cct+tol 失稳 双验不成立,
    多为真 CCT 低于二分分辨率的过估值);无 verify 字段(旧批/--verify 省略/删失行)放行。"""
    return row.get("verify", {}).get("ok") is not False


def load_examples(root: Path, labels_root: Path, cct_dirs, only_batches=None):
    """标签三源 join → 训练例 + 原始/归一化样本表。返回 (examples, raw_by_sid, coverage)。

    only_batches:白名单(标签目录名);None = 取 LIBS 全部已收口批。
    """
    # **CCT join 必须带 battery**(2026-07-15 对抗审查 critical):tag 不含 K%,
    # ieee39 v1(K=20)与 ieee39_v2(K=0)的同线同方式同时长 tag 逐字相同(实测 300 撞),
    # 裸 tag 全局字典会把一个电池的 CCT 真值挂到另一个电池的例子上。
    cct_by_key = {}
    cct_verify_failed = 0
    for cd in cct_dirs:
        for r in _read_jsonl(Path(cd) / "cct.jsonl"):
            # 双验质量门见 _cct_quality_ok(118 电池实测 ~38% FAIL)
            if not _cct_quality_ok(r):
                cct_verify_failed += 1
                continue
            cct_by_key[join_key(r)] = r
    if cct_verify_failed:
        print(f"[ts] CCT 双验质量门:剔除 {cct_verify_failed} 行 verify.ok=False"
              f"(保留 {len(cct_by_key)} 行)", flush=True)
    examples, raw_by_sid = [], {}
    cov = {"rows": 0, "skipped_infeasible": 0, "skipped_no_sample": 0,
           "skipped_low_base_vmin": 0, "skipped_superseded": 0,
           "cct_joined": 0, "cct_verify_failed": cct_verify_failed,
           "group_valid": 0, "breakup": 0,
           "batches": [], "batches_rejected": [], "batches_no_coherency": []}
    mode_ok: dict = {}
    for fam, (lib_name, lab_names) in LIBS.items():
        sids_by_lib = {}
        catalogs = {}
        fam_eval_only = [b for b, f in EVAL_ONLY_BATCHES.items() if f == fam]
        fam_override = [b for b, o in BATCH_OVERRIDES.items() if o["family"] == fam]
        for lab_name in list(lab_names) + fam_eval_only + fam_override:
            if not _batch_allowed(lab_name, only_batches):
                continue
            ovr = BATCH_OVERRIDES.get(lab_name)
            lab = (ovr["labels_dir"] if ovr else labels_root / lab_name)
            eff_lib = root / (ovr["lib"] if ovr else lib_name)
            if not lab.is_dir():
                continue
            ok, why, rows, coh = _closed_out(lab)
            if not ok:
                cov["batches_rejected"].append({"batch": lab_name, "why": why})
                print(f"[ts] ✋ 拒纳 {lab_name}:{why}", flush=True)
                continue
            if not coh:
                cov["batches_no_coherency"].append(lab_name)
                print(f"[ts] ⚠ {lab_name} 无 coherency.jsonl → 该批机群头无标签"
                      f"(不拦,如实计数)", flush=True)
            cov["batches"].append({"batch": lab_name, "family": fam, "rows": len(rows)})
            superseded = _load_superseded(lab)   # 档2:被 v2 重产而作废的方式(空集=不影响)
            # 同库多批共用方式库,按库只装一次(跨库批用 eff_lib;sid 域不撞故下游
            # mode_ok/catalogs 仍按 sid 键缓存)
            lib_key = str(eff_lib)
            if lib_key not in sids_by_lib:
                sids_by_lib[lib_key] = {s.sample_id: s for s in load_library(eff_lib)}
            sids = sids_by_lib[lib_key]
            for r in rows:
                cov["rows"] += 1
                sid = r["sample_id"]
                if sid in superseded:            # v1 运行点已抬压变更 → 剔除其旧标签行
                    cov["skipped_superseded"] += 1
                    continue
                if sid not in sids:
                    cov["skipped_no_sample"] += 1
                    continue
                # **暂稳方式门**(2026-07-16 用户裁决;modegate.py):基态 vmin <门槛的
                # 方式,判据②的稳态分句在故障前就已违反 → 该方式全部例子无条件判失稳,
                # **标签与故障无关**(cepri36 实测 750 例失稳率 100%,健康方式仅 47~50%)。
                # 模型会学到"低压方式 = 失稳"的捷径。门后所产的批天然不含这类行(ts_labels
                # 已挡),此处挡的是门前老批(cepri36 v1/v2)。
                if sid not in mode_ok:
                    mode_ok[sid] = ts_mode_ok(eff_lib / "cases" / sid)
                if not mode_ok[sid]:
                    cov["skipped_low_base_vmin"] += 1
                    continue
                if sid not in catalogs:
                    catalogs[sid] = Catalog(PsaspCase(eff_lib / "cases" / sid))
                ex = build_ts_example(r, sids[sid], catalogs[sid], coh.get(r["tag"]),
                                      cct_by_key.get(join_key(r)))
                if ex is None:
                    cov["skipped_infeasible"] += 1
                    continue
                # 豁免标记必须在 None 检查之后(2026-08-01 审查 HIGH:infeasible 行返回
                # None,先打标记会 TypeError 崩掉整次装载)
                if ovr and ovr.get("holdout_exempt"):
                    ex["holdout_exempt"] = True
                ex["family"] = fam
                ex["batch"] = lab_name
                cov["cct_joined"] += int(ex["cct_valid"])
                cov["group_valid"] += int(ex["group_valid"])
                cov["breakup"] += int(ex["breakup"])
                raw_by_sid[sid] = sids[sid]
                examples.append(ex)
    return examples, raw_by_sid, cov


def event_only_null(train_exs, eval_exs) -> dict:
    """**event-only 空基线**(2026-07-14 对抗审查 critical:验收判据的真正零点)。

    **动因**:原验收门是"判别力 > 1.0"(= 打赢恒定预测)。但恒定预测**太弱**——
    一个**完全看不见电网**、只读事件参数 (K%, Ts, Te, dur) 的查表模型,靠"故障越长
    越容易失稳"这条**不需要任何拓扑理解**的规律,实测在 [外] ieee39 上就能拿到
    **判别力 1.289 / PR-AUC/base 1.370**(本会话实测,train = cepri36 门后 + 118 已产)。
    也就是说:门槛 1.05 会把"学会看切除时长"误判成"跨拓扑能力"。**跨拓扑能力的
    真正零点 = 这条空基线**,模型必须打赢它才算数。

    实现 = 在**训练集**上按 (k_pct, dur) 分桶统计失稳率(桶内 ≥10 例才用;否则按
    dur 最近邻回退,如实标注),在评估集上照查。输入**逐位取自 ex["event_params"]**
    (= 模型吃的同一份),不碰任何图/档案信息 —— 这正是"零拓扑信息"的操作化定义。
    """
    def key(ex):
        p = ex["event_params"]
        return (round(float(p[0]), 3), round(float(p[3]), 3))     # (K/100, dur)

    tab: dict = {}
    for ex in train_exs:
        a, b = tab.setdefault(key(ex), [0, 0])
        tab[key(ex)] = [a + 1, b + int(ex["y_unstable"] > 0.5)]
    prior = (sum(v[1] for v in tab.values()) / max(sum(v[0] for v in tab.values()), 1)
             if tab else 0.5)
    n_fallback = 0

    def score(ex):
        nonlocal n_fallback
        k = key(ex)
        if k in tab and tab[k][0] >= 10:
            return tab[k][1] / tab[k][0]
        n_fallback += 1                       # 训练集未见的 (K,dur) 档 → dur 最近邻
        if not tab:
            return prior
        kk = min(tab, key=lambda t: abs(t[1] - k[1]))
        return tab[kk][1] / tab[kk][0]

    if not eval_exs:
        return {}
    y = np.array([float(ex["y_unstable"]) for ex in eval_exs])
    s = np.array([score(ex) for ex in eval_exs])
    pred = s > 0.5
    tp = int((pred & (y > 0.5)).sum())
    fp = int((pred & (y < 0.5)).sum())
    fn = int((~pred & (y > 0.5)).sum())
    base = float((y > 0.5).mean())
    prec = tp / max(tp + fp, 1)
    ap = average_precision(y, s)
    return {
        "n": len(y), "base_rate_unstable": base,
        "unstable_recall": tp / max(tp + fn, 1),
        "unstable_precision": prec,
        "pred_pos_rate": float(pred.mean()),
        "skill_over_const": (prec / base) if base else None,
        "pr_auc": ap,
        "pr_auc_over_base": (ap / base) if base else None,
        "n_buckets": len(tab), "n_fallback_rows": n_fallback,
        "note": "只读 (K%,dur) 事件参数的查表模型;零图信息 = 跨拓扑能力的真正零点",
    }


def dyn_naked_rows(Xtr, ytr, evals, feature_names):
    """动态参数侧通道的**裸值基线行 + 逐特征零点行**(铁律:新特征配裸值基线并过零点池,
    逐族测定)。零图零档案:特征向量直接判别——模型增益必须超过这里的最好行,
    才算"理解"而非"读数"。

    - 裸值基线 = 标准化特征上的逻辑回归(确定性满批 GD,无 RNG),训练集拟合、评估集照查;
    - 零点行 = 单特征双向 AP 取大(s0_zeropool 同款口径),/base 报判别力。
    evals = {口径名: (X, y)}。
    """
    def _lr_fit(X, y, iters=500, lr=0.5, l2=1e-3):
        Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        w = np.zeros(Xb.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
            g = Xb.T @ (p - y) / len(y) + l2 * w
            w -= lr * g
        return w

    w = _lr_fit(Xtr, ytr)
    out = {}
    for name, (X, y) in evals.items():
        if len(y) == 0:
            continue
        base = float((y > 0.5).mean())
        s = np.concatenate([X, np.ones((len(X), 1))], axis=1) @ w
        ap_lr = average_precision(y, s)
        per = {}
        for k, fn in enumerate(feature_names):
            ap = max(average_precision(y, X[:, k]), average_precision(y, -X[:, k]))
            per[fn] = round(ap / base, 4) if base else None
        out[name] = {"n": int(len(y)), "base_rate_unstable": base,
                     "lr_ap_over_base": round(ap_lr / base, 4) if base else None,
                     "zero_per_feature_ap_over_base": per,
                     "zero_max": max(per.values()) if per else None}
    return out


def _ts_data(ex, normed, raw, cfg):
    """一例 → PyG Data。编码器路 = 标准档案(不掩码);标签只挂 y_*。"""
    n = normed.n_nodes
    data = sample_to_data(normed, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim,
                          branch_random_dim=cfg.branch_random_dim)
    data.fault_row = torch.tensor([ex["fault_branch_row"]], dtype=torch.long)
    ev = torch.from_numpy(ex["event_params"]).unsqueeze(0)
    if "event_ext" in ex:      # W5 侧通道(D13):标准化后的故障电气严重度特征拼进事件向量
        ev = torch.cat([ev, torch.from_numpy(ex["event_ext"]).unsqueeze(0)], dim=1)
    data.event_params = ev
    data.gen_mask = torch.from_numpy(gen_node_mask(raw))
    data.y_unstable = torch.tensor([ex["y_unstable"]], dtype=torch.float32)
    data.y_subtypes = torch.from_numpy(ex["y_subtypes"]).unsqueeze(0)
    data.y_cct = torch.tensor([0.0 if math.isnan(ex["y_cct"]) else ex["y_cct"]],
                              dtype=torch.float32)
    data.cct_valid = torch.tensor([ex["cct_valid"]])
    data.cct_censored = torch.tensor([ex["cct_censored"]])
    yg = ex["y_group"] if ex["y_group"] is not None else np.zeros(n, dtype=np.float32)
    data.y_group = torch.from_numpy(yg)
    data.group_valid = torch.tensor([ex["group_valid"]])
    return data


def _forward(enc, reasoner, batch, use_amp, amp_dtype, dev, weights=None, enc_grad=False):
    """enc_grad=False(默认)= 历史行为逐位不变:编码器前向在 no_grad 下。

    **R2 第 2 条**:解冻臂必须 enc_grad=True,否则梯度到不了编码器 —— requires_grad
    放开了、优化器也建了组,但 `torch.no_grad()` 会把整段前向从图里摘掉,编码器梯度
    恒为 None,"解冻"完全是假的,而 loss/指标照常出数(静默)。凭据 = 首步梯度范数。"""
    with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
        with torch.set_grad_enabled(enc_grad and torch.is_grad_enabled()):
            h, g, e = enc(batch)
        eb = batch.batch[batch.edge_index[0]]
        ptr = fault_edge_pointer(eb, batch.fault_row)
        s, t, c, gr = reasoner(h, g, e, batch.batch, eb, ptr, batch.event_params)
    loss, diag = ts_loss(s, t, c, gr, batch, weights)
    return (s, t, c, gr), loss, diag


@torch.no_grad()
def eval_ts(enc, reasoner, exs, normed_by_sid, raw_by_sid, cfg, dev, use_amp, amp_dtype,
            weights, bs=32):
    """自然分布口径评估(8.6;平衡集未冻结,如实标注)。返回指标 + 平均损失。

    **R2 补**:解冻臂训练期 enc 处于 train() —— 评估必须先切 eval() 再切回,否则
    train/eval 语义不同的子模块(dropout/BN)会污染早停锚。GridEncoder 实测无此类
    子模块(见 train_mode_sensitive),但语义正确性不依赖那个事实。"""
    enc_was_training = enc.training
    enc.eval()
    reasoner.eval()
    s_prob, y_un = [], []
    t_prob, t_true = [], []
    cct_err, cct_cen_ok = [], []
    g_tp = g_fp = g_fn = 0
    n_group_rows = 0
    losses = []
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        datas = [_ts_data(x, normed_by_sid[x["sample_id"]], raw_by_sid[x["sample_id"]], cfg)
                 for x in chunk]
        batch = Batch.from_data_list(datas).to(dev)
        (s, t, c, gr), loss, _ = _forward(enc, reasoner, batch, use_amp, amp_dtype, dev,
                                          weights)
        losses.append(float(loss))
        s_prob += torch.sigmoid(s.float()).cpu().tolist()
        y_un += batch.y_unstable.cpu().tolist()
        un = batch.y_unstable > 0.5
        if un.any():
            t_prob += torch.sigmoid(t.float()[un]).cpu().tolist()
            t_true += batch.y_subtypes[un].cpu().tolist()
        cv = batch.cct_valid
        if cv.any():
            non = cv & ~batch.cct_censored
            if non.any():
                cct_err += (c.float()[non] - batch.y_cct[non]).abs().cpu().tolist()
            cen = cv & batch.cct_censored
            if cen.any():
                cct_cen_ok += (c.float()[cen] >= CCT_T_MAX - 0.01).cpu().tolist()
        g_rows = batch.group_valid[batch.batch] & batch.gen_mask
        if g_rows.any():
            n_group_rows += int(batch.group_valid.sum())
            pred = torch.sigmoid(gr.float()[g_rows]) > 0.5
            true = batch.y_group[g_rows] > 0.5
            g_tp += int((pred & true).sum())
            g_fp += int((pred & ~true).sum())
            g_fn += int((~pred & true).sum())
    reasoner.train()
    if enc_was_training:
        enc.train()
    s_prob, y_un = np.array(s_prob), np.array(y_un)
    pred_un = s_prob > 0.5
    tp = int((pred_un & (y_un > 0.5)).sum())
    fn = int((~pred_un & (y_un > 0.5)).sum())
    fp = int((pred_un & (y_un < 0.5)).sum())
    m = {
        "n": len(exs), "n_unstable": int((y_un > 0.5).sum()),
        "unstable_recall": tp / max(tp + fn, 1),        # 8.6 主指标(假阴性代价最高)
        "unstable_precision": tp / max(tp + fp, 1),
        "accuracy": float((pred_un == (y_un > 0.5)).mean()),
        "pr_auc": average_precision(y_un, s_prob),
        "cct_mae_noncensored": float(np.mean(cct_err)) if cct_err else None,
        "cct_n_noncensored": len(cct_err),
        "cct_censored_hinge_ok": (float(np.mean(cct_cen_ok)) if cct_cen_ok else None),
        "group_micro_f1": (2 * g_tp / max(2 * g_tp + g_fp + g_fn, 1)) if n_group_rows else None,
        "group_rows": n_group_rows,
        "loss": float(np.mean(losses)),
    }
    # **基率与恒定预测基线**(2026-07-15 对抗审查 critical):失稳类基率极高时,
    # "全预测为正"这一零学习策略就能刷出 1.000 召回。若不并报基率与该基线,
    # 召回数字毫无判别力(实测:[外] ieee39 的 angle/voltage 基率 = 185/185 = 100%,
    # 此前据此宣称"跨族 angle 召回 1.000 = 能力"是错的,已收回)。
    m["base_rate_unstable"] = float((y_un > 0.5).mean()) if len(y_un) else None
    m["const_pos_recall"] = 1.0 if len(y_un) else None      # 恒定预测正类的召回(必为 1)
    m["const_pos_precision"] = m["base_rate_unstable"]      # 其精确率 = 基率
    # **预测正类率**:"恒定预测"的直接指纹(≈1.0 = 见谁都判失稳)。判别力 ≈1 时,
    # 单看它就能分清"模型没能力"与"评估集基率高";二值头此前只有类型/子类有这一项。
    m["pred_pos_rate"] = float(pred_un.mean()) if len(y_un) else None
    m["skill_over_const"] = (                                # >1 才说明模型优于恒定预测
        (m["unstable_precision"] / m["base_rate_unstable"])
        if m["base_rate_unstable"] else None)
    # 阈值无关口径(与 event-only 空基线对打用):恒定预测的 AP **恰等于基率**(AP 并列
    # 修复后成立,单测锁定)→ pr_auc_over_base 与 skill_over_const 同零点、可并排比。
    m["pr_auc_over_base"] = ((m["pr_auc"] / m["base_rate_unstable"])
                             if m["base_rate_unstable"] else None)
    m["tp"], m["fp"], m["fn"] = tp, fp, int((~pred_un & (y_un > 0.5)).sum())
    m["n_pred_pos"] = int(pred_un.sum())     # 失败模式预演:≈n 即"见谁都判失稳"
    if t_true:
        t_prob, t_true = np.array(t_prob), np.array(t_true)
        # 细分五路(07-14 裁决:功角/低电压/过电压/高频/低频)
        for k, name in enumerate(TS_SUBTYPES):
            yp, yt = t_prob[:, k] > 0.5, t_true[:, k] > 0.5
            m[f"sub_{name}_recall"] = float((yp & yt).sum() / max(yt.sum(), 1))
            m[f"sub_{name}_precision"] = float((yp & yt).sum() / max(yp.sum(), 1))
            m[f"sub_{name}_n"] = int(yt.sum())
            m[f"sub_{name}_base_rate"] = float(yt.mean())    # 基率 = 恒定预测的精确率
            m[f"sub_{name}_pred_pos_rate"] = float(yp.mean())  # 模型输出正类的比例
        # 三大类 = 子类聚合(主报告口径)
        for name, cols in TS_TYPE_FROM_SUB.items():
            yp = (t_prob[:, list(cols)] > 0.5).any(axis=1)
            yt = (t_true[:, list(cols)] > 0.5).any(axis=1)
            m[f"type_{name}_recall"] = float((yp & yt).sum() / max(yt.sum(), 1))
            m[f"type_{name}_precision"] = float((yp & yt).sum() / max(yp.sum(), 1))
            m[f"type_{name}_base_rate"] = float(yt.mean())
            m[f"type_{name}_pred_pos_rate"] = float(yp.mean())
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据根(gridfm-data)")
    ap.add_argument("--labels-root", default="runs", help="ts_labels_* 所在目录")
    ap.add_argument("--cct", nargs="*", default=["runs/ts_cct_cepri36"],
                    help="cct.jsonl 目录(可多个)")
    ap.add_argument("--batches", nargs="*", default=None,
                    help="标签批白名单(目录名);缺省 = LIBS 里全部**已收口**批"
                         "(收口门:labels_v1 + summary 行数对齐 + audit 0 失败;"
                         "在产/半截批一律拒纳)")
    ap.add_argument("--v1-dir", default="runs/encoder_v1")
    ap.add_argument("--encoder", choices=["v1", "random"], default="v1")
    ap.add_argument("--out", default=None, help="默认 runs/ts_reasoner_<encoder>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=12000,
                    help="**2026-07-25 由 2000 提到 12000(R3 判决)**:此前全部判稳实验用的"
                         "4000 步 = batch16 × 4000 / 23,250 例 ≈ **2.75 epoch,从未收敛** —— "
                         "六座 r1b/r2 全部在末步 bad=0 且 eval loss 仍在降。实测真收敛点"
                         "11,000~13,000 步(三种子均早停),把 4000→20000 后 118 [内] acc "
                         "0.84878→0.94339(+9.46pp)。故默认改 12000(留早停兜底);"
                         "**复现旧 run 必须显式写 --steps 4000**。全档 "
                         "docs/R3-训练预算-判决结果-20260725.md")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--queries", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="**每个训练家族**各自的 [内] 留出方式占比(按 sample_id 切,"
                         "非按行;逐族独立 permutation,2026-07-16 拓扑维度 2 起)")
    ap.add_argument("--holdout-seed", type=int, default=None,
                    help="**留出专用种子**(R1,2026-07-25):缺省 None = 沿用 --seed"
                         "(历史行为逐位不变)。给定同一值 → 多种子共享同一留出集,"
                         "训练随机性仍由 --seed 决定 —— 这是做干净 ensemble 的前提:"
                         "此前各种子留出集互不相同,跨种子平均没有无泄漏验证口径")
    ap.add_argument("--train-mode-frac", type=float, default=1.0,
                    help="方式消融(2026-07-31 方式扩容重看):**切分之后**只在训练侧按方式"
                         "白名单保留 frac 比例(逐族独立 permutation,嵌套:同 mode-seed 下 "
                         "0.25 的保留集 ⊂ 0.5 的保留集)。留出/[外] 一行不动 —— [内] 5700 行"
                         "与 --holdout-seed 17260 逐位不变,评估口径跨剂量可比。默认 1.0 = "
                         "不消融 = 历史行为逐位不变")
    ap.add_argument("--train-mode-seed", type=int, default=20260731,
                    help="方式消融的抽样种子(与训练/留出种子解耦;全部臂共用同一值 → "
                         "各剂量臂的保留集嵌套且跨种子固定,种子只贡献训练随机性)")
    ap.add_argument("--train-mode-families", nargs="*", default=None,
                    help="方式消融只作用于这些家族(2026-07-31 审查改法:设门臂与采购决策"
                         "同构 —— 采购只加 ieee118 方式,消融就只砍 ieee118,cepri36 完整"
                         "保留,避免跨族迁移份额混进归因)。缺省 None = 全部训练家族"
                         "(消融旗给定时)")
    ap.add_argument("--split", choices=["mode_holdout", "random"], default="mode_holdout",
                    help="[内] 划分口径:mode_holdout=按方式留出(默认,生产口径,方式不泄漏);"
                         "random=行级随机切分(W4.2 文献对表口径,允许方式泄漏、偏乐观,仅诊断,"
                         "**不产生产模型**;[外] ieee39 仍整族只评估,D14 不破)")
    ap.add_argument("--out-families", nargs="+", default=list(OUT_FAMILIES),
                    help="[外] 评估家族名单(整族不入训)。默认 ieee39(行为逐位不变);"
                         "双外族实验加 cepri36(其标签退出训练、只评估;ieee39 永不可移出)")
    ap.add_argument("--side-channel", action="store_true",
                    help="W5 侧通道(D13 推理器侧):故障电气严重度特征(fault_severity,"
                         "档案+故障参数的确定函数,有泄漏单测)标准化后拼进事件向量;"
                         "标准化统计只在训练集拟合,存进检查点。默认关=行为逐位不变")
    ap.add_argument("--dyn-side", action="store_true",
                    help="动态参数/能量函数侧通道(审查 A2,信息上限第五因子):从 v0.3 "
                         "档案列取 DATALIB 正确值 Tj/X'd(dyn_side 模块,泄漏单测锁)"
                         "标准化后拼事件向量;报告并列裸值基线行与逐特征零点行。"
                         "默认关=行为逐位不变")
    ap.add_argument("--w-stable", type=float, default=1.0)
    ap.add_argument("--w-type", type=float, default=0.5)
    ap.add_argument("--w-cct", type=float, default=1.0)
    ap.add_argument("--w-group", type=float, default=0.5)
    ap.add_argument("--subtype-pos-weight", choices=["none", "auto"], default="none",
                    help="子类头 BCE 正类加权(B1):auto=按训练集失稳行子类基率 p 取 "
                         "(1-p)/p 截断 [1,25];none=历史行为逐位不变")
    ap.add_argument("--group-pos-weight", choices=["none", "auto"], default="none",
                    help="机群头 BCE 正类加权(B6):auto=按训练集机群行(group_valid × "
                         "发电机位)正类率 p 取 (1-p)/p 截断 [1,25] 全局标量;"
                         "none=历史行为逐位不变")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=8)
    # **早停下限**(2026-07-27 用户裁"改 A+C";`docs/早停虚假触发-实测与改法-20260727.md`)。
    # 早停锚是 [内] 总损失、上线门却是 118 acc,两者会脱节:实测 `r6_all_s0` 的 loss 在
    # 0.78~0.84 无趋势抖动(抖幅 ≫ 其改善速率)触发 patience,而 **acc 与 pr_auc 恰停在
    # 全程最高点**,该座 118 acc 仅 0.8241(健康座 0.90~0.95);这一座把该臂座间 sd 从
    # 1.596pp 抬到 5.045pp(所需座数 ∝ sd² → 10 倍)。B 协议已保护"报数落固定步锚",
    # 但早停会把轨迹**截断在锚之前**,让锚根本不存在。本旗即对症:**在 min_steps 之前
    # 一律不许早停**(best.pt 的选择语义不变)。**默认 0 = 关闭,老命令逐位不变。**
    ap.add_argument("--min-steps", type=int, default=0,
                    help="早停不得在该步数之前触发(建议设为最大预注册锚;0=关闭)")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 例(冒烟)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-last", type=int, default=0,
                    help="只保留最近 N 个周期检查点(best.pt 永不删)。**默认 0 = 不剪枝,"
                         "历史行为逐位不变**。动因(2026-07-25):解冻臂单个检查点 137MB"
                         "(冻结臂 15.7MB),20000 步 × eval_every 200 = 至多 100 个/座,"
                         "6 座最坏 ~60GB,会在长跑中途打满磁盘。--keep-last 3 后降到 ~1.3GB;"
                         "**--resume 只需最新一个,best.pt 另存,故剪枝对二者都安全**")
    ap.add_argument("--unfreeze-encoder", choices=list(UNFREEZE_CHOICES), default="none",
                    help="R2 编码器解冻臂(2026-07-25):none=历史行为逐位不变(编码器全冻);"
                         "last2=末两层 GPS + ln_out;all=全部编码器参数。"
                         "**解冻产物一律落任务分叉副本 runs/ts_reasoner_r2_{arm}_s{N}/;"
                         "共享 runs/encoder_v1/v1.pt 与 normalizer.npz 永不覆写、"
                         "normalizer 永不重拟合**")
    ap.add_argument("--encoder-lr", type=float, default=1e-5,
                    help="解冻臂**编码器参数组**的基准 lr(推理器组仍用 --lr)。"
                         "调度按各组 base_lr 等比缩放,见 R2 第 3 条")
    a = ap.parse_args()
    out = Path(a.out or f"runs/ts_reasoner_{a.encoder}")
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted((out / "ckpt").glob("step_*.pt"))
    if existing and not a.resume:
        raise SystemExit(f"[ts] {out}/ckpt 已有 {len(existing)} 个检查点且未 --resume,"
                         f"拒绝覆写;换 --out 或显式 --resume")

    v1 = Path(a.v1_dir)
    # **共享冻结基座指纹双验**(2026-07-25 对抗审查 major:此前训练入口从不验,而
    # ts_screen / g15_train_readout / g_diffusion_train 都验)。三臂对照的全部意义在于
    # "同一个 v1 起跳";基座若在两次烧机之间被动过,±2pp 上线门与遗忘门一起作废,而
    # **事后无任何日志能分辨**(ts_report 的 encoder 字段是假凭据,照写 "v1")。
    # 故:装载前断言 + 把指纹写进 ts_report.json,让"同基座"从此有机制凭据。
    v1_fp = _blake16(v1 / "v1.pt")
    nm_fp = _blake16(v1 / "normalizer.npz")
    if a.v1_dir == ap.get_default("v1_dir"):
        if v1_fp != FP_V1 or nm_fp != FP_NM:
            raise SystemExit(f"[ts] ✋ 共享冻结基座指纹不符(v1.pt {v1_fp} 期望 {FP_V1};"
                             f"normalizer {nm_fp} 期望 {FP_NM})—— 基座被改过,"
                             f"三臂不同源,拒绝训练")
        print(f"[ts] 冻结基座指纹双验:v1.pt {v1_fp} / normalizer {nm_fp} —— 过")
    else:
        print(f"[ts] ⚠ 非默认 --v1-dir({a.v1_dir}):指纹 v1.pt {v1_fp} / "
              f"normalizer {nm_fp}(不设门,如实入档)")
    ck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(a.seed)
    enc, _dec = build_models(cfg)
    if a.encoder == "v1":
        enc.load_state_dict(ck["model_state"])
    enc.to(dev)
    # **R2 第 1 条**(2026-07-25):按解冻臂放开对应层。凭据 = apply_unfreeze 打印的
    # 可训参数量 + 子模块清单 + train/eval 敏感子模块清单(none 臂打印 0/…,行为逐位不变)。
    enc_train_params = apply_unfreeze(enc, a.unfreeze_encoder)
    enc_unfrozen = a.unfreeze_encoder != "none"
    # 共享基座目录守卫(审查 minor:原写法只防解冻臂、只拒"完全相等",子目录与冻结臂不设防)
    _v1r, _outr = v1.resolve(), out.resolve()
    if _v1r == _outr or _v1r in _outr.parents or _outr in _v1r.parents:
        raise SystemExit(f"[ts] ✋ --out({_outr})与共享基座目录({_v1r})重叠 —— "
                         f"v1.pt / normalizer.npz 永不覆写,拒绝运行")
    norm = Normalizer.load(v1 / "normalizer.npz")
    print(f"[ts] 编码器 = {a.encoder}"
          f"({'解冻:' + a.unfreeze_encoder if enc_unfrozen else '冻结'},"
          f"d{cfg.d}/L{cfg.n_layers}/K{cfg.n_global});normalizer = 冻结档(禁重拟合)")

    assert_no_eval_only_batches(a.batches)   # 定向补测批入训锁(响亮拒绝,先于装载)
    examples, raw_by_sid, cov = load_examples(Path(a.root), Path(a.labels_root), a.cct,
                                              only_batches=a.batches)
    if not examples:
        raise SystemExit("[ts] 无可用训练例(全部批被收口门拒纳?见上方拒纳原因)")
    normed_by_sid = {}
    for sid, s in raw_by_sid.items():
        c = copy.deepcopy(s)
        norm.transform(c)
        normed_by_sid[sid] = c
    batch_desc = ", ".join(f"{b['batch']}[{b['family']}] {b['rows']} 行"
                           for b in cov["batches"])
    print(f"[ts] 入训批:{batch_desc}")
    print(f"[ts] 标签行 {cov['rows']}(infeasible 跳 {cov['skipped_infeasible']} / "
          f"缺样本跳 {cov['skipped_no_sample']} / **方式门跳 "
          f"{cov['skipped_low_base_vmin']}**(基态 vmin 不合格,标签与故障无关) / "
          f"superseded 跳 {cov['skipped_superseded']}(v2 重产作废));"
          f"例 {len(examples)};"
          f"CCT 挂标 {cov['cct_joined']} / 机群有效 {cov['group_valid']}"
          f"(breakup 回退 {cov['breakup']})")

    # 划分(D14):**逐训练家族**按方式留出 [内](同拓扑豁免,如实标注);[外] 家族整族评估。
    # **逐族留出**(2026-07-16,拓扑维度 2 起必需):此前按单一 cepri36 sample_id 池切分,
    # 两个训练家族混切会让留出集的家族占比随各族方式数漂移(118 90 方式 vs cepri36 88),
    # 甚至可能整族缺席 [内] → "同拓扑内是否成立"在某一族上无从检验。改为每族独立
    # permutation 留出 holdout_frac,再合并。**每族独立 rng**(seed + 族名哈希),否则
    # 两族的留出选择被同一序列耦合。
    out_fams = resolve_out_families(a.out_families)
    train_fams = [f for f in LIBS if f not in out_fams]
    held_by_fam = {}
    # 留出种子与训练种子解耦(R1):--holdout-seed 给定时多种子共享同一留出集
    split_seed = a.seed if a.holdout_seed is None else a.holdout_seed
    if a.holdout_seed is not None:
        print(f"[ts] 留出种子 = {split_seed}(与训练种子 {a.seed} 解耦;"
              f"多种子共享留出 → 可做无泄漏 ensemble)", flush=True)
    if a.split == "mode_holdout":
        held_by_fam = mode_holdout_by_family(examples, train_fams, a.holdout_frac,
                                             split_seed)
        held_all = {f: set(v) for f, v in held_by_fam.items()}
        train_exs = [x for x in examples if x["family"] in held_all
                     and x["sample_id"] not in held_all[x["family"]]]
        in_exs = [x for x in examples if x["family"] in held_all
                  and x["sample_id"] in held_all[x["family"]]]
    else:   # random —— W4.2 文献对表口径:训练家族内**行级**随机切分(允许方式泄漏,偏乐观)。
        # 仅诊断;[外] ieee39 仍整族只评估(下方 out_exs 不变,D14 不破)。salt 令其与
        # mode_holdout 的 permutation 序列解耦;fam_exs 先按 tag 排序,与装载顺序无关(确定性)。
        train_exs, in_exs = [], []
        for fam in train_fams:
            fam_exs = sorted((x for x in examples if x["family"] == fam), key=lambda x: x["tag"])
            if not fam_exs:
                continue
            rng_f = np.random.default_rng([split_seed, family_seed(fam), 20260715])
            n_held = max(1, int(len(fam_exs) * a.holdout_frac))
            held_pos = set(rng_f.permutation(len(fam_exs))[:n_held].tolist())
            held_tags = []
            for k, x in enumerate(fam_exs):
                (in_exs if k in held_pos else train_exs).append(x)
                if k in held_pos:
                    held_tags.append(x["tag"])
            held_by_fam[fam] = sorted(held_tags)   # 键=家族(过实测族断言);值=留出行 tag
    out_exs = [x for x in examples if x["family"] in out_fams]
    assert not any(x["family"] in out_fams for x in train_exs), \
        f"[外] 家族 {out_fams} 整族只准评估,禁入训练(D14)"
    assert not any(x["family"] in out_fams for x in in_exs), "[内] 不得混入 [外] 家族"
    kept_modes_by_fam = None
    if a.train_mode_frac < 1.0:
        # 方式消融:**只滤 train_exs**(in/out 一行不动 → 评估口径跨剂量逐位可比)。
        # 嵌套抽样:逐族对训练侧 sid 排序后按固定 permutation 取前 k —— 同 mode-seed 下
        # frac 小的保留集是 frac 大的子集,剂量-响应各臂比的是"同一批方式被逐步砍掉",
        # 不是各臂各抽一批(后者把抽样方差混进剂量效应)。
        assert 0.0 < a.train_mode_frac < 1.0, f"--train-mode-frac 取 (0,1):{a.train_mode_frac}"
        _all_fams = sorted({x["family"] for x in train_exs})
        _ablate = set(a.train_mode_families) if a.train_mode_families else set(_all_fams)
        _unknown = _ablate - set(_all_fams)
        assert not _unknown, f"--train-mode-families 含非训练家族 {sorted(_unknown)}(训练家族 = {_all_fams})"
        kept_modes_by_fam = {}
        for fam in _all_fams:
            sids = sorted({x["sample_id"] for x in train_exs if x["family"] == fam})
            if fam not in _ablate:            # 不消融的族:全保留
                kept_modes_by_fam[fam] = sids
                continue
            rng_m = np.random.default_rng([a.train_mode_seed, family_seed(fam)])
            perm = rng_m.permutation(len(sids))
            k = max(1, int(round(a.train_mode_frac * len(sids))))
            kept_modes_by_fam[fam] = sorted(sids[i] for i in perm[:k])
        n_before = len(train_exs)
        train_exs = [x for x in train_exs
                     if x["sample_id"] in set(kept_modes_by_fam[x["family"]])]
        print(f"[ts] **方式消融** --train-mode-frac {a.train_mode_frac}"
              f"(mode-seed {a.train_mode_seed},嵌套抽样):保留方式 "
              + " + ".join(f"{f} {len(v)}" for f, v in sorted(kept_modes_by_fam.items()))
              + f";训练行 {n_before} -> {len(train_exs)};留出/[外] 未动")
    if a.limit:
        # **按族分层取**(2026-07-14 对抗审查 major):examples 是按 LIBS 顺序 append 的
        # (cepri36 全部 → ieee118 → ieee39),裸切片 train_exs[:limit] 在 limit < cepri36
        # 例数时**一个 118 例都取不到** → 冒烟必然退化成拓扑维度 1,却仍打印/落盘
        # "topology_dim=2"(held_by_fam 在截断前算好)。后果:冒烟给出"双家族走通"的
        # 假绿灯,而 118 专属路径(lib118 装载、118 图、Catalog、gen_mask、机群)一行没跑过。
        def _stratify(exs, n):
            fams = sorted({x["family"] for x in exs})
            per = max(1, n // max(len(fams), 1))
            out_ = []
            for f in fams:
                out_ += [x for x in exs if x["family"] == f][:per]
            return out_
        train_exs = _stratify(train_exs, a.limit)
        in_exs = _stratify(in_exs, max(1, a.limit // 4))
        out_exs = _stratify(out_exs, max(1, a.limit // 4))
    # **拓扑维度以实测为准**,不以 held_by_fam 的声称为准(--limit / 收口门拒纳 / 某族
    # 零例 都会让实际训练集掉族;声称与实际不符 = 静默假结论,直接炸)
    train_fams_actual = sorted({x["family"] for x in train_exs})
    topo_dim = len(train_fams_actual)
    assert set(train_fams_actual) == set(held_by_fam), (
        f"训练集实际家族 {train_fams_actual} ≠ 划分声称的 {sorted(held_by_fam)}"
        "(--limit 截断或某族零例;拓扑维度会静默塌陷,拒绝继续)")
    fam_desc = " + ".join(f"{f} {len(held_by_fam[f])} 留出方式" for f in held_by_fam)
    print(f"[ts] 训练家族 {train_fams_actual}(**拓扑维度 {topo_dim}**,实测);"
          f"train {len(train_exs)} / [内] {len(in_exs)}({fam_desc},同拓扑)"
          f" / [外] {len(out_exs)}({'+'.join(out_fams)} 整族)")
    for fam in held_by_fam:
        n_tr = sum(1 for x in train_exs if x["family"] == fam)
        n_in = sum(1 for x in in_exs if x["family"] == fam)
        print(f"[ts]   {fam}: train {n_tr} / [内] {n_in}")
    held_key = json.dumps(held_by_fam, sort_keys=True)   # 检查点一致性校验的规范化键

    # ---- W5 侧通道(D13):故障电气严重度特征 → 标准化 → 拼事件向量 ----
    event_dim_total = EVENT_PARAM_DIM
    side_mu = side_sd = None
    side_feature_names = []
    if a.side_channel:
        from gridfm.reason.fault_severity import (
            FEATURE_NAMES as SIDE_FEATS, compute_fault_severity_features)
        side_feature_names = list(SIDE_FEATS)
        _fs_cache: dict = {}

        def _side_vec(ex):
            key = (ex["sample_id"], int(ex["fault_branch_row"]),
                   round(float(ex["event_params"][0]), 4))
            if key not in _fs_cache:
                f = compute_fault_severity_features(
                    raw_by_sid[ex["sample_id"]], int(ex["fault_branch_row"]),
                    float(ex["event_params"][0]))
                _fs_cache[key] = np.array([f[n] for n in SIDE_FEATS], dtype=np.float64)
            return _fs_cache[key]

        all_exs = train_exs + in_exs + out_exs
        mat_train = np.stack([_side_vec(ex) for ex in train_exs])
        side_mu = mat_train.mean(axis=0)                      # 标准化只在训练集拟合
        side_sd = np.maximum(mat_train.std(axis=0), 1e-6)
        for ex in all_exs:
            ex["event_ext"] = ((_side_vec(ex) - side_mu) / side_sd).astype(np.float32)
        event_dim_total = EVENT_PARAM_DIM + len(SIDE_FEATS)
        print(f"[ts] 侧通道开:{len(SIDE_FEATS)} 维故障电气严重度特征(缓存 {len(_fs_cache)} 键)"
              f";事件向量 {EVENT_PARAM_DIM}+{len(SIDE_FEATS)}={event_dim_total} 维"
              f";标准化 mu/sd 取自 train {len(train_exs)} 例")

    # ---- 动态参数侧通道(审查 A2;D13):v0.3 正确值 Tj/X'd → 标准化 → 拼事件向量 ----
    dyn_feature_names = []
    dyn_mu = dyn_sd = None
    dyn_train_mat = dyn_eval_mats = None
    if a.dyn_side:
        from gridfm.data.dataset import library_files
        from gridfm.reason.dyn_side import (
            DYN_FEATURE_NAMES, compute_dyn_side_features, dyn_zbus)
        from gridfm.schema.sample import GridSample
        dyn_feature_names = list(DYN_FEATURE_NAMES)
        # load_library 走 v0 视图(25 列,legacy tj/xdp 为 cepri36/ieee39 内联假 0)——
        # 动态参数必须全 schema 重载(sid = npz stem,装载与 load_library 严格同源)
        path_by_stem = {}
        for _fam, (_lib, _b) in LIBS.items():
            for p in library_files(Path(a.root) / _lib):
                path_by_stem[p.stem] = p
        _full_cache: dict = {}
        _zbus_cache: dict = {}    # Zbus 只依赖方式(与故障无关)→ 按 sid 缓存,免逐故障重复求逆
        _dyn_cache: dict = {}

        def _dyn_vec(ex):
            key = (ex["sample_id"], int(ex["fault_branch_row"]),
                   round(float(ex["event_params"][3]), 4))       # (sid, 支路, dur)
            if key not in _dyn_cache:
                sid = ex["sample_id"]
                if sid not in _full_cache:
                    _full_cache[sid] = GridSample.load_npz(path_by_stem[sid],
                                                           full_schema=True)
                    _zbus_cache[sid] = dyn_zbus(_full_cache[sid])
                f = compute_dyn_side_features(_full_cache[sid],
                                              int(ex["fault_branch_row"]),
                                              float(ex["event_params"][3]),
                                              zbus=_zbus_cache[sid])
                _dyn_cache[key] = np.array([f[n] for n in DYN_FEATURE_NAMES],
                                           dtype=np.float64)
            return _dyn_cache[key]

        all_exs = train_exs + in_exs + out_exs
        dyn_train_mat = np.stack([_dyn_vec(ex) for ex in train_exs])
        dyn_mu = dyn_train_mat.mean(axis=0)                   # 标准化只在训练集拟合
        dyn_sd = np.maximum(dyn_train_mat.std(axis=0), 1e-6)
        for ex in all_exs:
            v = ((_dyn_vec(ex) - dyn_mu) / dyn_sd).astype(np.float32)
            ex["event_ext"] = (np.concatenate([ex["event_ext"], v])
                               if "event_ext" in ex else v)
        # 裸值基线/零点行用:评估集特征矩阵(标准化后)与标签,终评时算
        dyn_eval_mats = {}
        for nm, exs_e in (("train", train_exs), ("in", in_exs), ("out", out_exs)):
            dyn_eval_mats[nm] = (
                np.stack([((_dyn_vec(ex) - dyn_mu) / dyn_sd) for ex in exs_e])
                if exs_e else np.zeros((0, len(DYN_FEATURE_NAMES))))
        event_dim_total += len(DYN_FEATURE_NAMES)
        print(f"[ts] 动态参数侧通道开:{len(DYN_FEATURE_NAMES)} 维(样本全 schema 重载 "
              f"{len(_full_cache)} 个;特征缓存 {len(_dyn_cache)} 键);事件向量共 "
              f"{event_dim_total} 维;标准化 mu/sd 取自 train {len(train_exs)} 例")

    weights = {"stable": a.w_stable, "type": a.w_type, "cct": a.w_cct, "group": a.w_group}
    if a.subtype_pos_weight == "auto":
        # B1:基率取自训练集失稳行(确定性,resume 复算一致);经 weights 字典流转,
        # 检查点一致性校验(weights 不符拒续训)与报告入档自动覆盖
        sub_mat = np.stack([x["y_subtypes"] for x in train_exs if x["y_unstable"] > 0.5])
        sub_p = sub_mat.mean(axis=0)
        weights["type_pos_weight"] = [
            round(float(np.clip((1.0 - p) / p, 1.0, 25.0)), 3) if p > 0 else 1.0
            for p in sub_p]
        print("[ts] 子类正类加权 auto:失稳行基率 "
              + " ".join(f"{n}={p:.4f}" for n, p in zip(TS_SUBTYPES, sub_p))
              + " -> pos_weight "
              + " ".join(f"{n}={v}" for n, v in zip(TS_SUBTYPES, weights["type_pos_weight"])))
    if a.group_pos_weight == "auto":
        # B6:正类率 = 训练集 group_valid 例的发电机位 y_group 均值(确定性,resume 一致);
        # 经 weights 字典流转,检查点一致性校验与报告入档自动覆盖
        _gm_cache: dict = {}
        g_pos = g_tot = 0.0
        for x in train_exs:
            if x["group_valid"] and x["y_group"] is not None:
                sid = x["sample_id"]
                if sid not in _gm_cache:
                    _gm_cache[sid] = gen_node_mask(raw_by_sid[sid])
                gm = _gm_cache[sid]
                g_pos += float(x["y_group"][gm].sum())
                g_tot += float(gm.sum())
        g_p = g_pos / max(g_tot, 1.0)
        weights["group_pos_weight"] = (
            round(float(np.clip((1.0 - g_p) / g_p, 1.0, 25.0)), 3) if g_p > 0 else 1.0)
        print(f"[ts] 机群正类加权 auto:机群行正类率 {g_p:.4f}"
              f"({int(g_pos)}/{int(g_tot)})-> pos_weight {weights['group_pos_weight']}")
    reasoner = TSReasoner(d=cfg.d, n_queries=a.queries, n_layers=a.layers,
                          event_dim=event_dim_total).to(dev)
    print(f"[ts] 推理器参数 {reasoner.num_params()/1e6:.2f}M(D9 目标 1e6 量级);"
          f"损失权重 {weights}")
    # **R2 第 3 条**:每组存 base_lr(调度按 base_lr 等比缩放,见下方 warm/cos)。
    # 冻结臂 param_groups 与旧写法逐位同(同一组、同序、同 lr),base_lr 键对 Adam 惰性。
    groups = [{"params": list(reasoner.parameters()), "lr": a.lr, "base_lr": a.lr,
               "group_name": "reasoner"}]
    if enc_unfrozen:
        groups.append({"params": [p for _nm, p in enc_train_params],
                       "lr": a.encoder_lr, "base_lr": a.encoder_lr,
                       "group_name": "encoder"})
    opt = torch.optim.Adam(groups, lr=a.lr)
    print("[ts] 优化器参数组:" + " | ".join(
        f"{g['group_name']} n_tensors={len(g['params'])} "
        f"n_params={sum(p.numel() for p in g['params'])} base_lr={g['base_lr']:.2e}"
        for g in opt.param_groups))
    use_amp = dev.type == "cuda"
    amp_dtype = torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    rng = np.random.default_rng(a.seed)
    eval_hist = []          # 每次评估的 (step, loss, acc, pr_auc, recall);判读收敛用
    # 续训累计账初值(不续训时全为 0/空;续训时自检查点读回,保证与不续训座可比)
    acc0 = {"n_nonfinite": 0, "n_steps": 0, "gn_sum": None, "gn_clipped": None,
            "gn_n": 0, "loss_first": None, "n_resumes": 0}
    start, best_val, bad = 0, float("inf"), 0
    ckpts = sorted((out / "ckpt").glob("step_*.pt"))
    if a.resume and ckpts:
        rck = torch.load(ckpts[-1], map_location=dev, weights_only=False)
        if rck.get("encoder") != a.encoder:
            raise SystemExit(f"[ts] 检查点编码器 {rck.get('encoder')} 与 --encoder 不符")
        # held_ids schema 2026-07-16 由**扁平 list** 改为 **{family: [ids]}**(逐族留出)。
        # 旧检查点的扁平 list 与新 dict 天然不等 → 自动拒续训(正确行为:粮仓与划分
        # 都变了,续训 = 混口径)。报错文案明说须换 --out 重训。
        if json.dumps(rck.get("held_ids"), sort_keys=True) != held_key:
            raise SystemExit("[ts] 检查点划分与本次不符(seed/holdout/家族集变更会泄漏留出集"
                             ";旧扁平 held_ids 检查点亦在此拒绝——须换 --out 重训)")
        if rck.get("weights") != weights:
            raise SystemExit(f"[ts] 检查点损失权重 {rck.get('weights')} 与本次不符")
        if bool(rck.get("config", {}).get("side_channel")) != bool(a.side_channel):
            raise SystemExit("[ts] 检查点侧通道开关与本次不符(事件向量维度不同,续训 = 混口径"
                             ";须换 --out 重训)")
        if bool(rck.get("config", {}).get("dyn_side")) != bool(a.dyn_side):
            raise SystemExit("[ts] 检查点动态参数侧通道开关与本次不符(事件向量维度不同,"
                             "续训 = 混口径;须换 --out 重训)")
        # 审查 major:旧写法 `ck_b is not None and ...` 会让**缺该键的旧检查点**
        # 静默通过 → 粮仓变更后续训混口径。改为必须存在且相等。
        ck_b = rck.get("batches")
        now_b = sorted(b["batch"] for b in cov["batches"])
        if ck_b != now_b:
            raise SystemExit(f"[ts] 检查点入训批 {ck_b} 与本次 {now_b} 不符"
                             f"(粮仓变更或旧检查点无 batches 键;须换 --out 重训,"
                             f"不得续训混口径)")
        # **R2**:--resume 的五道校验看不见解冻臂 —— 补第六道。不补则"冻结检查点 +
        # 解冻续训"会恢复编码器那组的 Adam 动量态、而编码器权重来自 v1.pt(口径混)。
        if str(rck.get("unfreeze", "none")) != a.unfreeze_encoder:
            raise SystemExit(f"[ts] 检查点解冻臂 {rck.get('unfreeze', 'none')!r} 与本次 "
                             f"{a.unfreeze_encoder!r} 不符(编码器权重出身不同,续训 = 混口径"
                             ";须换 --out 重训)")
        if enc_unfrozen:
            if "encoder_state" not in rck:
                raise SystemExit("[ts] 解冻臂检查点缺 encoder_state,无法续训(须换 --out 重训)")
            if float(rck.get("encoder_lr", -1.0)) != float(a.encoder_lr):
                raise SystemExit(f"[ts] 检查点 encoder_lr {rck.get('encoder_lr')} 与本次 "
                                 f"{a.encoder_lr} 不符(编码器组学习率变了 = 混配方;"
                                 f"须换 --out 重训)")
            enc.load_state_dict(rck["encoder_state"])
            print("[ts] 续训:编码器权重自检查点 encoder_state 恢复(非 v1.pt)")
        reasoner.load_state_dict(rck["reasoner_state"])
        opt.load_state_dict(rck["optimizer_state"])
        # load_state_dict 用存档 group 字典整体替换现有 group 设置 —— 旧检查点没有
        # base_lr/group_name 键,不回填则下方调度 pg["base_lr"] 直接 KeyError(且回填后
        # 才保证 base_lr 来自本次命令行而非存档)。
        for pg, g in zip(opt.param_groups, groups):
            pg["base_lr"], pg["group_name"] = g["base_lr"], g["group_name"]
        if "scaler_state" in rck:
            scaler.load_state_dict(rck["scaler_state"])
        eval_hist = list(rck.get("eval_history", []))
        acc0 = {"n_nonfinite": int(rck.get("acc_n_nonfinite", 0)),
                "n_steps": int(rck.get("acc_n_steps", 0)),
                "gn_sum": rck.get("acc_gn_sum"), "gn_clipped": rck.get("acc_gn_clipped"),
                "gn_n": int(rck.get("acc_gn_n", 0)),
                "loss_first": rck.get("acc_loss_first"),
                "n_resumes": int(rck.get("acc_n_resumes", 0)) + 1}
        start, best_val, bad = rck["step"], rck["best_val"], rck["bad"]
        rng.bit_generator.state = rck["rng"]
        print(f"[ts] 续训自 step {start}"
              + ("(GradScaler 状态已恢复)" if "scaler_state" in rck else
                 "(⚠ 旧检查点无 scaler_state:scale 从 65536 重新起跳,"
                 "该座 amp_nonfinite_rate 与不续训座不可比)"))

    warm = max(1, int(0.05 * a.steps))
    hist = []
    # **第 8 个静默失效点**(2026-07-25 冒烟新发现,不在 §0.0-R2 原表):AMP 溢出跳步。
    # 解冻把反传图从"推理器 1.3M"扩到"编码器 10.1M + 推理器",fp16 中间量更容易溢出;
    # GradScaler 遇 inf/nan 会**静默跳过该步**(loss/指标照常出数)。若跳步率高,解冻臂
    # 实际训练步数远少于名义 4000 → "解冻没用"是欠训练的假象。故计数并入报告。
    # 续训座从检查点接着累计(不续训座 acc0 全 0,与旧行为逐位同)
    n_nonfinite = acc0["n_nonfinite"]
    n_resumes = acc0["n_resumes"]
    enc_grad_seen = False
    # 逐组裁剪账(过账用):偏离原文改逐组裁,须能事后核对"推理器组是否被编码器连累"。
    gn_sum = dict(acc0["gn_sum"]) if acc0["gn_sum"] else {"reasoner": 0.0, "encoder": 0.0}
    gn_clipped = (dict(acc0["gn_clipped"]) if acc0["gn_clipped"]
                  else {"reasoner": 0, "encoder": 0})
    gn_n = acc0["gn_n"]
    n_steps_prev = acc0["n_steps"]      # 续训前已训步数(用于 amp 跳步率的分母)
    loss_first_prev = acc0["loss_first"]
    AMP_WARMUP_STEPS = 50      # GradScaler 回退窗口(每次溢出 scale 减半,50 步足够收敛)
    for step in range(start, a.steps):
        lr = (a.lr * (step + 1) / warm if step < warm else
              a.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, a.steps - warm - 1))))
        # **R2 第 3 条**:旧写法 `pg["lr"] = lr` 把**每一组**都设成由 a.lr 推出的同一个值 ——
        # 编码器组的 1e-5 会在 step 0 被抹成 3e-4(30×),"小 lr 微调"从未发生,而日志、
        # 报告、指标全都照常出数(静默)。改法:各组按自己的 base_lr 等比缩放。
        # 推理器组 base_lr == a.lr,取 lr 原值 → 冻结臂 lr 序列**逐位不变**(不走乘除法)。
        scale = (lr / a.lr) if a.lr else 0.0      # --lr 0 时不炸(审查 minor)
        for pg in opt.param_groups:
            pg["lr"] = lr if pg["base_lr"] == a.lr else pg["base_lr"] * scale
        if step in (start, start + 1):      # 凭据:step 0/1 打印**各组实际** lr
            print("[ts] step %d 各组实际 lr:" % step + " ".join(
                f"{pg['group_name']}={pg['lr']:.3e}(base {pg['base_lr']:.1e})"
                for pg in opt.param_groups), flush=True)
        idx = rng.choice(len(train_exs), size=min(a.batch, len(train_exs)), replace=False)
        datas = [_ts_data(train_exs[int(i)], normed_by_sid[train_exs[int(i)]["sample_id"]],
                          raw_by_sid[train_exs[int(i)]["sample_id"]], cfg) for i in idx]
        batch = Batch.from_data_list(datas).to(dev)
        opt.zero_grad()
        _outs, loss, diag = _forward(enc, reasoner, batch, use_amp, amp_dtype, dev, weights,
                                     enc_grad=enc_unfrozen)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        # **凭据必须取在裁剪之前**(2026-07-25 零机时冒烟抓到:裁剪后测是**假凭据**)——
        # clip_grad_norm_ 在总范数 = inf 时把 clip_coef 算成 max_norm/inf = 0,再把**所有**
        # 梯度乘成 0;而 AMP GradScaler 初始 scale 65536 + 解冻后加深的反传图,**首几步溢出
        # 属正常回退**(该步被 scaler.step 跳过、scale 减半)。在裁剪之后读编码器梯度范数,
        # 只要那步溢出就必然读到 0.0 → 误报"梯度没到编码器"。
        eg = [p.grad for _nm, p in enc_train_params if p.grad is not None]
        enc_gn = (float(torch.norm(torch.stack([g.detach().float().norm() for g in eg])))
                  if eg else 0.0)
        # **R2 第 6 条**:旧写法只裁 reasoner.parameters() —— 解冻后编码器梯度**全程不裁**
        # (裁剪本是防梯度尖峰的护栏,恰恰是新解冻的层最需要)。
        # **偏离 §0.0-R2 原文并过账(2026-07-25 对抗审查 critical)**:原文写"裁
        # opt.param_groups 全部参数",即**联合**裁剪。联合裁剪把编码器梯度算进同一个总范数,
        # clip_coef = 1/总范数 会**连带压小推理器组的更新** —— 编码器 10.1M 参数的梯度范数
        # 通常压过推理器 1.3M,于是解冻臂的推理器等效学习率被系统性调低,而冻结臂不受此累。
        # 这正是"解冻有害"假阴性的制造机(本轮最贵的失败模式)。改为**逐组各裁到 1.0**:
        # 推理器组与冻结臂逐位同(同集同序同阈),编码器组也确实被裁,原文意图不减。
        gn_by_group = {}
        for pg in opt.param_groups:
            gn_by_group[pg["group_name"]] = float(
                torch.nn.utils.clip_grad_norm_(pg["params"], 1.0))
        gnorm = gn_by_group["reasoner"]
        finite = all(math.isfinite(v) for v in gn_by_group.values())
        if finite:
            gn_n += 1
            for k, v in gn_by_group.items():
                gn_sum[k] += v
                gn_clipped[k] += 1 if v > 1.0 else 0
        n_nonfinite += 0 if finite else 1
        if enc_unfrozen and not enc_grad_seen and finite and enc_gn > 0.0:
            enc_grad_seen = True
            print(f"[ts] **梯度确已到编码器**:step {step} 裁剪前 编码器梯度范数="
                  f"{enc_gn:.6e};{len(eg)}/{len(enc_train_params)} 张量有梯度", flush=True)
        if step in (start, start + 1) or (enc_unfrozen and not enc_grad_seen
                                          and step - start < 5):
            print(f"[ts] step {step} 裁剪前逐组范数(各裁到 1.0)=" + " ".join(
                f"{k}={v:.4g}" for k, v in gn_by_group.items())
                + f";有限={finite};编码器梯度范数={enc_gn:.6e}"
                  f"({len(eg)}/{len(enc_train_params)} 张量有梯度)", flush=True)
        # 第 2 条的真断言:给 GradScaler 若干步回退窗口,窗口内一次都没见到"有限且非零的
        # 编码器梯度"才算没接上(no_grad / requires_grad / 参数组 三者之一断了)。
        if enc_unfrozen and not enc_grad_seen and step - start >= AMP_WARMUP_STEPS:
            raise SystemExit(
                f"[ts] ✋ 解冻臂前 {AMP_WARMUP_STEPS + 1} 步从未出现"
                f"'有限且非零'的编码器梯度 —— 梯度没到编码器,或 AMP 持续溢出;"
                f"继续跑只会产出'解冻无效'的假结论;停")
        scaler.step(opt)
        scaler.update()
        hist.append(diag["total"])
        if (step + 1) % max(1, a.steps // 20) == 0:
            print(f"[ts] (AMP 溢出跳步累计 {n_nonfinite}/{step + 1 - start}) ", end="")
            print(f"[ts] step {step+1}/{a.steps} lr={lr:.2e} loss={hist[-1]:.4f} "
                  f"(s={diag.get('stable', 0):.3f} t={diag.get('type', 0):.3f} "
                  f"c={diag.get('cct', 0):.3f} g={diag.get('group', 0):.3f})", flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            m_in = eval_ts(enc, reasoner, in_exs, normed_by_sid, raw_by_sid, cfg, dev,
                           use_amp, amp_dtype, weights)
            score = m_in["loss"]           # 冒烟期早停锚 = [内]总损失(不设指标死门禁)
            # **审查 major**:早停锚是 [内] 总损失、上线门却是 118 acc —— 只打 loss 就看不出
            # "loss 还在降但 acc 早就平了"或反之。逐次记轨迹,判读时可直接看 acc 是否收敛。
            eval_hist.append({"step": step + 1, "loss": round(score, 5),
                              "acc": round(m_in["accuracy"], 5),
                              "pr_auc": round(m_in["pr_auc"], 5),
                              "recall": round(m_in["unstable_recall"], 5)})
            print(f"[ts] eval step {step+1}: [内] recall={m_in['unstable_recall']:.3f} "
                  f"prec={m_in['unstable_precision']:.3f} ap={m_in['pr_auc']:.3f} "
                  f"acc={m_in['accuracy']:.4f} loss={score:.4f} best={best_val:.4f}", flush=True)
            (out / "ckpt").mkdir(parents=True, exist_ok=True)
            state = {"reasoner_state": reasoner.state_dict(),
                     "optimizer_state": opt.state_dict(),
                     "step": step + 1, "best_val": min(best_val, score), "bad": bad,
                     "rng": rng.bit_generator.state, "encoder": a.encoder,
                     # GradScaler 的 scale 入档(审查 minor):不存则 --resume 后从 65536
                     # 重新起跳、续训首几步再次静默跳步,且 amp 跳步率与不续训座不可比
                     "scaler_state": scaler.state_dict(),
                     "eval_history": list(eval_hist),
                     # **续训累计账**(2026-07-25 审查):不存则 --resume 后 amp 跳步率、
                     # 逐组裁剪账、train_loss_first 都只统计续训段,与不续训座不可比。
                     "acc_n_nonfinite": n_nonfinite, "acc_n_steps": len(hist),
                     "acc_gn_sum": dict(gn_sum), "acc_gn_clipped": dict(gn_clipped),
                     "acc_gn_n": gn_n, "acc_loss_first": (hist[0] if hist else None),
                     "acc_n_resumes": n_resumes,
                     "held_ids": held_by_fam, "weights": weights,   # {family: [sample_id]}
                     # 入训批入档:续训时若粮仓变了(如新批收口)必须显式拒绝,
                     # 否则训练集静默改变而报告仍挂旧口径
                     "batches": sorted(b["batch"] for b in cov["batches"]),
                     "config": {"queries": a.queries, "layers": a.layers, "d": cfg.d,
                                "event_dim": event_dim_total,
                                "side_channel": bool(a.side_channel),
                                "side_features": side_feature_names,
                                "side_mu": (side_mu.tolist() if side_mu is not None else None),
                                "side_sd": (side_sd.tolist() if side_sd is not None else None),
                                "dyn_side": bool(a.dyn_side),
                                "dyn_features": dyn_feature_names,
                                "dyn_mu": (dyn_mu.tolist() if dyn_mu is not None else None),
                                "dyn_sd": (dyn_sd.tolist() if dyn_sd is not None else None),
                                "subtypes": list(TS_SUBTYPES)}}
            # **R2 第 5 条**:解冻臂同存编码器权重与臂名 —— 不存则 best.pt 只有推理器,
            # 终评与全部下游都只能拿 v1.pt 凑数(错配)。**冻结臂不存,旧 run 逐位不变**。
            if enc_unfrozen:
                state["encoder_state"] = {k: v.detach().cpu()
                                          for k, v in enc.state_dict().items()}
                state["unfreeze"] = a.unfreeze_encoder
                state["encoder_lr"] = a.encoder_lr
            if step + 1 == a.eval_every or start == step:
                print(f"[ts] 检查点键清单:{sorted(state.keys())}")
            tmp = out / "ckpt" / f"step_{step+1:06d}.pt.tmp"
            torch.save(state, tmp); tmp.replace(out / "ckpt" / f"step_{step+1:06d}.pt")
            if a.keep_last > 0:      # 剪枝在**新检查点落盘之后**,保证任何时刻都有可续训点
                olds = sorted((out / "ckpt").glob("step_*.pt"))[:-a.keep_last]
                for p_old in olds:
                    try:
                        p_old.unlink()
                    except OSError:
                        pass
            if score < best_val - 1e-6:
                best_val, bad = score, 0
                torch.save(state, out / "ckpt" / "best.pt.tmp")
                (out / "ckpt" / "best.pt.tmp").replace(out / "ckpt" / "best.pt")
            else:
                bad += 1
                if bad >= a.patience:
                    if step + 1 < a.min_steps:      # 早停下限:锚集拿不到就不许停
                        print(f"[ts] (早停条件已满足于 step {step+1},但 <--min-steps "
                              f"{a.min_steps},按下限继续训练)", flush=True)
                    else:
                        print(f"[ts] 早停于 step {step+1}")
                        break

    # AMP 跳步账收口(第 8 个静默失效点):跳步率高 = 实际训练步数 < 名义步数,
    # "解冻没用"可能只是欠训练 —— 响亮打印,不让它只躺在 json 里。
    _rate = n_nonfinite / max(len(hist), 1)
    print(f"\n[ts] AMP 溢出跳步:{n_nonfinite}/{len(hist)}({_rate:.1%})"
          + ("  **⚠ 跳步率 >20%:名义步数与实际更新步数已明显脱节,判读'解冻是否有用'"
             "必须并看本行(可能是欠训练而非解冻无效)**" if _rate > 0.2 else "  (正常)"))
    print("[ts] 逐组裁剪账(各裁到 1.0):" + " ".join(
        f"{k} 平均范数 {(gn_sum[k] / gn_n if gn_n else 0):.3f} / 触发裁剪 "
        f"{(gn_clipped[k] / gn_n if gn_n else 0):.1%}" for k in gn_sum
        if k == "reasoner" or enc_unfrozen), flush=True)

    # ---- 终评(8.6 双报,自然分布口径;平衡集未冻结如实标注)----
    bck = torch.load(out / "ckpt" / "best.pt", map_location=dev, weights_only=False)
    if (bck.get("encoder") != a.encoder
            or json.dumps(bck.get("held_ids"), sort_keys=True) != held_key):
        raise SystemExit("[ts] best.pt 与本次配置(编码器/划分)不符,请换 --out")
    # **R2 第 4 条**:旧写法只 load 推理器,enc 仍是**末步**编码器 —— 报告就成了
    # "best 步推理器 + 末步编码器"的错配组合(早停 patience 8×200 可差 1600 步),
    # 而 encoder 字段照写 "v1"、指标照常出数(静默)。凭据 = 下面这条臂名断言。
    # 用 raise 不用 assert:assert 在 python -O 下被整条剥掉,凭据会静默消失
    if str(bck.get("unfreeze", "none")) != a.unfreeze_encoder:
        raise SystemExit(
            f"[ts] ✋ best.pt 解冻臂 {bck.get('unfreeze', 'none')!r} ≠ 本次 "
            f"{a.unfreeze_encoder!r}(权重与配方不同源,终评作废)")
    if enc_unfrozen:
        enc.load_state_dict(bck["encoder_state"])
        print(f"[ts] 终评编码器 = best.pt 的 encoder_state(step {bck['step']},"
              f"臂 {bck['unfreeze']});推理器同源 —— 无 best/末步错配")
        _dr = encoder_drift(enc, ck["model_state"])
        print(f"[ts] **编码器相对 v1 漂移** rel_L2={_dr['rel_l2_overall']:.6f}"
              f"({_dr['n_tensors_moved']} 个张量动过);漂移最大:"
              + ", ".join(f"{t['name']}={t['rel_l2']:.4f}"
                          for t in _dr["top_drifted_tensors"][:4]))
        if _dr["rel_l2_overall"] < 1e-4:
            print("[ts] ⚠ 漂移 <1e-4:**编码器几乎没动** —— 此时结论只能是'这个配方下解冻"
                  "没发生',**不得**表述为'解冻对判稳结构性无效'")
    reasoner.load_state_dict(bck["reasoner_state"])
    m_in = eval_ts(enc, reasoner, in_exs, normed_by_sid, raw_by_sid, cfg, dev,
                   use_amp, amp_dtype, weights)
    m_out = eval_ts(enc, reasoner, out_exs, normed_by_sid, raw_by_sid, cfg, dev,
                    use_amp, amp_dtype, weights)
    # **[内] 逐训练家族分报**(拓扑维度 2 起必需):合并口径会把两族的判别力平均掉 ——
    # "预训练档案在同拓扑内值钱"这一结论必须**在每个训练家族上各自成立**才算数,
    # 且 118 与 cepri36 的基率/物理天差地别(实测 118 ~77% vs cepri36 ~48%)。
    m_in_by_fam = {}
    for fam in held_by_fam:
        exs_f = [x for x in in_exs if x["family"] == fam]
        if exs_f:
            m_in_by_fam[fam] = eval_ts(enc, reasoner, exs_f, normed_by_sid, raw_by_sid,
                                       cfg, dev, use_amp, amp_dtype, weights)
    # **[外] 逐族分报**(2026-07-17 双外族实验起必需):合并口径无法回答"哪张考卷特殊";
    # 单外族时代本表恰有一项且与 metrics_out 相同,零行为变化。
    m_out_by_fam = {}
    for fam in out_fams:
        exs_f = [x for x in out_exs if x["family"] == fam]
        if exs_f:
            m_out_by_fam[fam] = eval_ts(enc, reasoner, exs_f, normed_by_sid, raw_by_sid,
                                        cfg, dev, use_amp, amp_dtype, weights)
    # **event-only 空基线**(验收的真正零点;零图信息,只读事件参数)——见 event_only_null
    null_arms = {"外": event_only_null(train_exs, out_exs),
                 "内·合并": event_only_null(train_exs, in_exs)}
    for fam, exs_f in ((f, [x for x in in_exs if x["family"] == f]) for f in held_by_fam):
        if exs_f:
            null_arms[f"内·{fam}"] = event_only_null(train_exs, exs_f)
    for fam in out_fams:
        exs_f = [x for x in out_exs if x["family"] == fam]
        if exs_f:
            null_arms[f"外·{fam}"] = event_only_null(train_exs, exs_f)
    na = null_arms.get("外") or {}
    if na:
        print(f"\n[ts][空基线·event-only] [外] n={na['n']}:判别力 {na['skill_over_const']:.3f}× / "
              f"PR-AUC/base {na['pr_auc_over_base']:.3f}×(预测正类率 {na['pred_pos_rate']:.3f};"
              f"{na['n_buckets']} 个 (K,dur) 桶,回退 {na['n_fallback_rows']} 行)")
        print("  ↳ 该基线**完全看不见电网**:模型必须打赢它,'跨拓扑能力'才算数"
              "(打赢恒定预测 = 1.0 远远不够)")

    # ---- 动态参数侧通道:裸值基线行 + 逐特征零点行(逐族;判读依据,见开场呈批 §2)----
    dyn_naked = None
    if a.dyn_side:
        y_tr = np.array([float(x["y_unstable"]) for x in train_exs])
        y_in = np.array([float(x["y_unstable"]) for x in in_exs])
        y_out = np.array([float(x["y_unstable"]) for x in out_exs])
        evals = {"内·合并": (dyn_eval_mats["in"], y_in),
                 "外·合并": (dyn_eval_mats["out"], y_out)}
        for fam in held_by_fam:
            idx = [i for i, x in enumerate(in_exs) if x["family"] == fam]
            if idx:
                evals[f"内·{fam}"] = (dyn_eval_mats["in"][idx], y_in[idx])
        for fam in out_fams:
            idx = [i for i, x in enumerate(out_exs) if x["family"] == fam]
            if idx:
                evals[f"外·{fam}"] = (dyn_eval_mats["out"][idx], y_out[idx])
        dyn_naked = dyn_naked_rows(dyn_eval_mats["train"], y_tr, evals, dyn_feature_names)
        print("\n[ts][动态参数裸值基线/零点行](模型增益须超过下列行才算理解而非读数)")
        for nm, row in dyn_naked.items():
            print(f"  {nm}: n={row['n']} LR裸值 {row['lr_ap_over_base']}x / "
                  f"单特征零点最大 {row['zero_max']}x")

    reports = [(f"内·{fam}(留出方式,同拓扑)", m) for fam, m in m_in_by_fam.items()]
    reports += [("内·合并(全训练家族留出)", m_in)]
    reports += [(f"外·{fam}(整族)", m) for fam, m in m_out_by_fam.items()]
    reports += [(f"外·合并({'+'.join(out_fams)})", m_out)]
    for name, m in reports:
        print(f"\n[ts][{name}] n={m['n']}(失稳 {m['n_unstable']};"
              f"基率 {m['base_rate_unstable']:.3f})")
        print(f"  失稳召回(主)={m['unstable_recall']:.4f} 精确率={m['unstable_precision']:.4f} "
              f"PR-AUC={m['pr_auc']:.4f} 准确率(参考)={m['accuracy']:.4f}")
        print(f"  ↳ 恒定预测基线(全判失稳):召回 1.000 / 精确率 {m['base_rate_unstable']:.3f}"
              f";**判别力 = 精确率/基率 = {m['skill_over_const']:.2f}×**(≤1 即无优势)")
        for t in TS_TYPE_FROM_SUB:
            if f"type_{t}_recall" in m:
                br = m[f"type_{t}_base_rate"]
                flag = "  ⚠基率=1.0,召回无判别力" if br >= 0.999 else ""
                print(f"  类型 {t}(聚合): recall={m[f'type_{t}_recall']:.3f} "
                      f"prec={m[f'type_{t}_precision']:.3f} "
                      f"[基率 {br:.3f} / 预测正类率 {m[f'type_{t}_pred_pos_rate']:.3f}]{flag}")
        for s in TS_SUBTYPES:
            if f"sub_{s}_recall" in m:
                br = m[f"sub_{s}_base_rate"]
                flag = ("  ⚠基率=1.0,召回无判别力" if br >= 0.999
                        else "  [评估集无正例]" if m[f"sub_{s}_n"] == 0 else "")
                print(f"    子类 {s}(n={m[f'sub_{s}_n']}): recall={m[f'sub_{s}_recall']:.3f} "
                      f"prec={m[f'sub_{s}_precision']:.3f} [基率 {br:.3f}]{flag}")
        print(f"  CCT MAE(非删失,n={m['cct_n_noncensored']})="
              f"{m['cct_mae_noncensored']}  机群 microF1={m['group_micro_f1']}"
              f"(行 {m['group_rows']})")
    # **encoder 字段是假凭据**:它写的是 --encoder 的值,解冻后照写 "v1"。判断某个 run
    # 该配哪个编码器,唯一真凭据 = unfreeze_encoder(与检查点 unfreeze 键成对校验)。
    rep = {"encoder": a.encoder, "unfreeze_encoder": a.unfreeze_encoder,
           "encoder_lr": (a.encoder_lr if enc_unfrozen else None),
           "encoder_trainable_params": sum(p.numel() for _nm, p in enc_train_params),
           "encoder_total_params": sum(p.numel() for p in enc.parameters()),
           "encoder_trainable_tensors": [nm for nm, _p in enc_train_params],
           "encoder_train_mode_sensitive": train_mode_sensitive(enc),
           # AMP 溢出跳步账(第 8 个静默失效点):跳步率高 = 实际训练步数 < 名义步数,
           # "解冻没用"可能只是欠训练。判读解冻结论时必须并看本行。
           # 跳步率分母 = **累计**训练步数(含续训前),否则续训座与不续训座不可比
           "amp_nonfinite_steps": n_nonfinite,
           "amp_nonfinite_rate": round(n_nonfinite / max(len(hist) + n_steps_prev, 1), 4),
           "n_resumes": n_resumes,
           "resume_note": ("本座**续训过 %d 次**(驱动崩/中断);amp/裁剪账已累计,"
                           "train_loss_first 取首段值" % n_resumes) if n_resumes else "未续训",
           "encoder_grad_reached": bool(enc_grad_seen) if enc_unfrozen else None,
           # 逐组裁剪账(偏离 §0.0-R2 原文"联合裁剪"的过账凭据):推理器组各裁到 1.0,
           # 与冻结臂同阈 → 解冻臂的推理器**不被编码器梯度连累**(联合裁剪会连累)。
           "grad_clip_mode": "per_param_group@1.0",
           "grad_norm_mean_by_group": {k: (round(v / gn_n, 4) if gn_n else None)
                                       for k, v in gn_sum.items()},
           "grad_clip_active_rate_by_group": {k: (round(v / gn_n, 4) if gn_n else None)
                                              for k, v in gn_clipped.items()},
           "v1_fingerprint": v1_fp, "normalizer_fingerprint": nm_fp,
           # 终评所用编码器的**权重指纹**:下游 build_encoder_for_run 装完复算比对,
           # 对不上即停 —— "编码器错配"从此机制上不可能(不再靠约定)。
           "encoder_state_fingerprint": encoder_fingerprint(enc),
           # **"解冻到底动了多少"**:没有这行就分不清"解冻没发生"与"解冻无效"两种结局。
           "encoder_drift_from_v1": encoder_drift(enc, ck["model_state"]),
           # **完整命令行入档**(对抗审查 major:parity 的 8 项对 --batch/--lr/--patience
           # 全盲,报告又从不落 argv → 配方写错也能 8/8 全绿)。落 argv 后配方差异可逐字核。
           "argv": sys.argv[1:],
           "args": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                    for k, v in sorted(vars(a).items())},
           # AMP 跳步是否已到"数字存疑"的程度(记账 + 明确判词,不只留个数)
           "amp_verdict": ("跳步率 >20%,名义步数与实际更新脱节,'解冻无效'可能是欠训练"
                           if n_nonfinite / max(len(hist), 1) > 0.2 else "正常"),
           "steps": a.steps, "best_step": bck["step"],
           "side_channel": bool(a.side_channel), "side_features": side_feature_names,
           "dyn_side": bool(a.dyn_side), "dyn_features": dyn_feature_names,
           "dyn_naked_baseline": dyn_naked,
           "reasoner_params": reasoner.num_params(), "weights": weights,
           "split": {"split_mode": a.split,       # mode_holdout(生产)/ random(W4.2 诊断)
                     "held_ids": held_by_fam,
                     "train_mode_frac": a.train_mode_frac,          # 1.0 = 未消融
                     "kept_train_modes": kept_modes_by_fam,          # None = 未消融;存证供复算
                     "n_train_modes_by_family": {
                         f: len({x["sample_id"] for x in train_exs if x["family"] == f})
                         for f in held_by_fam},
                     "train_families": train_fams_actual,   # 实测,非声称
                     "topology_dim": topo_dim,
                     "out_families": list(out_fams),
                     "n_train": len(train_exs), "n_in": len(in_exs), "n_out": len(out_exs),
                     "n_train_by_family": {f: sum(1 for x in train_exs if x["family"] == f)
                                           for f in held_by_fam},
                     "note": "[内]=各训练家族的留出方式(D14 单拓扑豁免,同拓扑,逐族分报);"
                             "[外]=out_families 整族(永不入训;ieee39 恒在列);"
                             "自然分布口径,平衡集未冻结(8.6)"},
           "coverage": cov, "metrics_in": m_in, "metrics_in_by_family": m_in_by_fam,
           "metrics_out": m_out, "metrics_out_by_family": m_out_by_fam,
           # 验收的真正零点(零图信息);report 侧据此判"是否打赢空基线"
           "null_event_only": null_arms,
           "train_loss_first_last": [
               (loss_first_prev if loss_first_prev is not None
                else (hist[0] if hist else None)),
               hist[-1] if hist else None],
           # 评估轨迹(step/loss/acc/pr_auc/recall):判"是否收敛"还是"被步数预算截断"的凭据。
           # 末条 bad=0 且 loss 仍在降 = 预算截断,此时任何"某改动无用"的结论只在该预算下成立。
           "eval_history": eval_hist,
           # **收敛判据必须同时看"跑满没有"**(2026-07-25 审查实测:旧写法只比末两次 loss,
           # 早停座 r3budget_s1 被误报成"截断")。**触发早停 = 收敛;跑满 steps = 截断。**
           "ran_full_steps": bool(eval_hist and eval_hist[-1]["step"] >= a.steps),
           "early_stopped": bool(eval_hist and eval_hist[-1]["step"] < a.steps),
           # **早停锚 vs 上线门的脱节凭据**(2026-07-27 用户裁"改 A+C" 的 C;纯观测,
           # 不改任何选择语义)。`budget_note` 只判"跑满被截断",判不出"早停锚抖动导致
           # 提前停":10 座 R6 全部触发早停 → 全部被判"预算充足",而其中 3 座的 acc
           # 在末次评估仍是全程最高(= 仍在爬就被停了)。此处直接把凭据落进报告。
           "best_acc_step": (max(eval_hist, key=lambda e: e["acc"])["step"]
                             if eval_hist else None),
           "acc_at_best_loss_step": next(
               (e["acc"] for e in eval_hist if e["step"] == bck["step"]), None),
           "acc_max": (max(e["acc"] for e in eval_hist) if eval_hist else None),
           # True = 末次评估恰是 acc 最高点 → **仍在爬就被停/截断了**,判读须并看
           "acc_still_rising": bool(
               eval_hist and eval_hist[-1]["acc"] >= max(e["acc"] for e in eval_hist)),
           "min_steps": a.min_steps,
           "budget_note": (
               "**跑满 steps 且末次评估仍在改进 = 被步数预算截断,非收敛**"
               if (eval_hist and eval_hist[-1]["step"] >= a.steps and len(eval_hist) >= 2
                   and eval_hist[-1]["loss"] < eval_hist[-2]["loss"])
               else ("触发早停 = 已收敛(预算充足)" if (eval_hist
                     and eval_hist[-1]["step"] < a.steps)
                     else "跑满 steps 但末次未再改进(预算大致够)"))}
    (out / "ts_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"[ts] 报告 {out / 'ts_report.json'}")
    # **训练预算门(R3,2026-07-25)**:跑满 --steps 且末次评估仍在改进 = 被预算截断,
    # 不是收敛。此时任何"某改动无用/指标卡住/逼近信息上限"的结论**都只在该预算下成立**。
    # 判稳线为此白做过一整轮(R2 解冻)与两个便宜杆,故此处必须响亮,不能只留在 json 里。
    if bck["step"] >= a.steps and len(eval_hist) >= 2 \
            and eval_hist[-1]["loss"] < eval_hist[-2]["loss"]:
        print("\n" + "=" * 78)
        print("[ts] ⚠⚠ **训练预算不足**:跑满 --steps={} 且末次评估仍在改进"
              "(loss {:.4f} < 前次 {:.4f}、best_step={} = 末步)。"
              .format(a.steps, eval_hist[-1]["loss"], eval_hist[-2]["loss"], bck["step"]))
        print("[ts]    本 run **未收敛**;据此得出的任何'某改动无用 / 指标卡住 / 逼近信息"
              "上限'结论,**只在这个预算下成立**,不得推广。")
        print("[ts]    先加大 --steps 至早停触发再下结论(R3 实测真收敛点 11~13k 步)。")
        print("=" * 78, flush=True)
    else:
        print(f"[ts] 训练预算门:过(best_step={bck['step']} / steps={a.steps};"
              f"{rep['budget_note']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
