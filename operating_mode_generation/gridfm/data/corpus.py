"""G4 多家族训练语料(方案 §6.7/8.4;交接-G线 §2-3 接口改造 F4)。

**单一出处**:G4 训练语料的组成(families/train 全部可解样本 + lib118_diverse 库内
训练份)只在本模块定义,g_latent_stats / g15_train_readout / 扩散训练一律经此获取,
杜绝各脚本自拼语料造成的组成漂移(D16 纪律同型)。

**铁律以断言强制**(而非各脚本自觉):留出家族(families/val)与留出规模档
(families/transfer)**永不入训**——加载时读磁盘 manifest 交叉断言零交集;
lib118_diverse 的库内留出(load_factor 排序 test_every 取 1)与 v0.x 逐位同款,
留出不入。语料指纹(BLAKE2b-128,manifest 驱动)随下游产物固化,升版可审计。
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..schema.fields import NODE_TABLE_V03
from ..schema.sample import GridSample
from .dataset import family_files, library_files


def load_lib_lf_sorted(lib_dir: Path) -> Tuple[List[GridSample], Dict[str, float]]:
    """可解样本按 load_factor 排序加载(g1/g15/诊断 A 同款前处理)。"""
    lib_dir = Path(lib_dir)
    lf_by_sid, solvable = {}, {}
    with open(lib_dir / "manifest.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lf_by_sid[r["sample_id"]] = float(r["load_factor"])
            solvable[r["sample_id"]] = r["solvable"] == "1"
    samples = [GridSample.load_npz(p) for p in library_files(lib_dir)]
    samples = [s for s in samples if solvable.get(s.sample_id)]
    samples.sort(key=lambda s: lf_by_sid[s.sample_id])
    return samples, lf_by_sid


def single_lib_split(lib_dir: Path, test_every: int = 5
                     ) -> Tuple[List[GridSample], List[GridSample]]:
    """库内留出切法(load_factor 排序每 test_every 取 1 = 留出),g1/g15 逐位同款。"""
    samples, _ = load_lib_lf_sorted(lib_dir)
    test_idx = set(range(0, len(samples), test_every))
    pool = [s for i, s in enumerate(samples) if i not in test_idx]
    test = [s for i, s in enumerate(samples) if i in test_idx]
    return pool, test


def _split_family_ids(fam_root: Path, split: str) -> set:
    """磁盘 manifest 的 family_id 列(独立事实源,非目录名)。"""
    ids = set()
    d = fam_root / split
    if d.exists():
        for mf in sorted(d.glob("*/manifest.csv")):
            with open(mf, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ids.add(row["family_id"])
    return ids


def _physics_hash(gs: GridSample) -> str:
    """身份无关的物理内容哈希:只哈希电网物理数组(node/branch 特征 + 缺失/掩码 +
    edge_index),**排除** sample_id/family_id 与 identity 派生的 node_random/branch_random。
    故"整份重贴新家族名再重序列化"不改物理 → 哈希不变,可被跨集比对拦下(审查:89)。"""
    hh = hashlib.blake2b(digest_size=16)
    for arr in (gs.node_feat, gs.branch_feat, gs.edge_index,
                gs.node_missing, gs.node_masked, gs.branch_missing, gs.branch_masked):
        a = np.ascontiguousarray(arr)
        hh.update(str(a.dtype).encode()); hh.update(str(a.shape).encode())
        hh.update(a.tobytes())
    return hh.hexdigest()


def _holdout_physics_hashes(root: Path) -> set:
    out = set()
    for sp in ("val", "transfer"):
        for p in family_files(root, sp):
            out.add(_physics_hash(GridSample.load_npz(p)))
    return out


def _train_families(root: Path, h, require_holdout: bool, check_content_leak: bool,
                    full_schema: bool = False):
    """families/train 装载 + 四道防泄漏守卫(g4 与 v2 共用**单一出处**,禁分叉)。

    返回 (samples, per_family, val_fams, xfer_fams);哈希器 h 按原序被更新
    (留出 manifest → 逐族名+manifest),故 g4 指纹逐位不变。
    """
    fam_root = root / "families"
    assert fam_root.exists(), f"{fam_root} 不存在:先跑 scripts/gen_families.py 生产多家族库"
    val_fams = _split_family_ids(fam_root, "val")
    xfer_fams = _split_family_ids(fam_root, "transfer")
    if require_holdout:
        assert val_fams and xfer_fams, (
            f"留出集为空(val {len(val_fams)} / transfer {len(xfer_fams)} 家族)——"
            f"families/val 或 families/transfer 缺失/未同步?留出排除断言会空转。"
            f"确属无留出场景请显式传 require_holdout=False")
    holdout_hashes = _holdout_physics_hashes(root) if check_content_leak else set()

    samples: List[GridSample] = []
    per_family: Dict[str, int] = {}
    for sp in ("val", "transfer"):
        sp_dir = fam_root / sp
        if sp_dir.exists():
            for mf in sorted(sp_dir.glob("*/manifest.csv")):
                h.update(f"holdout:{sp}:{mf.parent.name}".encode("utf-8"))
                h.update(mf.read_bytes())
    train_root = fam_root / "train"
    for fam_dir in sorted(d for d in train_root.iterdir() if d.is_dir()):
        mf = fam_dir / "manifest.csv"
        if not mf.exists():          # 半成品家族(生产中断)不入语料,响亮报出
            raise AssertionError(f"{fam_dir} 缺 manifest.csv:生产未收口,先补产再训")
        h.update(fam_dir.name.encode("utf-8"))
        h.update(mf.read_bytes())
        solvable = {}
        with open(mf, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                solvable[row["sample_id"]] = row["solvable"] == "1"
                fam = row["family_id"]
                assert fam not in val_fams and fam not in xfer_fams, \
                    f"泄漏:train 目录下家族 {fam} 出现在留出家族/留出规模档集合中"
        for p in library_files(fam_dir):
            s = GridSample.load_npz(p, full_schema=full_schema)
            if check_content_leak:
                # 内容级比对一律在 v0 视图上做(留出面为 25 列冻结件,两侧同尺)
                probe = s if not full_schema else GridSample.load_npz(p)
                assert _physics_hash(probe) not in holdout_hashes, \
                    f"内容级泄漏:train npz {p.name} 物理与某留出样本完全相同(重贴家族名?)"
            if not solvable.get(s.sample_id):
                continue
            assert s.family_id not in val_fams and s.family_id not in xfer_fams, \
                f"泄漏:样本 {s.sample_id} 的 family_id={s.family_id} 属留出集"
            samples.append(s)
            per_family[fam_dir.name] = per_family.get(fam_dir.name, 0) + 1
    return samples, per_family, val_fams, xfer_fams


def v2_train_corpus(root: Path, split_json: Path,
                    libs: Sequence[str] = ("lib118", "lib_cepri36"),
                    require_holdout: bool = True,
                    check_content_leak: bool = True) -> Tuple[List[GridSample], dict]:
    """**编码器 v2 预训练语料**(43 列 v0.3;M2/M2-b 之后的唯一出处)。

    = families/train(基态嫁接后 43 列)+ **lib118_diverse 的 158 训练份**
      + 标准算例库(lib118 / lib_cepri36)。万节点方式库(B2 产出)由调用方另行拼入。

    **与 g4_train_corpus 的关键差别(地雷)**:g4 对 diverse 用 `test_every=5` 切 240
    入训,与 v2 的 158/144 名单**是两套划分**——实测交集 = v2 冻结面 144 个中有 115 个
    (含 evalonly 30 中的 26 个)落在 g4 训练份里。v2 **必须**按 `diverse_split.json`
    取 158,**沿用 g4 切法即 OOD 主门当场作废**。本函数以断言强制:
      ①样本 id 与冻结面零交集;②evalonly30 零交集;③全部样本须 43 列(半截 schema
      不得静默入训);④families 留出家族排除(复用 g4 同一套四道守卫)。
    """
    root = Path(root)
    d = json.loads(Path(split_json).read_text(encoding="utf-8"))
    train_ids = list(dict.fromkeys(d["v2_train"]))
    frozen = set(d["frozen_eval_face"])
    evalonly = set(d.get("frozen_face_contains_evalonly30", []))
    assert not (set(train_ids) & frozen), "名单自相矛盾:训练份与冻结面有交集"
    assert not (set(train_ids) & evalonly), "evalonly30 出现在训练份——绝不入训红线被打穿"

    h = hashlib.blake2b(digest_size=16)
    samples, per_family, val_fams, xfer_fams = _train_families(
        root, h, require_holdout, check_content_leak, full_schema=True)

    # diverse 训练份:名单驱动,逐个显式加载(不走 test_every)
    div_dir = root / "lib118_diverse"
    h.update(b"diverse_split")
    h.update(Path(split_json).read_bytes())
    n_div = 0
    for sid in train_ids:
        p = div_dir / f"{sid}.npz"
        if not p.exists():
            raise AssertionError(f"名单方式缺失:{p}")
        s = GridSample.load_npz(p, full_schema=True)
        assert s.sample_id not in frozen, f"冻结面方式 {s.sample_id} 混入训练份"
        samples.append(s)
        n_div += 1

    per_lib: Dict[str, int] = {}
    for lib_name in libs:
        lib_dir = root / lib_name
        h.update(lib_name.encode("utf-8"))
        h.update((lib_dir / "manifest.csv").read_bytes())
        for p in library_files(lib_dir):
            samples.append(GridSample.load_npz(p, full_schema=True))
            per_lib[lib_name] = per_lib.get(lib_name, 0) + 1

    widths = {s.node_feat.shape[1] for s in samples}
    assert widths == {NODE_TABLE_V03.total_dim}, (
        f"v2 语料列宽不齐 {sorted(widths)}(应全为 {NODE_TABLE_V03.total_dim})——"
        f"半截 schema 不得入训;检查 families 嫁接与各库重装是否完成")
    got = {s.sample_id for s in samples}
    assert not (got & frozen), f"冻结面泄漏:{sorted(got & frozen)[:5]}"

    prov = {
        "corpus": "v2_train", "schema": "v0.3",
        "n_samples": len(samples), "n_synth_families": len(per_family),
        "per_family": per_family, "per_lib": per_lib,
        "diverse": {"split_json": str(split_json), "n_train": n_div,
                    "n_frozen_excluded": len(frozen), "n_evalonly_excluded": len(evalonly)},
        "excluded_splits": {"val_families": len(val_fams),
                            "transfer_families": len(xfer_fams)},
        "content_leak_checked": check_content_leak,
        "fingerprint": h.hexdigest(),
    }
    return samples, prov


def load_wjd_modes(libs: Sequence[Path], holdout_json: Path,
                   require_holdout: bool = True) -> Tuple[List[GridSample], dict]:
    """万节点方式库装载(v2 训练侧唯一入口;第十三任 J10"零交集断言载体"落地)。

    只装 solvable==1 的收敛方式;**留出面(wjd_holdout.json)绝不入训**:名单排除 +
    装载后零交集断言 + 留出文件哈希复验(字节被改即响亮报错)。
    require_holdout=True(默认)时留出件必须存在——防"留出未冻结先入训"的窗口期
    (确属留出冻结前的开发场景须显式 False,产物不得用于正式训练)。
    """
    import csv as _csv
    import hashlib as _hl
    holdout_ids: set = set()
    hj = Path(holdout_json)
    if hj.exists():
        hd = json.loads(hj.read_text(encoding="utf-8"))
        holdout_ids = set(hd.get("sample_ids", []))
        for sid, want in hd.get("blake2b", {}).items():
            for lib in libs:
                p = Path(lib) / f"{sid}.npz"
                if p.exists():
                    got = _hl.blake2b(p.read_bytes(), digest_size=16).hexdigest()
                    assert got == want, f"留出面 {sid} 字节被改({got} ≠ {want})"
                    break
    elif require_holdout:
        raise AssertionError(
            f"万节点留出面未冻结({hj} 不存在)——先跑 scripts/wjd_holdout_freeze.py;"
            f"开发场景显式 require_holdout=False")
    samples: List[GridSample] = []
    per_lib: Dict[str, int] = {}
    for lib in libs:
        lib = Path(lib)
        solvable = {}
        with open(lib / "manifest.csv", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                solvable[row["sample_id"]] = row["solvable"] == "1"
        for p in sorted(lib.glob("*.npz")):
            if not solvable.get(p.stem) or p.stem in holdout_ids:
                continue
            samples.append(GridSample.load_npz(p, full_schema=True))
            per_lib[lib.name] = per_lib.get(lib.name, 0) + 1
    got = {s.sample_id for s in samples}
    assert not (got & holdout_ids), f"留出面泄漏:{sorted(got & holdout_ids)[:5]}"
    w = {s.node_feat.shape[1] for s in samples}
    assert not samples or w == {NODE_TABLE_V03.total_dim}, f"列宽不齐 {sorted(w)}"
    return samples, {"per_lib": per_lib, "n": len(samples),
                     "holdout_excluded": len(holdout_ids)}


def v2_full_corpus(root: Path, runs_dir: Path,
                   wjd_libs: Sequence[str] = ("lib_wjd_base", "lib_wjd_modes",
                                              "lib_wjd_n1"),
                   transform: bool = True):
    """**v2 冒烟/训练唯一装载入口**(A5 前置件,2026-08-09)。

    = v2_train_corpus(7,993:families 嫁接 + diverse 158 + lib118 + cepri36)
      + load_wjd_modes(万节点基底 + 甲乙档入训方式,留出面排除;基底库件
        `lib_wjd_base` 由 scripts/freeze_wjd_base.py 与 normalizer 拟合基同参装配
        ——总账 375 = 甲 300 + 乙 144 − 留出 70 + 基底 1,账实一致)
      + NormalizerV2 定稿件施加(runs/encoder_v2/normalizer_v2.npz,**只读**;
        transform=True 原地归一化——npz 存原始物理值,训练前必须过此步)。

    冻结面纪律全部由下层函数断言强制(diverse 冻结面/evalonly30/万节点留出面
    零交集 + 留出件哈希复验);本入口再补:①合并后列宽统一断言;②归一化件字节
    指纹入 prov(报告引用 = 定稿件身份证);③两库样本 id 无碰撞断言。
    返回 (samples, norm, prov)。
    """
    root, runs_dir = Path(root), Path(runs_dir)
    split_json = runs_dir / "diverse_split.json"
    holdout_json = runs_dir / "wjd_holdout.json"
    norm_path = runs_dir / "normalizer_v2.npz"
    assert norm_path.exists(), f"normalizer v2 定稿件不存在:{norm_path}"

    from .normalize_v2 import NormalizerV2
    samples, prov_core = v2_train_corpus(root, split_json)
    wjd_paths = [root / lib for lib in wjd_libs if (root / lib).exists()]
    assert len(wjd_paths) == len(wjd_libs), (
        f"万节点库缺失:{[str(root / l) for l in wjd_libs if not (root / l).exists()]}")
    wjd_samples, prov_wjd = load_wjd_modes(wjd_paths, holdout_json)

    core_ids = {s.sample_id for s in samples}
    dup = core_ids & {s.sample_id for s in wjd_samples}
    assert not dup, f"样本 id 碰撞(核心语料 vs 万节点库):{sorted(dup)[:5]}"
    samples = samples + wjd_samples

    widths = {s.node_feat.shape[1] for s in samples}
    assert widths == {NODE_TABLE_V03.total_dim}, f"合并后列宽不齐 {sorted(widths)}"

    norm = NormalizerV2.load(norm_path)
    if transform:
        for s in samples:
            norm.transform(s)
    norm_fp = hashlib.blake2b(norm_path.read_bytes(), digest_size=16).hexdigest()
    prov = {
        "corpus": "v2_full", "schema": "v0.3",
        "n_total": len(samples), "n_core": prov_core["n_samples"],
        "n_wjd": prov_wjd["n"], "core": prov_core, "wjd": prov_wjd,
        "normalizer": {"path": str(norm_path), "blake2b": norm_fp,
                       "applied": bool(transform)},
    }
    return samples, norm, prov


def g4_train_corpus(root: Path, lib_name: str = "lib118_diverse",
                    test_every: int = 5, require_holdout: bool = True,
                    check_content_leak: bool = True) -> Tuple[List[GridSample], dict]:
    """G4 训练语料 = families/train 全部可解样本 + lib118_diverse 训练份。

    返回 (samples, provenance)。顺序确定(家族名排序 × manifest 序,后接 lib 训练份
    lf 序)——下游 val_every 切分可复现。

    **防泄漏(审查加固,2026-07-19):四道**——
    ① family_id 集合零交集(train ∩ (val∪transfer)=∅,manifest 列 + npz 字段双检);
    ② `require_holdout`(默认 True):val/transfer 目录必须存在且非空——否则留出集为空、
       家族名断言沦为空转却仍报"已强制"(partial-sync/机器B只回传 train 场景,审查:47);
    ③ `check_content_leak`(默认 True):**内容级**去重——留出 npz 的字节哈希集与 train
       npz 逐一比对,拦"整份重贴新家族名"绕过字符串断言的泄漏(审查:89;字节不变哈希不变);
    ④ 指纹并入 val/transfer manifest 字节,留出边界漂移在指纹上可见(审查:102/83)。

    2026-08-08:四道守卫的实现下沉至 `_train_families`(与 v2 语料**单一出处**,禁分叉);
    哈希更新次序不变 → 指纹 `c5333f2c…` 逐位复现(单测锁定)。
    """
    root = Path(root)
    h = hashlib.blake2b(digest_size=16)
    samples, per_family, val_fams, xfer_fams = _train_families(
        root, h, require_holdout, check_content_leak, full_schema=False)

    lib_dir = root / lib_name
    # [外协包 F,2026-08-25] 缺库容忍:lib118_diverse **未随 F 包发**(gridfm-data 下只有
    # families/;SHA256SUMS 零提及)。原实现无条件读它 → FileNotFoundError,g4 语料在本包
    # 内完全不可用。缺库时如实跳过并在指纹里留痕(而非静默当作空):后果是指纹必然不等于
    # 历史 c5333f2c…,历史那 240 份 diverse 训练份**无法复现**——这是事实,交付报告须标注。
    lib_present = (lib_dir / "manifest.csv").is_file()
    if lib_present:
        pool, test = single_lib_split(lib_dir, test_every)
        samples.extend(pool)
        h.update(lib_name.encode("utf-8"))
        h.update((lib_dir / "manifest.csv").read_bytes())
        h.update(f"test_every={test_every}".encode("utf-8"))
    else:
        pool, test = [], []
        h.update(f"lib_absent:{lib_name}".encode("utf-8"))

    prov = {
        "corpus": "g4_train",
        "n_samples": len(samples),
        "n_synth_families": len(per_family),
        "per_family": per_family,
        "lib_pool": {"lib": lib_name, "present": lib_present, "n_train": len(pool),
                     "n_holdout_excluded": len(test), "test_every": test_every},
        "excluded_splits": {"val_families": len(val_fams),
                            "transfer_families": len(xfer_fams)},
        "content_leak_checked": check_content_leak,
        "fingerprint": h.hexdigest(),
    }
    return samples, prov
