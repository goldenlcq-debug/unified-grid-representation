"""G4 训练语料模块单测(F4 接口改造;离线,不跑 PSASP)。

重点:**留出家族/留出规模档永不入训**的断言必须真的会咬人(防泄漏是铁律,
不能只靠生产侧 audit——语料加载侧再设一道,防 resume 合并/人工挪目录)。
"""
from __future__ import annotations

import pytest

from gridfm.data.corpus import g4_train_corpus, single_lib_split
from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.schema.assemble import build_sample


def _mk_lib(root, rel, fam, n, lf0=0.95, solvable_flags=None, case=None, cat=None):
    """构造一个带 manifest 的家族库目录(真实 GridSample,118 golden 骨架)。

    每样本对 node_feat 施加**确定性逐样本扰动**(keyed by fam+i),模拟真实库里各方式
    物理不同——否则 golden 骨架产出的样本物理逐位相同,会被内容级泄漏检查误判碰撞。"""
    import hashlib

    import numpy as np
    lib = root / rel
    lib.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        sid = f"{fam}_{i:05d}"
        s = build_sample(case, cat, sample_id=sid, family_id=fam)
        seed = int.from_bytes(hashlib.blake2b(sid.encode(), digest_size=4).digest(), "big")
        rng = np.random.default_rng(seed)
        s.node_feat = s.node_feat + rng.standard_normal(s.node_feat.shape).astype(
            s.node_feat.dtype) * 1e-3          # 逐样本可辨、幅度微小
        s.save_npz(lib / f"{sid}.npz")
        sv = 1 if solvable_flags is None else int(solvable_flags[i])
        rows.append(f"{sid},{fam},{sv},{lf0 + 0.01 * i}")
    (lib / "manifest.csv").write_text(
        "sample_id,family_id,solvable,load_factor\n" + "\n".join(rows) + "\n",
        encoding="utf-8")
    return lib


@pytest.fixture(scope="module")
def golden(case_dir):
    c = PsaspCase(case_dir)
    return c, Catalog(c)


def test_corpus_composition_and_order(tmp_path, golden):
    c, cat = golden
    _mk_lib(tmp_path, "families/train/synth150_s1000", "synth150_s1000", 3,
            case=c, cat=cat)
    _mk_lib(tmp_path, "families/train/synth180_s1001", "synth180_s1001", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "families/val/synth150_s2000", "synth150_s2000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "families/transfer/synth400_s3000", "synth400_s3000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "lib118_diverse", "ieee118", 10, case=c, cat=cat)

    samples, prov = g4_train_corpus(tmp_path, test_every=5)
    fams = {s.family_id for s in samples}
    # val/transfer 家族绝不在语料内
    assert "synth150_s2000" not in fams and "synth400_s3000" not in fams
    # 组成 = 2 合成家族(3+2)+ lib 训练份(10 - ceil(10/5)=2 留出 → 8)
    assert prov["n_synth_families"] == 2
    assert prov["per_family"] == {"synth150_s1000": 3, "synth180_s1001": 2}
    assert prov["lib_pool"]["n_train"] == 8
    assert prov["lib_pool"]["n_holdout_excluded"] == 2
    assert len(samples) == 3 + 2 + 8 == prov["n_samples"]
    # 顺序确定:两次加载逐位同序(下游 val_every 切分可复现)
    samples2, prov2 = g4_train_corpus(tmp_path, test_every=5)
    assert [s.sample_id for s in samples] == [s.sample_id for s in samples2]
    assert prov2["fingerprint"] == prov["fingerprint"]


def test_corpus_leak_assertion_bites(tmp_path, golden):
    c, cat = golden
    _mk_lib(tmp_path, "families/train/synth150_s1000", "synth150_s1000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "families/val/synth150_s2000", "synth150_s2000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "lib118_diverse", "ieee118", 5, case=c, cat=cat)
    # 泄漏注入:把 val 家族整个复制进 train 目录(family_id 列不变)
    _mk_lib(tmp_path, "families/train/synth150_s2000", "synth150_s2000", 2,
            case=c, cat=cat)
    # 无 transfer:关 require_holdout 以聚焦 family_id 断言(另有专测覆盖缺留出)
    with pytest.raises(AssertionError, match="泄漏"):
        g4_train_corpus(tmp_path, test_every=5, require_holdout=False)


def test_corpus_require_holdout_missing_dir_raises(tmp_path, golden):
    """审查:47——留出目录缺失时 require_holdout 必须响亮报错,而非空转留出断言。"""
    c, cat = golden
    _mk_lib(tmp_path, "families/train/synth150_s1000", "synth150_s1000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "families/val/synth150_s2000", "synth150_s2000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "lib118_diverse", "ieee118", 5, case=c, cat=cat)
    # transfer 目录缺失(partial-sync)→ 默认 require_holdout=True 须抛
    with pytest.raises(AssertionError, match="留出集为空"):
        g4_train_corpus(tmp_path, test_every=5)


def test_corpus_content_leak_catches_relabel(tmp_path, golden):
    """审查:89——把 val 样本重贴新家族名(family_id 列 + npz 字段全改)绕过字符串断言,
    但物理内容不变 → 物理哈希跨集比对必须拦下。"""
    import shutil
    c, cat = golden
    _mk_lib(tmp_path, "families/train/synth150_s1000", "synth150_s1000", 2,
            case=c, cat=cat)
    valdir = _mk_lib(tmp_path, "families/val/synth150_s2000", "synth150_s2000", 3,
                     case=c, cat=cat)
    _mk_lib(tmp_path, "families/transfer/synth400_s3000", "synth400_s3000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "lib118_diverse", "ieee118", 5, case=c, cat=cat)
    # 重贴攻击:val 的一个 npz 原封拷进 train,并把 manifest family_id 列改成新训练名
    leak = tmp_path / "families/train/synth150_s9999"
    leak.mkdir(parents=True)
    src = valdir / "synth150_s2000_00000.npz"
    shutil.copy(src, leak / "synth150_s9999_00000.npz")   # 物理字节不变,仅文件名变
    (leak / "manifest.csv").write_text(
        "sample_id,family_id,solvable,load_factor\n"
        "synth150_s9999_00000,synth150_s9999,1,0.95\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="内容级泄漏"):
        g4_train_corpus(tmp_path, test_every=5)


def test_corpus_unsolvable_excluded_and_halffinished_raises(tmp_path, golden):
    c, cat = golden
    _mk_lib(tmp_path, "families/train/synth150_s1000", "synth150_s1000", 4,
            solvable_flags=[1, 0, 1, 0], case=c, cat=cat)
    _mk_lib(tmp_path, "lib118_diverse", "ieee118", 5, case=c, cat=cat)
    samples, prov = g4_train_corpus(tmp_path, test_every=5, require_holdout=False)
    assert prov["per_family"]["synth150_s1000"] == 2      # 负类(solvable=0)不入训语料
    # 半成品家族(缺 manifest)必须响亮报错,不静默跳过
    (tmp_path / "families/train/synth999_broken").mkdir(parents=True)
    with pytest.raises(AssertionError, match="manifest"):
        g4_train_corpus(tmp_path, test_every=5, require_holdout=False)


def test_corpus_fingerprint_tracks_content(tmp_path, golden):
    c, cat = golden
    _mk_lib(tmp_path, "families/train/synth150_s1000", "synth150_s1000", 2,
            case=c, cat=cat)
    lib = _mk_lib(tmp_path, "lib118_diverse", "ieee118", 5, case=c, cat=cat)
    _, p1 = g4_train_corpus(tmp_path, test_every=5, require_holdout=False)
    # manifest 内容变 → 指纹变(manifest 驱动,与冻结哈希同哲学)
    mf = lib / "manifest.csv"
    mf.write_text(mf.read_text(encoding="utf-8").replace("0.95", "0.96"),
                  encoding="utf-8")
    _, p2 = g4_train_corpus(tmp_path, test_every=5, require_holdout=False)
    assert p1["fingerprint"] != p2["fingerprint"]


def test_corpus_fingerprint_tracks_holdout_boundary(tmp_path, golden):
    """审查:102/83——留出边界漂移(val 家族增删)须反映在语料指纹上。"""
    c, cat = golden
    _mk_lib(tmp_path, "families/train/synth150_s1000", "synth150_s1000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "families/val/synth150_s2000", "synth150_s2000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "families/transfer/synth400_s3000", "synth400_s3000", 2,
            case=c, cat=cat)
    _mk_lib(tmp_path, "lib118_diverse", "ieee118", 5, case=c, cat=cat)
    _, p1 = g4_train_corpus(tmp_path, test_every=5)
    # 新增一个 val 家族(留出边界变)→ 指纹必须变(否则边界漂移不可审计)
    _mk_lib(tmp_path, "families/val/synth150_s2001", "synth150_s2001", 2,
            case=c, cat=cat)
    _, p2 = g4_train_corpus(tmp_path, test_every=5)
    assert p1["fingerprint"] != p2["fingerprint"]


def test_single_lib_split_matches_g15_idiom(tmp_path, golden):
    c, cat = golden
    lib = _mk_lib(tmp_path, "lib118_diverse", "ieee118", 10, case=c, cat=cat)
    pool, test = single_lib_split(lib, test_every=5)
    # lf 排序后每 5 取 1 留出:索引 0/5 → 留出 2,训练 8(g1/g15 逐位同款)
    assert len(pool) == 8 and len(test) == 2
    assert test[0].sample_id == "ieee118_00000"
    assert test[1].sample_id == "ieee118_00005"
