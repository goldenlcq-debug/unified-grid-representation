"""[G] R2 解冻臂**口径逐位断言**(§0.0-R2 第 2 条;零机时)。

协议原文:"跑完**立即断言**臂②③ 报告的 steps / weights / coverage.batches(7 批)/
cct_joined(620)/ split.held_ids 与 r1b **逐位相同**,不同即口径已污染、数字作废。"

本脚本把那句话变成可执行的门:任何一项不等 → 非零退出 + 明示哪一项、差在哪。
**顺带多断三项**(同为"口径已污染"的等价证据,不额外要求机时):
  n_train / n_in / n_out(例数)、train_families(实测训练家族)、coverage.rows(标签行)。
另**并报**(不设门,仅入档,因解冻本就该改变这些):best_step、AMP 溢出跳步账、
编码器可训参数量、encoder_grad_reached。

  python scripts/r2_assert_parity.py --base runs/ts_reasoner_r1b_s{0,1,2} \\
      --arms runs/ts_reasoner_r2_last2_s{0,1,2} runs/ts_reasoner_r2_all_s{0,1,2}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# 逐位相同项(§0.0-R2 第 2 条原文 5 项 + 等价证据 3 项)
PARITY = {
    "steps":            lambda r: r["steps"],
    "weights":          lambda r: r["weights"],
    # 只比批名不满足协议"逐位相同"的字面(粮仓重产后批名不变而 rows 会变);
    # 连 family/rows 一起比(r1b 三座之间实测已两两相等,不会误报)。
    "coverage.batches": lambda r: sorted((b["batch"], b["family"], b["rows"])
                                         for b in r["coverage"]["batches"]),
    "coverage.rows":    lambda r: r["coverage"]["rows"],
    "cct_joined":       lambda r: r["coverage"]["cct_joined"],
    "split.held_ids":   lambda r: r["split"]["held_ids"],
    "split.train_families": lambda r: r["split"]["train_families"],
    "split.n":          lambda r: [r["split"]["n_train"], r["split"]["n_in"],
                                   r["split"]["n_out"]],
}
# 并报项(解冻本就该改变,不设门)
INFO = ("unfreeze_encoder", "best_step", "encoder_trainable_params",
        "encoder_grad_reached", "amp_nonfinite_steps", "amp_nonfinite_rate",
        "amp_verdict", "v1_fingerprint", "normalizer_fingerprint",
        # 收敛/续训必须显形(放行 resume 的代价就是要在报表上看得见)
        "early_stopped", "ran_full_steps", "budget_note", "n_resumes", "resume_note")
# **配方逐字比对**(对抗审查 major:上面 8 项对 --batch/--lr/--patience 等**完全盲**,
# 配方写错也能 8/8 全绿)。ts_report.args 落了全部 argparse 值,故可直接对比。
# 只允许下列键不同 —— 其余任一不同即配方漂移。
ARGS_ALLOWED_DIFF = {"out", "unfreeze_encoder", "encoder_lr", "seed",
                     # **不影响数值口径的执行细节**(2026-07-25 审查):
                     # resume —— 本机 NVIDIA 驱动(nvlddmkm)会打断长跑,续训是常态;
                     #   若不放行,崩一次该座就被判"配方漂移→数字作废",1h 机时白烧。
                     #   续训本身有六道一致性校验把关(编码器/划分/权重/批/侧通道/解冻臂),
                     #   且 scaler 与 amp/裁剪账已累计入档 → 数值可比。**但必须响亮并报**。
                     # keep_last —— 只删旧周期检查点,不碰 best.pt 与训练轨迹。
                     "resume", "keep_last"}

# **路径型参数**:比"指向同一个目录",不比"字符串长得一样"。
# 起因(2026-07-26 R5 实测):R4 六座是从 PowerShell 起的(`E:\AI-2\...` 反斜杠),
# R5 四座是从 bash 起的(`E:/AI-2/...` 正斜杠) —— 同一个归档目录,却被判"3 项配方漂移
# → 数字作废"。而同批的 8 项口径断言(coverage.rows / split.held_ids / cct_joined …)
# **全部逐位相同**,已独立证明装载到的数据一致。
# 归一只做**分隔符 + 大小写 + 冗余分隔**(Windows 语义),**不做 resolve()**
# —— resolve 会跟随符号链接/映射盘,那才可能把两个真不同的位置说成同一个。
# 归一后仍不同 = 真的指向别处,照旧判作废。
PATH_ARGS = {"root", "labels_root", "cct", "v1_dir", "faults", "labels", "validation"}

# **改码后新增的 argparse 旗及其默认值**(见下方 args 比对处的说明)。
# 加新旗时**必须同步登记到这里**,否则老基线会被判"配方漂移"。
NEW_ARG_DEFAULTS = {
    "min_steps": 0,             # 2026-07-27 早停下限(A);0 = 关闭 = 历史行为
    "train_mode_frac": 1.0,     # 2026-07-31 方式消融;1.0 = 不消融 = 历史行为
    "train_mode_seed": 20260731,
    "train_mode_families": None,
}


def _norm_path_arg(v):
    def one(x):
        return os.path.normcase(os.path.normpath(str(x))) if isinstance(x, str) else x
    if isinstance(v, (list, tuple)):
        return [one(x) for x in v]
    return one(v)


def _load(p: Path) -> dict:
    f = p / "ts_report.json"
    if not f.is_file():
        raise SystemExit(f"[parity] ✋ 缺 {f}")
    return json.loads(f.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True, help="臂① 冻结基线 run 目录")
    ap.add_argument("--arms", nargs="+", required=True, help="臂②③ run 目录")
    ap.add_argument("--out", default="runs/r2_parity.json")
    # **配对设计专用**(2026-07-28 审查 major):`ARGS_ALLOWED_DIFF` 里含 `"seed"` ——
    # 在 R2/R4/R6 那种"跨种子比臂"的设计里是对的,但**配对设计里 seed 相等是承重假设**,
    # 不加这条,把 `dose_b_s2` 和 `dose_c_s3` 配错对,parity 会 8/8 全绿放行。
    # 同理 `--allow-differ weights` 会放行**整个 weights 字典**,`--w-type` 打错也照过 ——
    # `--pair-weights-only` 限定只准 weights 里的指定键不同,其余逐键断言。
    ap.add_argument("--pair-same-seed", action="store_true",
                    help="配对设计:断言 base 与 arm 的 args.seed 相等")
    ap.add_argument("--pair-weights-only", nargs="*", default=None,
                    help="配对设计:weights 字典里只准这些键不同(其余逐键断言)")
    ap.add_argument("--allow-differ", nargs="*", default=[],
                    help="**显式声明**本轮的自变量(如 R3 的 steps),该项不同不判作废;"
                         "其余项照旧逐位断言。用于'只改一个变量'的对照实验 —— "
                         "不给此旗时任何一项不同都作废(R2 口径)")
    a = ap.parse_args()
    allow = set(a.allow_differ)
    # 合法的自变量名 = 8 项口径键 ∪ 已放行执行细节 ∪ **报告 args 里真实存在的旗**。
    # 补最后一类(2026-07-28):以前只认 PARITY/ARGS_ALLOWED_DIFF,于是"只改一个 argparse 旗"
    # 的对照实验(如 CCT 剂量-响应把 `--w-cct` 当自变量)根本无法声明 —— 声明会被拒,
    # 不声明又会被判配方漂移,两头堵死。放行前仍要求该键在**双方报告里都存在**,
    # 防止把打错的名字当成"已声明"而静默放过一处真漂移。
    _arg_keys = set()
    for p in list(a.base) + list(a.arms):
        rep = _load(Path(p))
        _arg_keys |= set(rep.get("args", {}))
    for k in allow:
        if k not in PARITY and k not in ARGS_ALLOWED_DIFF and k not in _arg_keys:
            raise SystemExit(f"[parity] ✋ --allow-differ {k!r} 不是已知比对项;"
                             f"可选口径键 {sorted(PARITY)};或任一在报告 args 中存在的旗名")
    if allow:
        print(f"[parity] ⚠ 本轮显式放行的自变量:{sorted(allow)}(其余项照旧逐位断言)")

    bases = {Path(p).name: _load(Path(p)) for p in a.base}
    # 基线自身先自洽(三种子共享留出 → 8 项应两两相同);不自洽说明基线就不可比
    ref_name, ref = next(iter(bases.items()))
    ref_vals = {k: fn(ref) for k, fn in PARITY.items()}
    bad = 0
    for nm, rep in bases.items():
        for k, fn in PARITY.items():
            if fn(rep) != ref_vals[k]:
                print(f"[parity] ✋ 基线内部不一致:{nm}.{k} ≠ {ref_name}.{k}")
                bad += 1
    print(f"[parity] 基线 {len(bases)} 座自洽性:{'过' if not bad else '**不过**'}"
          f"(基准 = {ref_name})")

    out = {"base_runs": sorted(bases), "reference": ref_name,
           "parity_keys": sorted(PARITY), "arms": {}, "verdict": {}}
    for p in a.arms:
        p = Path(p)
        rep = _load(p)
        diffs, declared = {}, {}
        for k, fn in PARITY.items():
            got = fn(rep)
            if got != ref_vals[k]:
                (declared if k in allow else diffs)[k] = {"expect": ref_vals[k], "got": got}
        # 配方逐字比对(仅当基线也落了 args;r1b 跑在改码前故无该键 —— 如实降级为"不可比")
        arg_diffs, arg_note = {}, ""
        if "args" in ref and "args" in rep:
            for k in sorted(set(ref["args"]) | set(rep["args"])):
                if k in ARGS_ALLOWED_DIFF or k in allow:
                    continue
                bv, av = ref["args"].get(k), rep["args"].get(k)
                # **新增旗留在默认值 = 配方未变**(2026-07-27)。args 取的是键并集,
                # 老 run 跑在新旗加进来之前、根本没有该键 → 不特判的话,此后每一座都会
                # 相对 R6 基线判"配方漂移 → 数字作废",而我们恰恰要拿 R6 当基线比。
                # 只在"一侧缺键 ∧ 另一侧等于该旗的默认值"时放行;显式改过默认值照旧判漂移。
                if k in NEW_ARG_DEFAULTS and (k not in ref["args"] or k not in rep["args"]):
                    present = av if k not in ref["args"] else bv
                    if present == NEW_ARG_DEFAULTS[k]:
                        continue
                if k in PATH_ARGS:
                    bv, av = _norm_path_arg(bv), _norm_path_arg(av)
                if bv != av:
                    arg_diffs[k] = {"base": ref["args"].get(k), "arm": rep["args"].get(k)}
                    if k in PATH_ARGS:
                        arg_diffs[k]["note"] = "路径已按分隔符/大小写归一后仍不同 = 真的指向别处"
        else:
            arg_note = ("基线 ts_report 无 args 键(跑在改码前)——配方逐字比对不可用,"
                        "只能靠上面 8 项 + 人工核命令")
        # **配对断言**(独立于上面的 8 项与配方比对,失败即响亮作废)
        pair_bad = {}
        if a.pair_same_seed:
            bs, as_ = ref.get("args", {}).get("seed"), rep.get("args", {}).get("seed")
            if bs != as_ or bs is None:
                pair_bad["seed"] = {"base": bs, "arm": as_,
                                    "why": "配对设计要求同 seed;配错对会让整轮 Δ 失去意义"}
        if a.pair_weights_only is not None:
            allow_w = set(a.pair_weights_only)
            bw, aw = ref.get("weights", {}) or {}, rep.get("weights", {}) or {}
            for k in sorted(set(bw) | set(aw)):
                if k in allow_w:
                    continue
                if bw.get(k) != aw.get(k):
                    pair_bad[f"weights.{k}"] = {"base": bw.get(k), "arm": aw.get(k),
                                                "why": "本轮只准这些权重键不同:"
                                                       f"{sorted(allow_w)}"}
        ok = not diffs and not arg_diffs and not pair_bad
        if pair_bad:
            print(f"[parity] ✋ {p.name}:**配对断言不过** {pair_bad}")
        out["arms"][p.name] = {"parity_ok": ok, "diffs": diffs, "arg_diffs": arg_diffs,
                               "pair_assert_failed": pair_bad,
                               "declared_independent_vars": declared, "arg_note": arg_note,
                               "info": {k: rep.get(k) for k in INFO}}
        for k, d in declared.items():
            print(f"[parity]      自变量 {k}:{d['expect']} -> {d['got']}(已显式声明,不判作废)")
        if arg_note:
            print(f"[parity] ⚠ {p.name}:{arg_note}")
        for k, d in arg_diffs.items():
            print(f"[parity] ✋ {p.name} 配方漂移 --{k}:基线 {d['base']!r} vs 本臂 {d['arm']!r}")
        out["verdict"][p.name] = "口径同 r1b(可比)" if ok else "**口径污染,数字作废**"
        mark = "OK  " if ok else "FAIL"
        print(f"[parity] {mark} {p.name}:" + (f"{len(PARITY) - len(declared)}/"
              f"{len(PARITY) - len(declared)} 项(除已声明自变量)与 r1b 逐位相同"
              + ("" if arg_note else " + 配方逐字同") if ok else
              f"**{len(diffs)} 项口径不同 / {len(arg_diffs)} 项配方漂移 → 数字作废**:"
              f"{sorted(diffs) + ['--' + k for k in arg_diffs]}"))
        for k, d in diffs.items():
            print(f"           {k}: 期望 {json.dumps(d['expect'], ensure_ascii=False)[:160]}")
            print(f"           {' ' * len(k)}  实得 {json.dumps(d['got'], ensure_ascii=False)[:160]}")
        info = out["arms"][p.name]["info"]
        print(f"           [并报,不设门] 臂={info['unfreeze_encoder']} "
              f"best_step={info['best_step']} 可训编码器参数={info['encoder_trainable_params']} "
              f"梯度到编码器={info['encoder_grad_reached']} "
              f"AMP跳步={info['amp_nonfinite_steps']}({info['amp_nonfinite_rate']})")
        print(f"           [收敛] 早停={info.get('early_stopped')} / 跑满={info.get('ran_full_steps')}"
              f";{info.get('budget_note')}")
        if info.get("n_resumes"):
            print(f"           ⚠ **本座续训过 {info['n_resumes']} 次**(放行但必须显形):"
                  f"{info.get('resume_note')}")
        # **配对断言也要计进总判**(2026-07-28):不计的话,末行会在有 FAIL 的情况下
        # 照打"可进入出数" —— 静默放行,正是本仓最忌讳的那类失效。
        bad += len(diffs) + len(arg_diffs) + len(pair_bad)

    op = REPO / a.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[parity] -> {op}")
    if bad:
        print("[parity] ✋ **口径已污染,本轮数字作废**(§0.0-R2 第 2 条)")
        return 1
    print("[parity] 全部臂与 r1b 逐位同口径 —— 可进入出数")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
