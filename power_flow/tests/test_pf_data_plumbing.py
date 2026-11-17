# -*- coding: utf-8 -*-
"""接线闸单测:凡是造 `_pf_data` 输入的调用点,必须把**影响输入特征的开关**逐个透传。

由来(同型故障连出两轮):
  · E18(§3.33(b)):新增 `--dc-encoding lossybv` 时,四个评估脚本的 `dc_dim` 映射漏加一处 → 12 个评估作业首跑全 rc=1;
  · E19(2026-08-24 对抗审查):新增 `--edge-base` 时,**训练脚本自身**三处 `eval_pu_mae` 漏传 → 会"训练用 θ_bΔ 边基线、
    验证用 acv 边基线",而验证 score 决定 best.pt、终评数进 pf_report.json 供判据读 —— 且**不报错**,只静默出错数。

因此改成机器检查:AST 扫全仓 `_pf_data` / `eval_pu_mae` 调用点,逐个核对开关关键字是否给全。
**加新开关时,把它加进 REQUIRED,这条单测会立刻指出所有漏点。**
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TARGETS = ("_pf_data", "eval_pu_mae")

# 影响输入特征的开关(默认值 = 旧行为,但"取默认"和"按检查点透传"是两码事)
REQUIRED = ("shz", "edge_base")

# 白名单:配方硬编码的历史期探针,与主线口径本就不同源(改了反而误导);注释在各文件内。
LEGACY = {("probes_5060/probe_refine_vs_probe.py", "_pf_data")}


def _calls():
    for p in sorted(SCRIPTS.rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:                       # 语法错另有单测/编译闸把关
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in TARGETS:
                rel = p.relative_to(SCRIPTS).as_posix()
                yield rel, name, node.lineno, {k.arg for k in node.keywords if k.arg}


def test_every_call_site_passes_input_switches():
    missing = []
    for rel, name, lineno, kw in _calls():
        if (rel, name) in LEGACY:
            continue
        if rel == "pf_reasoner.py" and name == "_pf_data" and lineno < 200:
            continue                              # eval_pu_mae 体内那次:形参已由本函数签名透传
        for req in REQUIRED:
            if req not in kw:
                missing.append(f"{rel}:{lineno} {name}() 漏传 {req}(现有 {sorted(kw)})")
    assert not missing, "以下调用点漏传输入开关(新开关必须四处同步):\n  " + "\n  ".join(missing)


def test_signatures_expose_the_switches():
    """两个函数的签名本身必须带上这些开关(否则调用点传了也会 TypeError)。"""
    tree = ast.parse((SCRIPTS / "pf_reasoner.py").read_text(encoding="utf-8"))
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in TARGETS:
            seen[node.name] = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
    assert set(seen) == set(TARGETS), f"没找到 {set(TARGETS) - set(seen)}"
    for name, args in seen.items():
        for req in REQUIRED:
            assert req in args, f"{name}() 签名缺 {req}"


def test_legacy_whitelist_still_points_at_real_code():
    """白名单不许烂掉:里面的文件必须还在,且确实有该调用。"""
    got = {(rel, name) for rel, name, _l, _k in _calls()}
    for entry in LEGACY:
        assert entry in got, f"白名单条目 {entry} 已不存在,请从 LEGACY 删掉"
