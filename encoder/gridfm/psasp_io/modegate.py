"""暂稳方式门(2026-07-16 用户裁决):基态电压不合格的方式不做暂稳、不入粮仓。

**动因(空白对照实证)**:D21 ② 的稳态分句 = "末 1s 全网最低电压 <0.9 即失稳"。
若某方式**故障前**的潮流解本身 vmin <0.9,则不论在哪条线打什么故障、故障多久,
末 1s 电压照样 <0.9 → **一律判失稳,标签与故障无关**。118 实测:不加任何故障跑
满 12s,系统完全平稳(功角恒 66°、电压恒 0.8955)却被判 voltage 失稳。

污染面(2026-07-16 全库实测):lib118 52/150 方式基态 <0.9(中位 0.9154,IEEE-118
基准算例自身 vmin=0.8991);lib_cepri36 12/102(其 750 例已入粮仓,**实测失稳率
100%**,而健康方式仅 47~50%);lib_ieee39 0/100(min 0.9417,不受影响)。

**处置 = 数据门,不动判据**(用户裁决 A):基态就违反稳态电压限的方式不是合法的
暂稳研究起点。

**⚠ 门槛 0.905 的余量标定是错的,且已知不足**(2026-07-14 对抗审查 critical,实测复核):
- 原论证"自清除故障下末 1s vmin ≈ 基态 vmin(实测 0.8955→0.89548,压降 2e-5)"取自
  一次**无故障**的空白对照跑 —— **拿无故障数据去标定带故障场景的余量**,正是本项目
  最贵的 bug 形状(基准本身没被校验)。
- 真实带故障数据(runs/ts_labels_ieee118_v1,1,832 条 benign 行 = 功角/频率 OK 且无
  low_recover/overvoltage,只有稳态分句可能咬人):良性故障的**永久压降**中位
  0.011~0.028、**p90 = 0.030**,是 0.005 余量的 **6 倍**。
- 后果(已量化,不是推测):**仅 low_steady 触发**的残余行 = **118 的 5.6%**
  (355/6,383)。这些行的失稳与故障的关系被削弱(但**并非无关**:各 base_vmin 档内
  失稳率仍随切除时长单调上升,门确实杀掉了那批 100% 失稳的退化方式)。
- **为什么不抬门槛**:要让判据②真正由故障决定需 ≳0.9 + p90 ≈ **0.93**,而 lib118 门后
  的 base_vmin 中位才 0.9256 → 抬到 0.93 会清空 118 方式库、**拓扑维度塌回 1、本轮
  实验直接作废**。根治要么改 118 方式库的运行点(基态电压普遍贴着 0.9),要么把判据②
  的稳态分句改成**相对自身基态**的口径(末1s vmin ≥ base_vmin − δ)—— 后者是 D21
  判据变更,须用户裁决 + 升 CRITERIA_VERSION + 全库重判。**待用户裁决,本轮如实带病上路。**
- **验收尺子未被污染**(关键):[外] ieee39 的"仅 low_steady"行 = **0 / 10,200**
  (该网基态 vmin 最低 0.9417,门一个不剔),每条 ieee39 失稳行都同时被功角/频率/
  low_recover 判定 → 稳态分句从未独立翻转过 [外] 标签。且 118 的 5.6% 污染**低于**
  存量 cepri36(6.3% / 7.0%)—— 这条伪影既不是新引入的,也不是 118 特有的。
- 残余风险(须在重训报告里核):模型可能学到"基态低压 ⇒ 失稳"的捷径(母线电压 v 在
  节点特征里,min-pool 即得),而该捷径与家族强相关(118 门后中位 0.926 / cepri36
  0.983 / ieee39 0.972)→ 套到 ieee39 上会系统性预测"稳"。**诊断量 = [外] 的
  pred_pos_rate 与召回**(已进报告)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import cards
from .lowlevel import Card

TS_BASE_VMIN_FLOOR = 0.905


def base_vmin(case_dir: Path) -> Optional[float]:
    """算例故障前潮流解的全网最低母线电压(LF.LP1);无解/不可读 → None。"""
    lp1 = Path(case_dir) / "LF.LP1"
    if not lp1.exists():
        return None
    try:
        bus = cards.parse_lp1(Card.load(lp1))
    except Exception:  # noqa: BLE001  卡损坏 → 无基准(调用方按"不合格"处理,保守)
        return None
    return min((b.v_pu for b in bus), default=None)


def ts_mode_ok(case_dir: Path, floor: float = TS_BASE_VMIN_FLOOR) -> bool:
    """该方式是否够格做暂稳(基态 vmin ≥ floor)。取不到基准 → False(保守拒收)。"""
    v = base_vmin(case_dir)
    return v is not None and v >= floor
