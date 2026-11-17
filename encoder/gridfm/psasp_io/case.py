"""PsaspCase:加载一个 PSASP 算例目录,持有各卡片,提供类型化视图与写回(方案 T1.2/T1.4)。

写回契约(D16/5.2):只改动方式变量字段(LF.L5 的 Pg/Qg/V0/mark、LF.L6 的 Pl/Ql、
LF.L2/L3 的 mark、ST.S11 整卡生成),其余字节原样保留;开机 mark 改动须 LF.L5 与 ST.S5 同步。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from . import cards
from .lowlevel import Card

# 潮流侧 + 稳定侧关注的卡片(文件名可无扩展名,如 118 工程;也可带 .txt,如格式说明目录)
_KNOWN = [
    "LF.L0", "LF.L1", "LF.L2", "LF.L3", "LF.L5", "LF.L6", "LF.L7",
    "LF.ZONE", "LF.AREA", "LF.PSTATION", "LF.L10", "LF.L20", "LF.BUS",
    "ST.S0", "ST.S1", "ST.S5", "ST.S6", "ST.S11",
    "LF.CAL", "LF.LP1", "LF.LP2", "LF.LP3", "LF.LP5",
]


def _resolve(case_dir: Path, name: str) -> Optional[Path]:
    for cand in (case_dir / name, case_dir / f"{name}.txt"):
        if cand.exists():
            return cand
    return None


class PsaspCase:
    def __init__(self, case_dir: Union[str, Path]) -> None:
        self.dir = Path(case_dir)
        self.cards: Dict[str, Card] = {}
        for name in _KNOWN:
            p = _resolve(self.dir, name)
            if p is not None:
                self.cards[name] = Card.load(p)
        # 记录每张卡的实际路径(写回用)
        self._paths: Dict[str, Path] = {n: _resolve(self.dir, n) for n in self.cards}

    def has(self, name: str) -> bool:
        return name in self.cards

    # ---- 输入侧类型化视图 ----
    def buses(self) -> List[cards.BusView]:
        recs = self.cards["LF.L1"].records
        return [cards.BusView(r, i + 1) for i, r in enumerate(recs)]  # 行号从 1 起 = 计算母线编号

    def aclines(self) -> List[cards.AclineView]:
        if "LF.L2" not in self.cards:
            return []
        return [cards.AclineView(r, i) for i, r in enumerate(self.cards["LF.L2"].records)]

    def xfmrs(self) -> List[cards.XfmrView]:
        if "LF.L3" not in self.cards:
            return []
        return [cards.XfmrView(r, i) for i, r in enumerate(self.cards["LF.L3"].records)]

    def gens(self) -> List[cards.GenView]:
        if "LF.L5" not in self.cards:
            return []
        return [cards.GenView(r, i) for i, r in enumerate(self.cards["LF.L5"].records)]

    def loads(self) -> List[cards.LoadView]:
        if "LF.L6" not in self.cards:
            return []
        return [cards.LoadView(r, i) for i, r in enumerate(self.cards["LF.L6"].records)]

    def gen_dyns(self) -> List[cards.GenDynView]:
        if "ST.S5" not in self.cards:
            return []
        return [cards.GenDynView(r) for r in self.cards["ST.S5"].records]

    def datalib(self):
        """DATALIB.DAT 型号参数库(schema v0.3/S1;缓存)。不存在返回 {}。

        注意:DATALIB 不在 _KNOWN(它是型号库不是方式卡,写回契约不触碰),按需直读。
        """
        if not hasattr(self, "_datalib_cache"):
            from .datalib import parse_datalib
            self._datalib_cache = parse_datalib(self.dir / "DATALIB.DAT")
        return self._datalib_cache

    def faults(self) -> List[cards.FaultView]:
        if "ST.S11" not in self.cards:
            return []
        return [cards.FaultView(r) for r in self.cards["ST.S11"].records]

    # ---- 结果卡 ----
    def bus_results(self) -> List[cards.BusResult]:
        return cards.parse_lp1(self.cards["LF.LP1"]) if "LF.LP1" in self.cards else []

    def branch_flows(self) -> List[cards.BranchFlow]:
        """LP2(交流线)+ LP3(变压器,同版式;2026-07-12 修——此前 LP3 从未被读,
        变压器支路潮流状态恒缺失,118 夹具 KCL 在变压器母线差 2.33 pu 抓获)。"""
        out = cards.parse_lp2(self.cards["LF.LP2"]) if "LF.LP2" in self.cards else []
        if "LF.LP3" in self.cards:
            out = out + cards.parse_lp2(self.cards["LF.LP3"])
        return out

    def gen_results(self) -> List[cards.GenResult]:
        return cards.parse_lp5(self.cards["LF.LP5"]) if "LF.LP5" in self.cards else []

    def converged(self) -> Optional[bool]:
        return cards.parse_cal_converged(self.cards["LF.CAL"]) if "LF.CAL" in self.cards else None

    # ---- 方式变量修改(带 L5/S5 mark 同步)----
    def set_gen_online(self, gen_row: int, online: bool) -> None:
        """按 LF.L5 行号设开机状态,并同步 ST.S5 同行(逐行同序,方案 T1.4)。"""
        g = self.gens()[gen_row]
        g.mark = 1 if online else 0
        dyns = self.gen_dyns()
        if gen_row < len(dyns):
            assert dyns[gen_row].bs_no == g.bs_no, "LF.L5 与 ST.S5 行序不一致,拒绝盲改 mark"
            dyns[gen_row].mark = 1 if online else 0

    # ---- 写回 ----
    def save(self, out_dir: Union[str, Path]) -> None:
        """写回所有卡片到 out_dir(未改动的字节级不变)。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, card in self.cards.items():
            fname = self._paths[name].name  # 保留原文件名(有无扩展名)
            card.save(out / fname)

    def save_inplace(self) -> None:
        for name, card in self.cards.items():
            card.save(self._paths[name])

    @property
    def n_buses(self) -> int:
        return len(self.cards["LF.L1"].records)
