"""DATALIB.DAT 型号参数库解析(schema v0.3 / S1,2026-07-16)。

PSASP 把设备动态参数放在**型号库**里,`ST.S5` 等指派卡只存"型号 + 参数组号"引用:
`ST.S5.Lg → DATALIB #,GEN, 段首列 Par_No`(励磁 Lvr→#,AVR*、调速 Lgo→#,GOV1、
PSS Lpss→#,PSS1 同法)。cepri36/ieee39 的 ST.S5 内联 X'd/Tj 列源文件即为 0,真参数
只在此库中——这是"档案 tj/xdp=0"的根因(S1 侦察 2026-07-16;变更记录 07-13 亦有记载)。

文件格式(gb18030,逗号分隔,行尾逗号):
  #,3.7,            ← 版本头 + 各段行数计数行
  #,GEN,            ← 段名行
  <Par_No>, v1, v2, ...   ← 参数行(首列 = 参数组号,其余浮点,尾部可有引号名)

`#,GEN,` 段列义(cepri36/ieee118 golden 实测对拍;idx 含 Par_No):
  0=Par_No 1=Xd 2=Xd' 3=Xd'' 4=Xq 5=Xq' 6=Xq'' 7=X2 8=X0/Ra
  9=Td0' 10=Td0'' 11=Tq0' 12=Tq0'' 13=Tj 14=D
哨兵:9999(≈"该时间常数不建模")—— 消费方自行处理(assemble 置 0)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

ENC = "gb18030"

# #,GEN, 段列名 → 列号(含 Par_No 的 0 基索引)
GEN_COLS = {"xd": 1, "xdp": 2, "xdpp": 3, "xq": 4, "xqp": 5, "xqpp": 6,
            "x2": 7, "ra": 8, "td0p": 9, "td0pp": 10, "tq0p": 11, "tq0pp": 12,
            "tj": 13, "damp": 14}
SENTINEL = 9000.0     # ≥此值视为"不建模"哨兵(实测 9999)


def parse_datalib(path) -> Dict[str, Dict[int, List[Optional[float]]]]:
    """DATALIB.DAT → {段名: {参数组号: [逐列值(float;不可解析处 None)]}}。

    段名 = `#,NAME,` 的 NAME(如 GEN/AVR1/GOV1/PSS1/LOAD/MOTO);版本头段(如 3.7)照收无害。
    文件不存在返回 {}。
    """
    p = Path(path)
    if not p.is_file():
        return {}
    out: Dict[str, Dict[int, List[Optional[float]]]] = {}
    section = None
    for line in p.read_bytes().decode(ENC, errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(",")
            section = parts[1].strip() if len(parts) > 1 else None
            if section:
                out.setdefault(section, {})
            continue
        if section is None:
            continue
        cells = [c.strip() for c in line.split(",")]
        while cells and cells[-1] == "":
            cells.pop()
        if not cells:
            continue
        try:
            par_no = int(float(cells[0]))
        except ValueError:
            continue
        row: List[Optional[float]] = []
        for c in cells:
            try:
                row.append(float(c))
            except ValueError:
                row.append(None)          # 引号名等非数值单元
        out[section][par_no] = row
    return out


def gen_params(lib: Dict[str, Dict[int, List[Optional[float]]]], par_no: int
               ) -> Optional[Dict[str, float]]:
    """#,GEN, 段第 par_no 组 → {字段名: 值}(哨兵原样保留,消费方处理);缺组/缺列返回 None。"""
    sec = lib.get("GEN", {})
    row = sec.get(par_no)
    if row is None:
        return None
    need = max(GEN_COLS.values())
    if len(row) <= need:
        return None
    vals = {}
    for name, idx in GEN_COLS.items():
        v = row[idx]
        if v is None:
            return None
        vals[name] = float(v)
    return vals
