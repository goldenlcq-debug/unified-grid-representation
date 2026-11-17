"""PSASP 51 格式底层读写(方案附录 D / D16)。

保证:解析 → 无修改写回 → 字节级一致(T1.4a)。做法:记录保存原始逗号分隔的
"单元格"子串(含空白与引号),往返即 ','.join(cells);修改字段时只替换对应单元格文本
并尽量保留原宽度,其余单元格与行尾逐字节不变。

51 格式规则(附录 D):gb18030 编码;英文逗号分隔;每行记录末尾以逗号结尾(→ 末单元格为空);
'#' 开头为文件说明行;字符串用成对单引号;字符串内禁用逗号(故 split(',') 安全)。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

ENCODING = "gb18030"


def _decimals_of(cell: str) -> int:
    """推断单元格数值的小数位数(用于修改时保持格式)。"""
    s = cell.strip()
    if "." in s:
        frac = s.split(".", 1)[1]
        # 去掉可能的指数部分
        for ch in ("e", "E"):
            if ch in frac:
                frac = frac.split(ch, 1)[0]
        return len(frac)
    return 0


class Record:
    """一条记录行,持有精确的逗号分隔单元格。"""

    __slots__ = ("cells", "dirty")

    def __init__(self, cells: List[str]) -> None:
        self.cells = cells
        self.dirty = False

    @classmethod
    def parse(cls, line: str) -> "Record":
        return cls(line.split(","))

    def text(self) -> str:
        return ",".join(self.cells)

    # ---- 读取 ----
    def raw(self, i: int) -> str:
        return self.cells[i]

    def get_int(self, i: int) -> int:
        return int(self.cells[i].strip())

    def get_float(self, i: int) -> float:
        return float(self.cells[i].strip())

    def get_str(self, i: int) -> str:
        """去空白与成对单引号,返回内容(保留引号内两端空白之外的原文,右侧空白 rstrip)。"""
        s = self.cells[i].strip()
        if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
            s = s[1:-1]
        return s.rstrip()

    def n_fields(self) -> int:
        return len(self.cells)

    # ---- 修改(保留宽度)----
    def _set_cell_preserve_width(self, i: int, value_text: str) -> None:
        width = len(self.cells[i])
        self.cells[i] = value_text.rjust(width) if len(value_text) < width else value_text
        self.dirty = True

    def set_int(self, i: int, value: int) -> None:
        self._set_cell_preserve_width(i, str(value))

    def set_float(self, i: int, value: float, decimals: int | None = None) -> None:
        if decimals is None:
            decimals = _decimals_of(self.cells[i]) or 6
        self._set_cell_preserve_width(i, format(value, f".{decimals}f"))

    def set_str(self, i: int, value: str) -> None:
        # 保留原单元格前导空白与引号风格:重建为 '<value 补足到原内宽>'
        orig = self.cells[i]
        lead = orig[: len(orig) - len(orig.lstrip())]
        inner_width = 0
        s = orig.strip()
        if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
            inner_width = len(s) - 2
        new_inner = value.ljust(inner_width) if len(value) < inner_width else value
        self.cells[i] = f"{lead}'{new_inner}'"
        self.dirty = True


class Card:
    """一个 51 格式文件:有序元素(说明行/空行/记录)+ 逐行独立行尾。

    逐行保存行尾(而非统一风格),以应对 PSASP 混合行尾的真实文件
    (多数行 \\r\\n、末行 \\n)——保证字节级往返一致。
    """

    def __init__(
        self,
        elements: List[Union[str, Record]],
        endings: List[str],
        path: Path | None = None,
    ) -> None:
        self.elements = elements
        self.endings = endings  # 与 elements 等长,每项为该行之后的原始换行(\r\n / \n / '')
        self.path = path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Card":
        path = Path(path)
        text = path.read_bytes().decode(ENCODING)
        parts = text.split("\n")
        elements: List[Union[str, Record]] = []
        endings: List[str] = []
        n = len(parts)
        for idx, part in enumerate(parts):
            is_last = idx == n - 1
            if is_last:
                if part == "":
                    break  # 末尾换行已由上一行的 ending 记录,无额外元素
                content, ending = part, ""
            elif part.endswith("\r"):
                content, ending = part[:-1], "\r\n"
            else:
                content, ending = part, "\n"
            stripped = content.strip()
            if stripped == "" or stripped.startswith("#"):
                elements.append(content)  # 原样保留(说明行/空行)
            else:
                elements.append(Record.parse(content))
            endings.append(ending)
        return cls(elements, endings, path)

    @property
    def records(self) -> List[Record]:
        return [e for e in self.elements if isinstance(e, Record)]

    def to_text(self) -> str:
        buf = []
        for el, ending in zip(self.elements, self.endings):
            buf.append(el.text() if isinstance(el, Record) else el)
            buf.append(ending)
        return "".join(buf)

    def to_bytes(self) -> bytes:
        return self.to_text().encode(ENCODING)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_bytes(self.to_bytes())

    @property
    def dirty(self) -> bool:
        return any(isinstance(e, Record) and e.dirty for e in self.elements)
