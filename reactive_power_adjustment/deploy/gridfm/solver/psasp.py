"""真实 PSASP 求解器(方案 T2.2/T2.3 / 7.0 / D16)。实现 SolverRunner。

调用纪律(psasp skill 实战核实):gb18030;OMP_NUM_THREADS=2 钉死;SetErrorMode 抑制弹窗;
工作目录=数据目录;暂态用 Popen + 轮询 STANA 稳定 + taskkill(wmudrt 算完会弹不自动关的窗口,
subprocess.run 会被它卡死——必须 kill,STANA 稳定后结果已在盘上);slm_init error=加密狗失效熔断。

**窗口抑制(2026-07-14)**:wmudrt 的 GUI 窗口跑在**独立隐藏桌面**上(见
_hidden_desktop_popen);SW_HIDE/CREATE_NO_WINDOW 对它无效(实测)。计算行为不变。
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

if sys.platform == "win32":
    from ctypes import wintypes
else:                                    # 非 Windows(CI/开发机):只为 import 可用
    wintypes = None                      # noqa: N816

from ..psasp_io import cards
from ..psasp_io.lowlevel import Card
from .base import CCTResult, FaultSpec, Job, PFResult, SolverRunner, TSResult

# 2026-08-21 Linux 部署补丁:子进程输出编码随卡片编码;默认与补丁前一致
ENC = os.environ.get("GRIDFM_CARD_ENCODING", "gb18030")
if ENC.lower() == "auto":
    ENC = "gb18030"
DEFAULT_BIN = os.environ.get(
    "GRIDFM_PSASP_BIN",
    r"E:\DISK1中国电力科学研究院有限公司7.95.07.10758冬滚正式版\PSASP",
)


def _suppress_dialogs():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        except Exception:
            pass


def _no_window_kwargs() -> dict:
    """控制台型子进程(WMLFRTMsg / taskkill)的窗口抑制。

    CREATE_NO_WINDOW = 不分配控制台窗口;STARTF_USESHOWWINDOW+SW_HIDE = nCmdShow 传隐藏。
    **对 wmudrt 无效**(MFC GUI 自己 ShowWindow 覆盖 nCmdShow,实测仍弹——见
    _hidden_desktop_popen)。只改窗口显示,不改进程/文件/数值行为。
    GRIDFM_PSASP_SHOW_WINDOW=1 临时放开(调试用)。
    """
    if sys.platform != "win32" or os.environ.get("GRIDFM_PSASP_SHOW_WINDOW") == "1":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0                      # SW_HIDE
    return {"startupinfo": si,
            "creationflags": subprocess.CREATE_NO_WINDOW}


# ---------------- 隐藏桌面(wmudrt 窗口的唯一有效解) ----------------
# 2026-07-14 用户诉求:批产时暂态引擎窗口逐例闪烁妨碍前台工作。
# 实测:SW_HIDE + CREATE_NO_WINDOW **无效**(wmudrt 是 MFC GUI,主窗口自己
# ShowWindow(SW_SHOW),不理会 nCmdShow;探测到 72/214 采样可见窗口)。
# 有效解 = **窗口站内另开一个桌面**,把引擎进程的 STARTUPINFOW.lpDesktop 指过去:
# 它的窗口创建在那个桌面上,当前桌面根本枚举不到,也就无从闪烁。
# 进程/文件/数值行为完全不变(引擎照常算、照常写 STANA/STOUT/ST.CAL;收尸按 PID 不变)。
# subprocess.STARTUPINFO 不暴露 lpDesktop → 只能走 ctypes CreateProcessW。
_HIDDEN_DESKTOP_NAME = "gridfm_psasp_hidden"
_desktop_handle = None
_desktop_warned = False

_STILL_ACTIVE = 259
_CREATE_NO_WINDOW = 0x08000000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_STARTF_USESHOWWINDOW = 0x00000001
_PROCESS_TERMINATE = 0x0001


if sys.platform == "win32":               # 结构体在类体求值,非 Windows 不得定义

    class _StartupInfoW(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                    ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                    ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                    ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                    ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                    ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                    ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                    ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
                    ("hStdError", wintypes.HANDLE)]

    class _ProcessInformation(ctypes.Structure):
        _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                    ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


class _DesktopProc:
    """Popen 最小兼容面(_kill_wmudrt 只用 pid/poll/kill)。"""

    def __init__(self, h_process, h_thread, pid: int) -> None:
        self._h, self.pid = h_process, pid
        ctypes.windll.kernel32.CloseHandle(h_thread)
        self._code = None

    def poll(self):
        if self._code is not None:
            return self._code
        code = wintypes.DWORD()
        if ctypes.windll.kernel32.GetExitCodeProcess(self._h, ctypes.byref(code)):
            if code.value != _STILL_ACTIVE:
                self._code = code.value
        return self._code

    def kill(self) -> None:
        try:
            ctypes.windll.kernel32.TerminateProcess(self._h, 1)
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            ctypes.windll.kernel32.CloseHandle(self._h)
        except Exception:  # noqa: BLE001
            pass


def _ensure_hidden_desktop():
    """进程内单例;句柄必须一直持有,关掉桌面就没了。失败返回 None(退回可见窗口)。"""
    global _desktop_handle, _desktop_warned
    if _desktop_handle is not None:
        return _desktop_handle
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.CreateDesktopW.restype = wintypes.HANDLE
        h = user32.CreateDesktopW(_HIDDEN_DESKTOP_NAME, None, None, 0,
                                  0x10000000,     # GENERIC_ALL
                                  None)
        if not h:
            raise OSError(ctypes.get_last_error())
        _desktop_handle = h
        return h
    except Exception as exc:  # noqa: BLE001
        if not _desktop_warned:
            print(f"[psasp] 隐藏桌面创建失败({exc}),暂态引擎窗口将可见;不影响计算",
                  flush=True)
            _desktop_warned = True
        return None


def _env_block(env: dict) -> ctypes.Array:
    items = "".join(f"{k}={v}\0" for k, v in env.items()) + "\0"
    return ctypes.create_unicode_buffer(items)


def _hidden_desktop_popen(exe: str, cwd: str, env: dict):
    """在隐藏桌面上起引擎进程;桌面不可用时退回普通 Popen(窗口可见但计算不受影响)。"""
    if sys.platform != "win32" or os.environ.get("GRIDFM_PSASP_SHOW_WINDOW") == "1" \
            or _ensure_hidden_desktop() is None:
        return subprocess.Popen([exe], cwd=cwd, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                **_no_window_kwargs())
    si = _StartupInfoW()
    si.cb = ctypes.sizeof(_StartupInfoW)
    si.lpDesktop = _HIDDEN_DESKTOP_NAME       # ← 窗口去别的桌面创建,本桌面看不见
    si.dwFlags = _STARTF_USESHOWWINDOW
    si.wShowWindow = 0                        # SW_HIDE(GUI 忽略也无妨,桌面隔离才是主力)
    pi = _ProcessInformation()
    cmdline = ctypes.create_unicode_buffer(f'"{exe}"')
    ok = ctypes.windll.kernel32.CreateProcessW(
        None, cmdline, None, None, False,
        _CREATE_NO_WINDOW | _CREATE_UNICODE_ENVIRONMENT,
        _env_block(env), cwd, ctypes.byref(si), ctypes.byref(pi))
    if not ok:
        err = ctypes.get_last_error()
        print(f"[psasp] CreateProcessW(隐藏桌面)失败 err={err},退回可见窗口", flush=True)
        return subprocess.Popen([exe], cwd=cwd, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                **_no_window_kwargs())
    return _DesktopProc(pi.hProcess, pi.hThread, int(pi.dwProcessId))


def _env(psasp_bin: str, omp: str = "2") -> dict:
    env = os.environ.copy()
    env["PATH"] = psasp_bin + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = omp
    env["KMP_AFFINITY"] = "disabled"
    return env


def _license_dead(text: str) -> bool:
    return "slm_init error" in (text or "")


class LicenseError(RuntimeError):
    """加密狗失效——必须熔断整批(继续=制造假结果,7.0 纪律 8)。"""


# 潮流结果卡 = 引擎每次运行的产物。**解算前必须清场**：donor/底图普遍自带这些卡
# （2026-08-24 实测：lib118、lib_cepri36、合成族 _base_case、MATPOWER 转换件的
# LF.CAL 首字段全是 1 = "收敛"），而 fsutil.robust_copytree 每轮把它们重新拷进
# 工作目录；若本次引擎静默失败，判定就会读到上一手的"收敛"判决。
# 静默失败是真实存在的：实测缺 LF.L1 时引擎不写 LF.CAL，但 **returncode 仍为 0**
# （收敛 / 不收敛 / 根本没跑成，三种情形返回码都是 0）→ 返回码无法识别失败，
# 清场是唯一有效的防御。来源：C 线代码人工核对第一条发现。
PF_OUTPUT_FILES = ("LF.CAL", "LF.LP1", "LF.LP2", "LF.LP3", "LF.LP5", "LF.LP6", "LF.P1D")
# 解算前默认清除集 = 全集**去掉 LF.P1D**。理由：LF.P1D 不参与收敛判定
# （判定只看 LF.CAL 与 LF.LP1），清它对本修复零贡献；而它是 L0 Meq=1 批处理
# 热启动的**输入**——实测教训：热启动路径若在拷入 P1D 之后再被清，
# ACTIVSg70k 会退回平启动 120/120 全发散
# （见 大算例样本线-20260820/scripts/viol_evalgen_mp.py 2026-08-21 15:55 条）。
# 需要连 P1D 一起清的调用方（如每抽重置热启动基准）自行调
# clear_pf_outputs(dir, files=PF_OUTPUT_FILES)。
PF_VERDICT_FILES = tuple(f for f in PF_OUTPUT_FILES if f != "LF.P1D")


def clear_pf_outputs(case_dir: Path, keep: Sequence[str] = (),
                     files: Sequence[str] = PF_VERDICT_FILES) -> None:
    """删除工作目录内的潮流结果卡（解算前调用；不碰 donor 源目录）。

    files：待清文件集，默认 PF_VERDICT_FILES（不含 LF.P1D，见上方说明）。
    keep：在 files 基础上额外保留的文件名。
    """
    case_dir = Path(case_dir)
    for name in files:
        if name in keep:
            continue
        (case_dir / name).unlink(missing_ok=True)


class PsaspSolver(SolverRunner):
    def __init__(self, psasp_bin: str = DEFAULT_BIN, omp: str = "2") -> None:
        self.bin = psasp_bin
        self.omp = omp
        # 2026-08-21 Linux 部署补丁:可执行文件名可配置(Linux 版通常无 .exe 后缀)
        self.lf_exe = Path(psasp_bin) / os.environ.get(
            "GRIDFM_PSASP_LF_NAME", "WMLFRTMsg.exe")
        self.st_exe = Path(psasp_bin) / os.environ.get(
            "GRIDFM_PSASP_ST_NAME", "wmudrt.exe")

    # ---------------- 潮流 ----------------
    def run_powerflow(self, case_dir: Path, overrides: Optional[dict] = None,
                      timeout: int = 120, clear_stale: bool = True,
                      keep_outputs: Sequence[str] = ()) -> PFResult:
        case_dir = Path(case_dir)
        if not self.lf_exe.exists():
            raise FileNotFoundError(self.lf_exe)
        if not (case_dir / "LF.L0").exists():
            raise FileNotFoundError(f"{case_dir} 缺 LF.L0(工作目录必须是数据目录)")
        if clear_stale:                      # 见 PF_OUTPUT_FILES 处的说明
            clear_pf_outputs(case_dir, keep=keep_outputs)
        _suppress_dialogs()
        proc = subprocess.run([str(self.lf_exe)], cwd=str(case_dir), env=_env(self.bin, self.omp),
                              capture_output=True, text=True, encoding=ENC, errors="replace",
                              timeout=timeout, **_no_window_kwargs())
        out = (proc.stdout or "") + (proc.stderr or "")
        if _license_dead(out):
            raise LicenseError("slm_init error:加密狗不可用")
        conv = None
        if (case_dir / "LF.CAL").exists():
            conv = cards.parse_cal_converged(Card.load(case_dir / "LF.CAL"))
        bus = cards.parse_lp1(Card.load(case_dir / "LF.LP1")) if (case_dir / "LF.LP1").exists() else None
        # 清场后 LF.CAL 必为本次产物（实测：收敛写 1、不收敛写 -1、引擎没跑成则不写）。
        # 无 LF.CAL = 引擎未产出判决 → 判不收敛。**不能用"有 LP1 就算收敛"兜底**：
        # 发散时引擎照样写满 LP1（实测 ACTIVSg70k 不收敛也写 70,001 行，含溢出值 v≈9997.8）。
        # clear_stale=False（显式不清场）时保留旧兜底，不改历史调用行为。
        if conv is not None:
            converged = bool(conv)
        elif clear_stale:
            converged = False
        else:
            converged = bus is not None and len(bus) > 0
        return PFResult(
            converged=converged, iterations=0, bus_voltage=bus,
            branch_flow=cards.parse_lp2(Card.load(case_dir / "LF.LP2")) if (case_dir / "LF.LP2").exists() else None,
            gen_output=cards.parse_lp5(Card.load(case_dir / "LF.LP5")) if (case_dir / "LF.LP5").exists() else None,
            raw_dir=case_dir,
        )

    # ---------------- 暂态 ----------------
    def _poll_file_stable(self, path: Path, deadline: float, stable_checks: int = 3) -> bool:
        last, stable = -1, 0
        while time.time() < deadline:
            if path.exists():
                sz = path.stat().st_size
                if sz > 0 and sz == last:
                    stable += 1
                    if stable >= stable_checks:
                        return True
                else:
                    stable = 0
                last = sz
            time.sleep(0.3)
        return path.exists() and path.stat().st_size > 0

    def _poll_done(self, case_dir: Path, deadline: float, settle_s: float = 1.0):
        """暂态完成信号(2026-07-11 评审修订,勿改回 OR 口径):

        - 工程配 ST.SME(会产 STANA)→ **必须等 STANA.DAT 自身尺寸稳定**:纪律 §7.0-5
          记载 wmudrt 进程 ~2.3s 返回而 STANA 大算例后台续写至 ~10s,ST.CAL 先稳就收尸
          会截断 STANA,半截极值序列 → 多摆失稳被判"稳"(评审 critical);随后再给
          ST.CAL 一个宽限窗(避免反向竞态把可行运行误标 infeasible)。
        - 工程无 ST.SME(如 118 fixture,不产 STANA)→ 等 ST.CAL 稳定
          (只等 STANA 会白烧整个 timeout,2026-07-11 实测)。
        - kill 前统一加 settle 缓冲(psasp_adapter settle_s=1.0 实网经验)。"""
        sme = case_dir / "ST.SME"
        expect_stana = sme.exists() and sme.stat().st_size > 0
        primary = case_dir / ("STANA.DAT" if expect_stana else "ST.CAL")
        done = self._poll_file_stable(primary, deadline)
        if done and expect_stana:
            # STOUT.DDD 与 STANA 同为后台续写,列宽随 Type=2 注入涨 5~9×(阶段 2:
            # 45/82 列),STANA 先稳不担保 STOUT 收全;截断轨迹的"末 1 s / 故障后 10s"
            # 落在错误时刻(judge_files_v1 有行数校验兜底弃用,但白丢精确口径)。
            # 2026-07-16:上限 10s→60s(文件已稳时轮询立即退出,零边际成本;上限只在
            # 引擎仍在续写大 STOUT 时才吃满)。注:此前怀疑的"118 STOUT 被 10s 上限
            # 截断"经复算证伪——旧探针 1.1MB 文件 = 1204 行 × 119 列**完整**(Type=1
            # 被 118 工程无视,见 stout.build_sme_from_scratch);真拦路虎是 STANA
            # 列布局二分(stana.parse_stana)。上限扩容仅作 138 列(1.3MB)后的保险。
            self._poll_file_stable(case_dir / "STOUT.DDD",
                                   min(deadline, time.time() + 60.0))
            self._poll_file_stable(case_dir / "ST.CAL", min(deadline, time.time() + 5.0))
        if done:
            time.sleep(min(settle_s, max(0.0, deadline - time.time())))
        return done

    def _kill_wmudrt(self, proc):
        """按 PID 杀进程树(收割 wmudrt 弹出的模态窗口子进程,psasp_adapter 实网经验)。

        **2026-07-14 用户裁决 A:删除 `taskkill /F /IM wmudrt.exe` 镜像名兜底。**
        该兜底写于串行时代("串行执行,安全"),ts_labels 上 workers=2 后变成
        "杀掉本机全部引擎"——worker A 收尾时连根拔掉 worker B 在算的例子,B 无
        ST.CAL → 伪 infeasible(三批串行复核 100% 翻案:58/58、391/391、154 例;
        cepri36 伪率 ~18% ≫ ieee39 ~3% 与"单例慢=重叠窗大"排序一致)。
        前置实验(裁决要求):PID 树杀 7/7 轮零残留(5 算完 + 2 中途;wmudrt 派生
        1 个子进程,/T 树杀恰好收净)——见变更记录 07-14 晚条。
        """
        try:
            if proc.poll() is None:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=15, **_no_window_kwargs())
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    def run_transient_file(self, case_dir: Path, fault_s11: Path, timeout: int = 120) -> TSResult:
        """用给定 ST.S11 文件跑一次暂态。Popen + 轮询 + kill,避免窗口卡死。"""
        case_dir = Path(case_dir)
        if not self.st_exe.exists():
            raise FileNotFoundError(self.st_exe)
        if Path(fault_s11).resolve() != (case_dir / "ST.S11").resolve():
            shutil.copy2(fault_s11, case_dir / "ST.S11")
        # **产物清场**(STOUT.DDD 于 2026-07-16 对抗审查补入):search_cct 在**同一个**
        # case_dir 上反复仿真(8 次二分 + 2 次 verify),旧 STOUT 不删则本次若未重写,
        # judge 会解出**上一次仿真的轨迹**——行数与 STANA 恰好相同(同窗同步长),
        # judge_files_v1 的行数自洽守卫抓不住,静默用错轨迹判精确口径。
        for f in ("ST.CAL", "STANA.DAT", "STOUT.DDD", "STERR.LIS", "STERR"):
            try:
                (case_dir / f).unlink()
            except FileNotFoundError:
                pass
        _suppress_dialogs()
        proc = _hidden_desktop_popen(str(self.st_exe), str(case_dir),
                                     _env(self.bin, self.omp))
        deadline = time.time() + timeout
        stana_ok = self._poll_done(case_dir, deadline)
        self._kill_wmudrt(proc)

        # 初始化致命失败识别
        init_infeasible = False
        for name in ("STERR.LIS", "STERR"):
            p = case_dir / name
            if p.exists():
                txt = p.read_text(encoding=ENC, errors="replace")
                if _license_dead(txt):
                    raise LicenseError("slm_init error")
                if "initialization warning" in txt.lower():
                    init_infeasible = True
        st_cal = case_dir / "ST.CAL"
        feasible = st_cal.exists() and not init_infeasible
        stable, detail = self._verdict(case_dir) if feasible else (False, {"reason": "infeasible"})
        return TSResult(feasible=feasible, stable=stable, verdict_detail=detail,
                        extrema={"stana_ok": stana_ok}, traj_dir=case_dir)

    def run_transient(self, case_dir: Path, fault: FaultSpec, t_end: float = 10.0) -> TSResult:
        """由 FaultSpec 写 ST.S11(在既有卡上改切除时刻等)再跑。

        t_end 落 ST.S0 第 14 字段(2026-07-12 前为死参数,时长由算例卡默认值决定——
        cepri36 的 5s 窗曾截断慢发展失稳,见 set_s0_duration)。"""
        case_dir = Path(case_dir)
        s11 = case_dir / "ST.S11"
        write_fault_card(s11, fault)
        set_s0_duration(case_dir, t_end)
        return self.run_transient_file(case_dir, s11)

    @staticmethod
    def _fault_clear_time(case_dir: Path) -> float:
        """低压判据起算时刻 = ST.S11 有效记录的最大有限 Te(999.9 永久故障哨兵除外——
        永久故障期间的持续低压是真失效,不豁免)。无卡/全永久 → 0.0(不豁免)。"""
        s11 = case_dir / "ST.S11"
        if not s11.exists() or not s11.read_bytes().strip():
            return 0.0
        tes = []
        try:
            for rec in Card.load(s11).records:
                fv = cards.FaultView(rec)
                if rec.get_int(cards.FaultView.MARK) != 1:
                    continue
                te = fv.rec.get_float(cards.FaultView.TE)
                if te < 900.0:
                    tes.append(te)
        except Exception:  # noqa: BLE001  卡损坏时不豁免(保守方向)
            return 0.0
        return max(tes) if tes else 0.0

    @staticmethod
    def _n_gen_case(case_dir: Path):
        """算例真实机组数(LF.LP5 行数)= 机端/功角覆盖门的基准。

        不能拿 SME 功角列机数当基准——功角记录来自算例 GUI 模板,本身可能缺机
        (无人校验),那样"9/9 全覆盖"会被静默认可(2026-07-13 审查)。"""
        lp5 = Path(case_dir) / "LF.LP5"
        if not lp5.exists():
            return None
        try:
            return len(cards.parse_lp5(Card.load(lp5)))
        except Exception:  # noqa: BLE001  拿不到基准 → 退回无基准(不阻塞判稳)
            return None

    @staticmethod
    def _fault_start_time(case_dir: Path) -> float:
        """故障起始时刻 = ST.S11 有效记录的最小 Ts(v1 电压恢复分句锚点)。无卡 → 0.0。"""
        s11 = case_dir / "ST.S11"
        if not s11.exists() or not s11.read_bytes().strip():
            return 0.0
        tss = []
        try:
            for rec in Card.load(s11).records:
                if rec.get_int(cards.FaultView.MARK) != 1:
                    continue
                tss.append(rec.get_float(cards.FaultView.TS))
        except Exception:  # noqa: BLE001
            return 0.0
        return min(tss) if tss else 0.0

    def _verdict(self, case_dir: Path):
        """判稳 v1 判据(2026-07-12 用户裁决:功角 500° 任意时刻 + 电压 0.8@故障后10s/
        末态 0.9 + 频率 49/51Hz;阶段 2 起经 judge_files_v1 叠加 COI 频率与机端/逐母线
        精确电压口径,解析与列语义注记见 gridfm/psasp_io/stana.py、stout.py)。

        STANA 缺失(工程无 ST.SME)时回退旧 ST.CAL k 值口径并标注 legacy——本环境实测
        该口径无效(ST.CAL 尾行恒 `1, 0.00000, -999`,有无故障逐字节相同),仅作占位。"""
        stana = case_dir / "STANA.DAT"
        if stana.exists() and stana.stat().st_size > 0:
            from ..psasp_io.stana import judge_files_v1
            det = judge_files_v1(stana, case_dir / "STOUT.DDD", case_dir / "ST.SME",
                                 fault_ts=self._fault_start_time(case_dir),
                                 n_gen_case=self._n_gen_case(case_dir))
            det["source"] = "stana"
            return det["stable"], det
        cal = case_dir / "ST.CAL"
        try:
            recs = Card.load(cal).records
            k = recs[-1].get_float(1)
            angle_stable = k >= 1.0
            return angle_stable, {"angle_k": k, "angle_stable": angle_stable,
                                  "source": "st_cal_legacy_unreliable"}
        except Exception as e:  # noqa: BLE001
            return False, {"reason": f"cal parse fail: {e}", "source": "st_cal_legacy_unreliable"}

    # ---------------- CCT 二分(方案 7.3)----------------
    def search_cct(self, case_dir: Path, fault: FaultSpec, lo: float = 0.0, hi: float = 0.5,
                   tol: float = 0.01, max_sims: int = 8,
                   t_end: float = 10.0) -> CCTResult:
        """二分**故障持续时间**(切除时长 d = Te − Ts):稳定则增大、失稳则减小。
        返回 cct_value = 临界持续时间(切除时刻 = fault.ts + cct);上界仍稳 → 删失。

        **语义定钉(2026-07-13 审查修复)**:此前二分变量直接写进 `te`(= 切除**时刻**),
        在 ts=0 的单测里与"持续时间"恰好等价,故一直未暴露;而生产口径 ts=0.1
        (ts_seeds/ts_labels 的 TS),CCT 会整体偏移 +0.1s 并与"切除时长档 DURS"口径打架。
        CCT(critical clearing time)= 故障从发生到切除的最长可稳持续时间,与 7.3 的
        "二分区间 [0, t_max]"、与 DURS 档同口径。

        infeasible(初始化失败/超时无产物)与切除时长无关(故障前初始化),一律中止并
        单独打标——混入失稳分支会产出 cct=0.0 且非删失的伪标签(2026-07-11 评审;7.3
        纪律"初始化失败样本单独打 infeasible 标,不入稳/失稳统计")。

        t_end 必须与判稳批同窗(阶段 2 = 12s):判据含"故障后 10s ≥0.8"分句,10s 窗下
        该分句不可评,CCT 会按更松的判据搜出偏大的值(2026-07-13)。"""
        case_dir = Path(case_dir)
        n = 0
        # 先测上界:hi 处稳定 → 删失
        n += 1
        res = self.run_transient(case_dir, _with_dur(fault, hi), t_end=t_end)
        if not res.feasible:
            return CCTResult(cct_value=float("nan"), is_censored=False, n_sims=n, infeasible=True)
        if res.stable:
            return CCTResult(cct_value=hi, is_censored=True, n_sims=n)
        last_stable = lo
        any_stable = False              # 是否**真的仿真到过**一个稳定点(下侧删失判据)
        while hi - lo > tol and n < max_sims:
            mid = (lo + hi) / 2
            n += 1
            res = self.run_transient(case_dir, _with_dur(fault, mid), t_end=t_end)
            if not res.feasible:
                return CCTResult(cct_value=float("nan"), is_censored=False, n_sims=n,
                                 infeasible=True)
            if res.stable:
                last_stable, lo, any_stable = mid, mid, True
            else:
                hi = mid
        # **不得四舍五入**(2026-07-27 实测修复):CCT 的定义 = *实测稳定过*的最长持续
        # 时间,`round(x, 3)` 有一半的二分网格点会**向上**取整 → 返回一个从未跑过、且可能
        # 已越过临界点的值,把标签写错。实测指纹:归档 ieee118 的 44 条双验失败**全部且
        # 仅仅**是 `cct=0.055`(该档 44/44 失败,其余 18 个取值 106/106 全过),失败模式
        # 全为单侧 `te_cct_stable=False`。探针(`scripts/cct_verify_probe.py`,2 例 ×4 次
        # 仿真)钉死机制:dur=0.0546875(=7/128,二分实测**稳**)→ round 后 0.055 → **失稳**,
        # 0.054 稳 / 0.056 失稳 —— 引擎把清除时刻量化到 0.01s 网格,而 0.055 恰落在 0.005
        # 平分点上,舍入把它推过了网格线。返回未舍入值 = "最大实测稳定时长",与 DoD 双验
        # (Te=cct 必稳)自洽。**旧标签不追改**(归档只读);消费方若要捞回那 44 条,见
        # `scripts/cct_repair_rounding.py`。
        # **下侧删失**:一个 mid 都没稳过 → `last_stable` 还是没跑过的 `lo`,不是测量值。
        # 判据必须用 `any_stable`,**不能**用 `last_stable == lo` —— 二分里 `last_stable`
        # 与 `lo` 是同步赋值的,正常情形结束时二者恒等,那样写会把每一行都误标成删失。
        if not any_stable:
            return CCTResult(cct_value=lo, is_censored=False, n_sims=n,
                             is_censored_low=True)
        return CCTResult(cct_value=last_stable, is_censored=False, n_sims=n)

    def batch_run(self, jobs: List[Job], n_workers: int = 1, serial: bool = True) -> List[object]:
        # 暂态/标签级一律串行(7.0 纪律 7);潮流可并行(此处 v0 串行,后续加进程池)
        out: List[object] = []
        for j in jobs:
            if j.kind == "pf":
                out.append(self.run_powerflow(j.case_dir, j.overrides))
            elif j.kind == "ts":
                out.append(self.run_transient(j.case_dir, j.fault))
            elif j.kind == "cct":
                out.append(self.search_cct(j.case_dir, j.fault))
        return out


def _with_dur(fault: FaultSpec, dur: float) -> FaultSpec:
    """按**故障持续时间**改写切除时刻:Te = Ts + dur(CCT 二分的唯一自由量)。"""
    from dataclasses import replace
    return replace(fault, te=round(fault.ts + dur, 6))


def set_s0_duration(case_dir: Path, t_end: float) -> None:
    """ST.S0 第 14 字段(0 起 idx=13)= 暂态仿真时长(s)。

    实测锚:cepri36_golden=5 / ieee39_golden=10,与 STANA 窗长逐例一致(2026-07-12)。
    背景:run_transient 的 t_end 参数曾是死参数(时长由算例 ST.S0 决定),cepri36 的
    5s 窗截断慢发展失稳 → 误判稳(实例 cepri36_00044_L9-8-32_d200:窗末 v=0.284 仍跌、
    角差 189° 仍涨;变更记录 07-12)。v0 只支持整数秒(引擎卡为整数格,防字节格式漂移)。
    """
    if abs(t_end - round(t_end)) > 1e-9:
        raise ValueError(f"仿真时长 v0 只支持整数秒,得到 {t_end}")
    s0 = Path(case_dir) / "ST.S0"
    if not s0.exists():
        raise FileNotFoundError(f"{s0} 不存在:无法设置仿真时长")
    card = Card.load(s0)
    card.records[0].set_int(13, int(round(t_end)))
    card.save(s0)


S0_NDIST_IDX = 8      # ST.S0 第 9 字段 Ndist(扰动个数;字段表 数据文件格式 V3.8 ST.S0.txt)


def sync_s0_nfault(case_dir: Path, n: int) -> None:
    """同步 ST.S0 第 8 字段 Nfault(故障个数;字段表见数据文件格式 V3.8 ST.S0.txt)。

    引擎按此计数读 ST.S11——计数为 0 时卡内容被**静默忽略**(CEPRI-36 实算:四变体
    扫描 STANA 逐字节不变,补计数后三相短路 vmin 0.0001;变更记录 2026-07-11)。
    """
    s0 = Path(case_dir) / "ST.S0"
    if not s0.exists():
        raise FileNotFoundError(f"{s0} 不存在:无法同步 Nfault(引擎按计数读故障卡)")
    card = Card.load(s0)
    card.records[0].set_int(7, n)
    card.save(s0)


def blank_s12(case_dir: Path) -> bool:
    """删除交付模板的 S12 切机扰动卡:ST.S12 置空 + ST.S0 第 9 字段 Ndist(0 起 idx=8)
    计数清零。返回是否实际动过卡(无 S12 且 Ndist=0 → False,幂等空操作)。

    **生产协议常设决定**(2026-07-24 用户裁定,判稳线批产适用;2026-07-25 入主产线):
    118 交付模板预置 S12 切机卡(母线 119,t=10s)须在判稳批产时删除。
    计数必须同步清零——引擎按 ST.S0 计数读扰动卡(Nfault 同款教训见 sync_s0_nfault:
    计数与卡不同步时行为未定义)。Ndist 与 S12 的对应为实测:118 模板 Ndist=1 且
    ST.S12 单条记录,cepri36 无 ST.S12 且 Ndist=0。

    删卡的**理由是"无害且简化模板"而非数据修正**:勘误实测(docs/勘误-S12判稳标签-
    20260724.md)该卡在全 118 粮仓 22,155 条轨迹中从未动作,删卡对照重仿翻案 0/453。
    """
    case_dir = Path(case_dir)
    s12 = case_dir / "ST.S12"
    touched = False
    if s12.exists() and s12.read_bytes().strip():
        s12.write_bytes(b"")
        touched = True
    s0 = case_dir / "ST.S0"
    if not s0.exists():
        raise FileNotFoundError(f"{s0} 不存在:无法清零 Ndist(引擎按计数读扰动卡)")
    card = Card.load(s0)
    if card.records[0].get_int(S0_NDIST_IDX) != 0:
        card.records[0].set_int(S0_NDIST_IDX, 0)
        card.save(s0)
        touched = True
    return touched


def _fault_point_bus_name(case_dir: Path, fault: FaultSpec) -> str:
    """故障点母线命名:照 39 GUI 模板版式『{线路基名}_{K 整数}_{K 两位小数}』
    (线 34-28-23 K=20.00 → 'AC23_20_00';基名 = LF.L2 尾字段 IDName 去 '_ac')。

    引擎只认 ST.S11.Add_No 行号、不校验名字语义(118 模板 Add_No=122 名 'AC155_10'
    与卡内线 184/K=50 名实不符仍可用,2026-07-15 解剖),名字的职责只有两条:
    在 LF.L1/ST.S1 里唯一(Bs_Name 是主键)+ 对人可读。同线不同 K 名字天然不同。"""
    l2 = case_dir / "LF.L2"
    if not l2.exists():
        raise FileNotFoundError(f"{l2} 不存在:无法定位故障线路命名故障点母线")
    base = None
    for rec in Card.load(l2).records:
        try:
            ij = (rec.get_int(1), rec.get_int(2))
            if rec.get_int(3) != fault.id_no or set(ij) != {fault.i_no, fault.j_no}:
                continue
            name = cards.AclineView(rec, 0).idname
        except (ValueError, IndexError):
            continue  # 并补等异版式行,跳过
        base = name[:-3] if name.endswith("_ac") else (name or f"AC{fault.id_no}")
        break
    if base is None:
        raise ValueError(f"LF.L2 找不到支路 ({fault.i_no},{fault.j_no},{fault.id_no}):"
                         "K%≠0 故障必须落在既有交流线上")
    k_int = int(fault.k_pct)
    k_frac = int(round((fault.k_pct - k_int) * 100))
    return f"{base}_{k_int}_{k_frac:02d}"


def ensure_fault_point_bus(case_dir: Path, fault: FaultSpec) -> int:
    """K%≠0 线路中间故障的故障点母线注册(T4.1 完整范围),返回其计算母线编号。

    解剖依据(2026-07-15,ieee39 模板母线 43 + ieee118 模板母线 122 双参照):
    故障点母线只落笔四处——①ST.S1 末行追加 `'名字',`(行号即编号,ST.S1.txt:
    "该文件要首先形成");②LF.L1 末行追加同名占位行(名字补 20 宽、Vbase=0、
    **无第 8 字段 Bus_Name**,两模板一致);③ST.S11.Add_No=该行号(调用方写);
    ④ST.S0 首字段 Nbus 同步为新总数。LF.L2/ST.S2 不拆线(引擎按 K% 内部拆分),
    LF.L0 NBB、LF.BUS、PHYBUS.LFI 均不动;潮流侧无该母线解(build_sample 走
    STATE missing 分支,行号对齐"末尾追加"天然稳定)。

    幂等:同名(同线同 K)已注册则复用行号——CCT 二分对同一 case_dir 反复写卡。
    """
    if not (0.0 < fault.k_pct < 100.0):
        raise ValueError(f"故障点母线仅支持 0 < K% < 100(请求 {fault.k_pct});"
                         "端部故障请用 K%=0(I 侧)或对侧线端 K%=0")
    case_dir = Path(case_dir)
    name = _fault_point_bus_name(case_dir, fault)
    s1_path, l1_path, s0_path = (case_dir / n for n in ("ST.S1", "LF.L1", "ST.S0"))
    for p in (s1_path, l1_path, s0_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} 不存在:K%≠0 需在完整算例目录上注册故障点母线")
    s1_recs = Card.load(s1_path).records
    for i, rec in enumerate(s1_recs):
        if rec.get_str(0) == name:
            return i + 1  # 已注册,幂等复用(行号从 1 起 = 计算母线编号)
    n_l1 = len(Card.load(l1_path).records)
    s0 = Card.load(s0_path)
    if len(s1_recs) != n_l1 or s0.records[0].get_int(0) != n_l1:
        raise ValueError(f"算例母线计数不自洽(ST.S1={len(s1_recs)} LF.L1={n_l1} "
                         f"ST.S0.Nbus={s0.records[0].get_int(0)}),拒绝追加故障点母线")

    def _append_line(path: Path, line: str) -> None:
        b = path.read_bytes()
        if b and not b.endswith(b"\n"):
            b += b"\r\n"  # 模板末行可能无换行(GUI 追加行是裸 \n,lowlevel 兼容混合行尾)
        path.write_bytes(b + line.encode(ENC))

    _append_line(s1_path, f" '{name}',\r\n")
    _append_line(l1_path, f" '{name:<20}',    0.0000,   0,    0.0000,    0.0000,"
                          f"    0.0000,    0.0000,\r\n")
    new_no = n_l1 + 1
    s0.records[0].set_int(0, new_no)
    s0.save(s0_path)
    return new_no


def write_fault_card(s11_path: Path, fault: FaultSpec) -> None:
    """FaultSpec → ST.S11(方案 4.2 / T4.1)+ 同步 ST.S0 Nfault 计数。

    K%≠0(线路中间故障)先经 ensure_fault_point_bus 注册故障点母线,把编号写进
    Add_No;K%=0 时 Add_No 显式写 0——修复潜伏缺陷:复用路径此前不碰 Add_No,
    在 118 模板(交付卡自带 K=50/Add_No=122)上改写故障会残留陈旧故障点母线指向。
    既有非空卡:沿用首条记录结构改写字段。空卡/无卡:按 ST.S11.txt 字段表
    (数据文件格式 V3.8,17 字段 Mark..X+Add_Name_2)从零写出。
    """
    s11_path = Path(s11_path)
    add_no = 0
    if abs(fault.k_pct) > 1e-9:
        add_no = ensure_fault_point_bus(s11_path.parent, fault)
    if s11_path.exists() and s11_path.read_bytes().strip():
        card = Card.load(s11_path)
        rec = card.records[0]
        F = cards.FaultView
        rec.set_int(F.I, fault.i_no)
        rec.set_int(F.J, fault.j_no)
        rec.set_int(F.ID, fault.id_no)
        # 小数位按 ST.S11.txt 字段表显式给定(K% DOUBLE(15,2)、Ts/Te/R/X DOUBLE(15,6))。
        # 不给则 set_float 沿用模板单元格位数——39 模板 Te 只有 1 位小数,te=0.25 会被
        # 写成 0.2,CCT 二分(tol 0.01)整个被量化到 0.1s(2026-07-11 单测抓获)。
        rec.set_float(F.KPCT, fault.k_pct, decimals=2)
        rec.set_int(F.ADD_NO, add_no)
        rec.set_int(F.A, fault.phases[0])
        rec.set_int(F.B, fault.phases[1])
        rec.set_int(F.C, fault.phases[2])
        rec.set_int(F.GROUND, fault.ground)
        rec.set_int(F.SHORT, fault.short)
        rec.set_int(F.OPEN, fault.open_)
        rec.set_float(F.TS, fault.ts, decimals=6)
        rec.set_float(F.TE, fault.te, decimals=6)
        rec.set_float(F.R, fault.r, decimals=6)
        rec.set_float(F.X, fault.x, decimals=6)
        card.save(s11_path)
        n = len(card.records)
    else:
        a, b, c = fault.phases
        row = (f"  1,{fault.i_no:6d},{fault.j_no:6d},{fault.id_no:6d},"
               f"{fault.k_pct:9.2f},{add_no:6d}, {a}, {b}, {c}, {fault.ground},"
               f" {fault.short}, {fault.open_},{fault.ts:15.6f},{fault.te:15.6f},"
               f"{fault.r:15.6f},{fault.x:15.6f},     0,\r\n")
        s11_path.write_bytes(row.encode(ENC))
        n = 1
    sync_s0_nfault(s11_path.parent, n)
