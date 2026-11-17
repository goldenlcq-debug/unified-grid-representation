import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rl_poweradj.run_store import atomic_json


def winerror(code):
    exc=PermissionError('simulated Windows file contention')
    exc.winerror=code
    return exc


class AtomicJsonRetryTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_real_windows_reader_lock(self):
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        kernel.CloseHandle.restype=wintypes.BOOL
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.json';atomic_json(p,{'old':1})
            handle=kernel.CreateFileW(str(p),0x80000000,3,None,3,0x80,None)
            self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
            released=[]
            def release(_):
                self.assertEqual(json.loads(p.read_text()),{'old':1})
                self.assertTrue(kernel.CloseHandle(handle));released.append(True)
            try:
                with patch('rl_poweradj.run_store.time.sleep',side_effect=release) as sleep:
                    atomic_json(p,{'new':2})
                self.assertEqual(sleep.call_count,1)
                self.assertEqual(json.loads(p.read_text()),{'new':2})
            finally:
                if not released:kernel.CloseHandle(handle)

    def test_transient_windows_errors_preserve_old_file_then_succeed(self):
        for code in (5,32,33):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as d:
                p=Path(d)/'run.json';atomic_json(p,{'old':1});real_replace=os.replace;calls=[]
                def replace(src,dst):
                    calls.append(1)
                    self.assertEqual(json.loads(p.read_text()),{'old':1})
                    if len(calls)<3:raise winerror(code)
                    real_replace(src,dst)
                with patch('rl_poweradj.run_store.os.replace',side_effect=replace), patch('rl_poweradj.run_store.time.sleep') as sleep:
                    atomic_json(p,{'new':2})
                self.assertEqual(len(calls),3);self.assertEqual(sleep.call_count,2)
                self.assertEqual(json.loads(p.read_text()),{'new':2});self.assertFalse(list(Path(d).glob('*.tmp')))

    def test_persistent_windows_error_is_bounded_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.json';atomic_json(p,{'old':1})
            with patch('rl_poweradj.run_store.os.replace',side_effect=winerror(5)) as replace, patch('rl_poweradj.run_store.time.sleep') as sleep:
                with self.assertRaises(PermissionError):atomic_json(p,{'new':2})
            self.assertEqual(replace.call_count,8);self.assertEqual(sleep.call_count,7)
            self.assertEqual(json.loads(p.read_text()),{'old':1})
            tmp=list(Path(d).glob('*.tmp'));self.assertEqual(len(tmp),1);self.assertEqual(json.loads(tmp[0].read_text()),{'new':2})

    def test_other_error_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('rl_poweradj.run_store.os.replace',side_effect=OSError('disk error')) as replace, patch('rl_poweradj.run_store.time.sleep') as sleep:
                with self.assertRaises(OSError):atomic_json(Path(d)/'run.json',{})
            self.assertEqual(replace.call_count,1);sleep.assert_not_called()

    def test_normal_write_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.json';atomic_json(p,{'value':1});self.assertEqual(json.loads(p.read_text()),{'value':1})
            with self.assertRaises(ValueError):atomic_json(p,{'bad':float('nan')})
            self.assertEqual(json.loads(p.read_text()),{'value':1})

if __name__=='__main__':unittest.main()
