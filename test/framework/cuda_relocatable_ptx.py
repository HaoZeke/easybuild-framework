"""CUDA architecture inspection distinguishes executable and relocatable fatbins."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import easybuild.tools.systemtools as systemtools


ELF = "Fatbin elf code:\n================\narch = sm_80\n"
RELOCATABLE = "Fatbin elf code:\n================\narch = sm_95\n"
PTX = "Fatbin ptx code:\n================\narch = sm_90\n"


class CudaRelocatablePtxTest(unittest.TestCase):
    def invoke(self, function, *args, all_output=ELF + RELOCATABLE + PTX, **kwargs):
        calls = []

        def command(cmd, **unused):
            calls.append(cmd)
            if cmd == "file /synthetic/program":
                output = "ELF 64-bit LSB executable"
            elif cmd == "cuobjdump /synthetic/program":
                output = ELF
            elif cmd == "cuobjdump --all-fatbin /synthetic/program":
                output = all_output
            else:
                raise AssertionError("Unexpected command: " + cmd)
            return SimpleNamespace(exit_code=0, output=output)

        with patch.object(systemtools, 'run_shell_cmd', side_effect=command), \
                patch.object(systemtools.shutil, 'which', return_value="/synthetic/cuobjdump"):
            result = function("/synthetic/program", *args, **kwargs)
        return result, calls

    def test_raw_default_retains_executable_scope(self):
        result, calls = self.invoke(systemtools.get_cuda_object_dump_raw)
        self.assertEqual(result, ELF)
        self.assertEqual(calls, ["file /synthetic/program", "cuobjdump /synthetic/program"])

    def test_explicit_raw_mode_includes_relocatable_sections(self):
        result, calls = self.invoke(systemtools.get_cuda_object_dump_raw, all_fatbin=True)
        self.assertEqual(result, ELF + RELOCATABLE + PTX)
        self.assertEqual(calls[-1], "cuobjdump --all-fatbin /synthetic/program")

    def test_ptx_architectures_include_relocatable_ptx(self):
        result, calls = self.invoke(systemtools.get_cuda_architectures, "ptx")
        self.assertEqual(result, ["9.0"])
        self.assertEqual(calls[-1], "cuobjdump --all-fatbin /synthetic/program")

    def test_device_architectures_exclude_unlinked_relocatable_elf(self):
        result, calls = self.invoke(systemtools.get_cuda_architectures, "elf")
        self.assertEqual(result, ["8.0"])
        self.assertNotIn("9.5", result)
        self.assertEqual(calls[-1], "cuobjdump /synthetic/program")

    def test_missing_ptx_remains_missing_after_complete_inspection(self):
        result, calls = self.invoke(systemtools.get_cuda_architectures, "ptx", all_output=ELF + RELOCATABLE)
        self.assertIsNone(result)
        self.assertEqual(calls[-1], "cuobjdump --all-fatbin /synthetic/program")

    def test_raw_modes_have_distinct_shell_cache_keys(self):
        default, default_calls = self.invoke(systemtools.get_cuda_object_dump_raw)
        complete, complete_calls = self.invoke(systemtools.get_cuda_object_dump_raw, all_fatbin=True)
        self.assertNotEqual(default, complete)
        self.assertNotEqual(default_calls[-1], complete_calls[-1])


def suite(loader=None):
    """Return the CUDA relocatable PTX regression cases."""
    return (loader or unittest.defaultTestLoader).loadTestsFromTestCase(CudaRelocatablePtxTest)


if __name__ == '__main__':
    unittest.main()
