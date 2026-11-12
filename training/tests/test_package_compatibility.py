"""Renamed package entry points and compatibility paths; no expert requests."""
from importlib import import_module
import subprocess
import sys
import unittest


class PackageCompatibilityTests(unittest.TestCase):
    def test_aliases_share_implementation(self):
        for name in ("rules", "opponents", "env", "collect", "api", "records"):
            with self.subTest(module=name):
                self.assertIs(import_module(f"barricade_lab.{name}"),
                              import_module(f"quoridor_lab.{name}"))
        from quoridor_lab import QuoridorEnv
        from barricade_lab.env import BarricadeEnv
        self.assertIs(QuoridorEnv, BarricadeEnv)

    def test_module_cli_help(self):
        for name in ("quoridor_lab", "quoridor_lab.cli", "barricade_lab", "barricade_lab.cli"):
            with self.subTest(module=name):
                result = subprocess.run([sys.executable, "-m", name, "--help"],
                                        capture_output=True, text=True, check=True)
                self.assertIn("Quoridor Lab", result.stdout)
