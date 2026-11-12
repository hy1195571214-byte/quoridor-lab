"""Compatibility imports for notebooks created before the Quoridor Lab rename."""
from importlib import import_module
import sys

from quoridor_lab import Action, State, ACTION_COUNT, QuoridorEnv
from quoridor_lab.env import BarricadeEnv

__all__ = ["Action", "State", "ACTION_COUNT", "QuoridorEnv", "BarricadeEnv"]

# Share module objects so existing patches, classes and stored references resolve
# to the same implementation used by the canonical package.
for _name in ("rules", "opponents", "env", "collect", "api", "records"):
    _module = import_module(f"quoridor_lab.{_name}")
    globals()[_name] = _module
    sys.modules[f"{__name__}.{_name}"] = _module
