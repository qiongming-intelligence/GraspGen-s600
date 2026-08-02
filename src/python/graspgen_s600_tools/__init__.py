"""
GraspGen-s600 Tools
===================

Tools for adapting GraspGen to Horizon Sunrise 6 (S600) platform.

Modules:
- export: ONNX export utilities
- convert: Model conversion (precision, optimization)
- runtime: BPU inference adapters
- debug: Validation and debugging tools
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "lvyufeng"

__all__ = ["export", "convert", "runtime", "debug"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
