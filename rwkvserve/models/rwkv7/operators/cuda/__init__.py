"""
RWKV7 CUDA Operators - backward compatibility layer.

This module has been refactored; operator loading logic is now in wkv_loader.py.

New interface:
    from rwkvserve.models.rwkv7.operators import load_cuda_operator_infctx
    op = load_cuda_operator_infctx()

Migration:
    Old:
        from rwkvserve.models.rwkv7.operators.cuda import load_wkv7_infctx_operator
        op = load_wkv7_infctx_operator()

    New:
        from rwkvserve.models.rwkv7.operators import load_cuda_operator_infctx
        op = load_cuda_operator_infctx()
"""

import warnings

from ..wkv_loader import load_cuda_operator_infctx as _load_cuda_operator_infctx


def load_wkv7_infctx_operator():
    """
    Load infinite-context CUDA operator (backward-compatible, deprecated).

    Use load_cuda_operator_infctx() instead.
    """
    warnings.warn(
        "load_wkv7_infctx_operator() is deprecated. Use:\n"
        "  from rwkvserve.models.rwkv7.operators import load_cuda_operator_infctx",
        DeprecationWarning,
        stacklevel=2
    )
    return _load_cuda_operator_infctx()


wkv7_infctx_op = None


def __getattr__(name):
    """Lazy-load global variables (backward compatibility)."""
    if name == 'wkv7_infctx_op':
        global wkv7_infctx_op
        if wkv7_infctx_op is None:
            warnings.warn(
                "Direct access to wkv7_infctx_op is deprecated. Use load_cuda_operator_infctx()",
                DeprecationWarning,
                stacklevel=2
            )
            wkv7_infctx_op = _load_cuda_operator_infctx()
        return wkv7_infctx_op
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['wkv7_infctx_op', 'load_wkv7_infctx_operator']
