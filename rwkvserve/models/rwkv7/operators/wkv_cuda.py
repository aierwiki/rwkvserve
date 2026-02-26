"""
Backward compatibility layer - deprecated.

This file has been refactored. Use wkv_loader.py instead.

New interface:
- load_wkv_operator(device='cuda'|'cpu', for_inference=False, head_size=64)
- load_cpu_operator(head_size=64)
- load_cuda_operator_train(head_size=64)
- load_cuda_operator_inference(head_size=64)
- rwkv7_wkv(q, w, k, v, a, b, ...)

Migration:
    Old:
        from rwkvserve.models.rwkv7.operators.wkv_cuda import load_wkv7_operator
        op = load_wkv7_operator(head_size=64, device='cuda')

    New:
        from rwkvserve.models.rwkv7.operators import load_wkv_operator
        op = load_wkv_operator(device='cuda', head_size=64)
"""

import warnings
from typing import Callable

from .wkv_loader import (
    load_wkv_operator as _load_wkv_operator,
    rwkv7_wkv as _rwkv7_wkv,
)


def load_wkv7_operator(head_size: int = 64, device: str = 'cuda', with_state: bool = False) -> Callable:
    """Backward-compatible loader (deprecated). Use load_wkv_operator() instead."""
    warnings.warn(
        "load_wkv7_operator() is deprecated. Use load_wkv_operator(device, for_inference, head_size)",
        DeprecationWarning,
        stacklevel=2
    )
    return _load_wkv_operator(device=device, for_inference=with_state, head_size=head_size)


def load_wkv7_operator_cuda(head_size: int = 64) -> Callable:
    """Backward-compatible CUDA loader (deprecated). Use load_cuda_operator_train() instead."""
    warnings.warn(
        "load_wkv7_operator_cuda() is deprecated. Use load_cuda_operator_train(head_size)",
        DeprecationWarning,
        stacklevel=2
    )
    from .wkv_loader import load_cuda_operator_train
    return load_cuda_operator_train(head_size)


def rwkv7_wkv(*args, **kwargs):
    """Backward-compatible convenience interface, proxies to new implementation."""
    return _rwkv7_wkv(*args, **kwargs)


warnings.warn(
    "wkv_cuda.py is deprecated and has been refactored into wkv_loader.py.\n"
    "Update imports to:\n"
    "  from rwkvserve.models.rwkv7.operators import load_wkv_operator, rwkv7_wkv",
    DeprecationWarning,
    stacklevel=2
)
