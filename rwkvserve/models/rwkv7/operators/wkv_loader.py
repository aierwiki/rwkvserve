"""
RWKV7 WKV operator loader - unified entry point.

Responsibilities:
- Load the appropriate WKV operator based on device type (CPU/CUDA)
- Manage operator cache to avoid redundant compilation
- Provide a unified interface for upstream callers
- Manage standard training and infinite-context (infctx) operators
"""

import os
import sys
import time
import threading
from contextlib import contextmanager
from typing import Callable, Any
from pathlib import Path


# Global cache
_cpu_operators = {}
_cuda_operator_train = {}
_cuda_operator_inference = {}
_cuda_operator_infctx = None
_load_lock = threading.Lock()


def _get_rank():
    """Get the current process rank."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except Exception:
        return 0
    return 0

def _get_world_size():
    """Get the total number of processes."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1
    return 1

def _barrier():
    """Distributed synchronization barrier."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def _get_cache_dir():
    """Get PyTorch extension cache directory."""
    import torch.utils.cpp_extension as cpp_ext
    cache_dir = os.environ.get('TORCH_EXTENSIONS_DIR')
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch_extensions')
    return Path(cache_dir)


@contextmanager
def _interprocess_build_lock(module_name: str):
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f".{module_name}.lock"
    f = open(lock_path, "a+")
    try:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass

def _check_cache_exists(module_name: str) -> bool:
    """Check if CUDA operator compilation cache exists."""
    try:
        import torch
        cache_dir = _get_cache_dir()

        python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
        cuda_version = torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'

        candidates = [
            cache_dir / module_name,
            cache_dir / f"{python_version}_cu{cuda_version}" / module_name,
        ]

        for module_cache_dir in candidates:
            if module_cache_dir.exists():
                so_files = list(module_cache_dir.rglob('*.so')) + list(module_cache_dir.rglob('*.pyd'))
                if len(so_files) > 0:
                    return True

        return False
    except Exception:
        return False

def _should_print_compile_info() -> bool:
    """Only print compile info on rank 0 or single-GPU mode."""
    return _get_rank() == 0


# ============================================================================
# CPU operator
# ============================================================================

def load_cpu_operator(head_size: int = 64) -> Callable:
    """
    Load the CPU WKV operator (official implementation).

    Args:
        head_size: Attention head size.

    Returns:
        CPU operator function.
    """
    global _cpu_operators

    if head_size in _cpu_operators:
        return _cpu_operators[head_size]

    from .wkv_cpu import RUN_CPU_RWKV7

    print(f"Loading RWKV7 CPU operator (head_size={head_size})")
    print("Warning: CPU operator is 100-1000x slower than GPU")

    def _cpu_operator_wrapper(r, w, k, v, a, b, state=None, return_state=False):
        output, final_state = RUN_CPU_RWKV7(r, w, k, v, a, b, head_size, state=state)
        if return_state:
            return output, final_state
        else:
            return output

    _cpu_operators[head_size] = _cpu_operator_wrapper

    return _cpu_operator_wrapper


# ============================================================================
# CUDA operators
# ============================================================================

def load_cuda_operator_train(head_size: int = 64) -> Callable:
    """
    Load the CUDA WKV operator (training version).

    Features:
    - Supports backpropagation
    - Uses Wind Backstepping algorithm
    - Auto JIT compilation
    - Distributed training support (multi-GPU)
    - Automatic cache detection

    Args:
        head_size: Attention head size.

    Returns:
        CUDA training operator function.
    """
    global _cuda_operator_train

    if head_size in _cuda_operator_train:
        return _cuda_operator_train[head_size]

    with _load_lock:
        if head_size in _cuda_operator_train:
            return _cuda_operator_train[head_size]

        import torch
        from torch.utils.cpp_extension import load
        from typing import cast

        rank = _get_rank()
        world_size = _get_world_size()

        local_cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda')
        module_name = f"wind_backstepping_h{head_size}"

        cache_exists = _check_cache_exists(module_name)

        if rank == 0:
            if cache_exists:
                print(f"Loading RWKV7 CUDA training operator (cached, head_size={head_size})")
            else:
                print(f"First-time RWKV7 CUDA training operator compilation (head_size={head_size})")
                print("  Compiling CUDA kernel... (subsequent loads will use cache)")
        elif not cache_exists:
            print(f"[Rank {rank}/{world_size}] Waiting for rank 0 to compile CUDA operator...")

        CHUNK_LEN = 16

        flags = [
            '-res-usage',
            f'-D_C_={head_size}',
            f'-D_CHUNK_LEN_={CHUNK_LEN}',
            '--use_fast_math',
            '-O3',
            '-Xptxas -O3',
            '--extra-device-vectorization'
        ]

        if rank == 0:
            with _interprocess_build_lock(module_name):
                start_time = time.time()
                wkv_module = load(
                    name="wind_backstepping",
                    sources=[
                        os.path.join(local_cuda_dir, 'wkv7_cuda.cu'),
                        os.path.join(local_cuda_dir, 'wkv7_op.cpp')
                    ],
                    is_python_module=False,
                    verbose=False,
                    extra_cuda_cflags=flags
                )
                load_time = time.time() - start_time
                print(f"Training operator loaded ({load_time:.2f}s)")

        _barrier()

        if rank > 0:
            with _interprocess_build_lock(module_name):
                wkv_module = load(
                    name="wind_backstepping",
                    sources=[
                        os.path.join(local_cuda_dir, 'wkv7_cuda.cu'),
                        os.path.join(local_cuda_dir, 'wkv7_op.cpp')
                    ],
                    is_python_module=False,
                    verbose=False,
                    extra_cuda_cflags=flags
                )
            print(f"[Rank {rank}] Training operator loaded")

        class WindBackstepping(torch.autograd.Function):
            @staticmethod
            def forward(ctx, w, q, k, v, z, b):
                B, T, H, C = w.shape
                assert T % CHUNK_LEN == 0, f"Sequence length {T} must be a multiple of {CHUNK_LEN}"

                w, q, k, v, z, b = [i.to(torch.bfloat16) if i.dtype != torch.bfloat16 else i
                                    for i in [w, q, k, v, z, b]]

                assert all(i.is_contiguous() for i in [w, q, k, v, z, b])

                y = torch.empty_like(v)
                s = torch.empty(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
                sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
                torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
                ctx.save_for_backward(w, q, k, v, z, b, s, sa)
                return y

            @staticmethod
            def backward(ctx, *grad_outputs):
                (dy,) = grad_outputs
                assert dy.dtype == torch.bfloat16
                assert dy.is_contiguous()
                w, q, k, v, z, b, s, sa = ctx.saved_tensors
                dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
                torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
                return dw, dq, dk, dv, dz, db

        def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
            B, T, HC = q.shape
            q = q.to(torch.bfloat16) if q.dtype != torch.bfloat16 else q
            w = w.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w
            k = k.to(torch.bfloat16) if k.dtype != torch.bfloat16 else k
            v = v.to(torch.bfloat16) if v.dtype != torch.bfloat16 else v
            a = a.to(torch.bfloat16) if a.dtype != torch.bfloat16 else a
            b = b.to(torch.bfloat16) if b.dtype != torch.bfloat16 else b
            q, w, k, v, a, b = [i.view(B, T, HC//head_size, head_size) for i in [q, w, k, v, a, b]]
            y_chunk = WindBackstepping.apply(w, q, k, v, a, b)
            y_tensor = cast(torch.Tensor, y_chunk)
            return torch.reshape(y_tensor, (B, T, HC))

        _cuda_operator_train[head_size] = RUN_CUDA_RWKV7g

        if rank == 0:
            print("RWKV7 CUDA training operator ready")

        return _cuda_operator_train[head_size]


def load_cuda_operator_inference(head_size: int = 64) -> Callable:
    """
    Load the CUDA WKV operator (inference version with state support).

    Features:
    - Supports state passing (RNN-mode inference)
    - Forward-only
    - Optimized for inference
    - Distributed support (multi-GPU)
    - Automatic cache detection

    Args:
        head_size: Attention head size.

    Returns:
        CUDA inference operator function.
    """
    global _cuda_operator_inference

    if head_size in _cuda_operator_inference:
        return _cuda_operator_inference[head_size]

    with _load_lock:
        if head_size in _cuda_operator_inference:
            return _cuda_operator_inference[head_size]

        import torch
        from torch.utils.cpp_extension import load

        rank = _get_rank()
        world_size = _get_world_size()

        local_cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda')
        module_name = f"wkv7s_custom_h{head_size}"

        cache_exists = _check_cache_exists(module_name)

        if rank == 0:
            if cache_exists:
                print(f"Loading RWKV7s CUDA inference operator (cached, head_size={head_size})")
            else:
                print(f"First-time RWKV7s CUDA inference operator compilation (head_size={head_size})")
                print("  Compiling CUDA kernel...")
        elif not cache_exists:
            print(f"[Rank {rank}/{world_size}] Waiting for rank 0 to compile inference operator...")

        flags = [
            '-res-usage',
            f'-D_N_={head_size}',
            '--use_fast_math',
            '-O3',
            '-Xptxas -O3',
            '--extra-device-vectorization'
        ]

        try:
            if rank == 0:
                with _interprocess_build_lock(module_name):
                    start_time = time.time()
                    load(
                        name="wkv7s_custom",
                        sources=[
                            os.path.join(local_cuda_dir, 'wkv7s_op.cpp'),
                            os.path.join(local_cuda_dir, 'wkv7s.cu')
                        ],
                        is_python_module=False,
                        verbose=False,
                        extra_cuda_cflags=flags
                    )
                    load_time = time.time() - start_time
                    print(f"Inference operator loaded ({load_time:.2f}s)")

            _barrier()

            if rank > 0:
                with _interprocess_build_lock(module_name):
                    load(
                        name="wkv7s_custom",
                        sources=[
                            os.path.join(local_cuda_dir, 'wkv7s_op.cpp'),
                            os.path.join(local_cuda_dir, 'wkv7s.cu')
                        ],
                        is_python_module=False,
                        verbose=False,
                        extra_cuda_cflags=flags
                    )
                print(f"[Rank {rank}] Inference operator loaded")

        except Exception as e:
            print(f"Warning: CUDA operator compilation failed: {e}")
            print("  Falling back to pure PyTorch implementation (slower)")
            _cuda_operator_inference[head_size] = load_cpu_operator(head_size)
            return _cuda_operator_inference[head_size]

        class WKV7S(torch.autograd.Function):
            @staticmethod
            def forward(ctx, state, r, w, k, v, a, b):
                with torch.no_grad():
                    if r.ndim == 2:
                        T, C = r.size()
                        B = 1
                        y = torch.empty((T, C), device=r.device, dtype=r.dtype)
                    elif r.ndim == 3:
                        B, T, C = r.size()
                        y = torch.empty((B, T, C), device=r.device, dtype=r.dtype)
                    else:
                        raise RuntimeError(f"Unexpected input shape: {r.shape}, expected [T, C] or [B, T, C]")

                    H = C // head_size

                    if r.dtype == torch.float16:
                        torch.ops.wkv7s_custom.forward_fp16(B, T, C, H, state, r, w, k, v, a, b, y)
                    elif r.dtype == torch.bfloat16:
                        torch.ops.wkv7s_custom.forward_bf16(B, T, C, H, state, r, w, k, v, a, b, y)
                    elif r.dtype == torch.float32:
                        torch.ops.wkv7s_custom.forward_fp32(B, T, C, H, state, r, w, k, v, a, b, y)
                    else:
                        raise RuntimeError(f"Unsupported dtype for wkv7s operator: {r.dtype}")
                    return y

        def RUN_CUDA_RWKV7S(r, w, k, v, a, b, state=None, return_state=False):
            if state is None:
                raise ValueError("wkv7s requires a state argument")
            # The wkv7s kernel writes the updated state back into the passed
            # buffer in-place. Clone here so caller-owned tensors (state caches,
            # other requests' slots) are never corrupted; CPU path is
            # out-of-place, so this also unifies CPU/CUDA mutation semantics.
            # Return the updated clone.
            state_work = state.clone()
            y = WKV7S.apply(state_work, r, w, k, v, a, b)
            if return_state:
                return y, state_work
            return y

        _cuda_operator_inference[head_size] = RUN_CUDA_RWKV7S

        if rank == 0:
            print("RWKV7s CUDA inference operator ready (with state support)")

        return _cuda_operator_inference[head_size]


# ============================================================================
# Unified loading interface
# ============================================================================

def load_wkv_operator(device: str = 'cuda', for_inference: bool = False, head_size: int = 64) -> Callable:
    """
    Load WKV operator (unified entry point).

    Args:
        device: 'cpu' or 'cuda'
        for_inference: Whether for inference (only matters for CUDA)
        head_size: Attention head size

    Returns:
        WKV operator function.
    """
    if device == 'cpu':
        return load_cpu_operator(head_size)
    elif device == 'cuda':
        if for_inference:
            return load_cuda_operator_inference(head_size)
        else:
            return load_cuda_operator_train(head_size)
    else:
        raise ValueError(f"Unsupported device type: {device}. Only 'cpu' and 'cuda' are supported.")


# ============================================================================
# Convenience interface (backward compatible)
# ============================================================================

def rwkv7_wkv(q, w, k, v, a, b, head_size=64, state=None, return_state=False):
    """
    RWKV7 WKV operation convenience interface.

    Automatically selects the appropriate operator based on input tensor device.

    Args:
        q, w, k, v, a, b: Input tensors [B, T, C]
        head_size: Attention head size
        state: Input state (optional, for inference)
        return_state: Whether to return state

    Returns:
        Output tensor [B, T, C]
        state: Final state (only when return_state=True)
    """
    import os

    device_type = os.environ.get("RWKV_DEVICE", "auto")
    if device_type == "auto":
        device_type = "cuda" if q.is_cuda else "cpu"

    for_inference = (state is not None) or return_state

    operator = load_wkv_operator(device=device_type, for_inference=for_inference, head_size=head_size)

    if for_inference and device_type == 'cuda':
        return operator(q, w, k, v, a, b, state, return_state)
    elif for_inference and device_type == 'cpu':
        return operator(q, w, k, v, a, b, state=state, return_state=return_state)
    else:
        return operator(q, w, k, v, a, b)


# ============================================================================
# Infinite context (infctx) operator
# ============================================================================

def load_cuda_operator_infctx():
    """
    Load the CUDA WKV operator (infinite-context training version).

    Features:
    - Supports infinite-length context training
    - Uses segmented state passing
    - Auto JIT compilation
    - Distributed training support (multi-GPU)
    - Automatic cache detection

    Returns:
        CUDA infctx operator module.
    """
    global _cuda_operator_infctx

    if _cuda_operator_infctx is not None:
        return _cuda_operator_infctx

    with _load_lock:
        if _cuda_operator_infctx is not None:
            return _cuda_operator_infctx

        import torch
        from torch.utils.cpp_extension import load

        rank = _get_rank()
        world_size = _get_world_size()

        cuda_dir = Path(__file__).parent / 'cuda'
        module_name = 'wkv7_infctx_op'

        cache_exists = _check_cache_exists(module_name)

        if rank == 0:
            if cache_exists:
                print("Loading RWKV7 infinite-context CUDA operator (cached)")
            else:
                print("First-time RWKV7 infinite-context CUDA operator compilation")
                print("  Compiling CUDA kernel...")
        elif not cache_exists:
            print(f"[Rank {rank}/{world_size}] Waiting for rank 0 to compile infctx operator...")

        try:
            if rank == 0:
                with _interprocess_build_lock(module_name):
                    start_time = time.time()
                    _cuda_operator_infctx = load(
                        name='wkv7_infctx_op',
                        sources=[
                            str(cuda_dir / 'wkv7_infctx_op.cpp'),
                            str(cuda_dir / 'wkv7_infctx_cuda.cu'),
                        ],
                        extra_cflags=['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                        extra_cuda_cflags=[
                            '-O3',
                            '--use_fast_math',
                            '-std=c++17',
                            '-D_GLIBCXX_USE_CXX11_ABI=0',
                            '-gencode=arch=compute_70,code=sm_70',
                            '-gencode=arch=compute_75,code=sm_75',
                            '-gencode=arch=compute_80,code=sm_80',
                            '-gencode=arch=compute_86,code=sm_86',
                            '-gencode=arch=compute_89,code=sm_89',
                            '-gencode=arch=compute_90,code=sm_90',
                        ],
                        verbose=False
                    )
                    load_time = time.time() - start_time
                    print(f"Infinite-context operator loaded ({load_time:.2f}s)")

            _barrier()

            if rank > 0:
                with _interprocess_build_lock(module_name):
                    _cuda_operator_infctx = load(
                        name='wkv7_infctx_op',
                        sources=[
                            str(cuda_dir / 'wkv7_infctx_op.cpp'),
                            str(cuda_dir / 'wkv7_infctx_cuda.cu'),
                        ],
                        extra_cflags=['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                        extra_cuda_cflags=[
                            '-O3',
                            '--use_fast_math',
                            '-std=c++17',
                            '-D_GLIBCXX_USE_CXX11_ABI=0',
                            '-gencode=arch=compute_70,code=sm_70',
                            '-gencode=arch=compute_75,code=sm_75',
                            '-gencode=arch=compute_80,code=sm_80',
                            '-gencode=arch=compute_86,code=sm_86',
                            '-gencode=arch=compute_89,code=sm_89',
                            '-gencode=arch=compute_90,code=sm_90',
                        ],
                        verbose=False
                    )
                print(f"[Rank {rank}] Infinite-context operator loaded")

            return _cuda_operator_infctx

        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to load wkv7_infctx_op: {e}")
                print("  Falling back to pure PyTorch implementation (slower)")
            _cuda_operator_infctx = None
            return None
