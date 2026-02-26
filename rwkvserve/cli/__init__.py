"""
Command-line interface tools for RWKVServe

Uses lazy imports to avoid RuntimeWarning when running via `python -m rwkvserve.cli.serve`.
"""

def __getattr__(name):
    """Lazy import to avoid module execution order warnings."""
    if name == "infer_main":
        from rwkvserve.cli.infer import main
        return main
    elif name == "serve_main":
        from rwkvserve.cli.serve import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["infer_main", "serve_main"]
