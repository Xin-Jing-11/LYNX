"""Device management for LYNX.

Usage:
    import lynx
    lynx.cuda_available()  # True/False
    calc = lynx.DFT(xc="PBE", device="gpu")
    calc.to("cpu")
"""

def cuda_available() -> bool:
    """Check if CUDA support is compiled in."""
    try:
        from lynx._core import Calculator
        return Calculator.cuda_available()
    except (ImportError, AttributeError):
        return False


def device(name: str = "cpu") -> str:
    """Validate and normalize device name.

    Args:
        name: "cpu", "gpu", or "cuda" (alias for gpu)

    Returns:
        Normalized device string: "cpu" or "gpu"
    """
    name = name.lower().strip()
    if name in ("cuda", "gpu"):
        if not cuda_available():
            raise RuntimeError(
                "GPU requested but LYNX was built without CUDA support. "
                "Rebuild with -DUSE_CUDA=ON to enable GPU."
            )
        return "gpu"
    elif name == "cpu":
        return "cpu"
    else:
        raise ValueError(f"Unknown device: '{name}'. Use 'cpu' or 'gpu'.")
