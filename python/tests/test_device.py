"""Tests for lynx.device module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lynx


class TestDevice:
    def test_cpu(self):
        assert lynx.device("cpu") == "cpu"

    def test_cuda_alias(self):
        # Should normalize "cuda" to "gpu" (or raise if no GPU)
        try:
            d = lynx.device("cuda")
            assert d == "gpu"
        except RuntimeError:
            pass  # OK if no CUDA

    def test_invalid(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown device"):
            lynx.device("tpu")

    def test_cuda_available_returns_bool(self):
        result = lynx.cuda_available()
        assert isinstance(result, bool)
