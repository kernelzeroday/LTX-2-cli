import importlib.util
import sys
from types import ModuleType
if "triton" not in sys.modules:
    triton = ModuleType("triton")
    triton.language = ModuleType("triton.language")
    triton.jit = lambda f: f
    spec = importlib.util.spec_from_loader("triton", loader=None, origin="test-mock")
    triton.__spec__ = spec
    triton.__package__ = "triton"
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = triton.language
import pytest


@pytest.fixture
def sample_config_yaml() -> str:
    return "prompt: 'A test prompt'\nseed: 42\noutput_path: /tmp/out.mp4\n"


@pytest.fixture
def sample_config_toml() -> str:
    return 'prompt = "A test prompt"\nseed = 42\noutput_path = "/tmp/out.mp4"\n'
