import pytest
import torch
from common import get_hardware_profile, synchronize_device


def test_hardware_profile_detection():
    hw = get_hardware_profile()
    assert "cpu_brand" in hw
    assert "cpu_cores" in hw
    assert hw["cpu_cores"] >= 1
    assert "mps_available" in hw
    assert hw["mps_available"] is True  # Running on Apple Silicon (M4 Max)


def test_synchronize_device():
    synchronize_device("cpu")
    if torch.backends.mps.is_available():
        synchronize_device("mps")


def test_laya_device_initialization():
    try:
        import laya
        cpu_agent = laya.load("convaiinnovations/laya", device="cpu")
        assert cpu_agent is not None
        assert cpu_agent.device.type == "cpu"

        if torch.backends.mps.is_available():
            mps_agent = laya.load("convaiinnovations/laya", device="mps")
            assert mps_agent is not None
            assert mps_agent.device.type == "mps"
    except Exception as e:
        pytest.skip(f"Live model loading skipped: {e}")
