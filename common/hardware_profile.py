"""Hardware profile and device detection for Apple Silicon and x86/CUDA systems."""

from __future__ import annotations
import os
import platform
import subprocess
from typing import Dict, Any, Optional
try:
    import torch
except (ImportError, Exception):
    torch = None


def get_hardware_profile() -> Dict[str, Any]:
    """Detects CPU, GPU / Accelerator (Metal / MPS, CUDA), cores, and architecture."""
    system = platform.system()
    machine = platform.machine()
    processor = platform.processor()

    cpu_brand = processor or machine
    chip_model = "Apple Silicon" if "arm" in machine.lower() and system == "Darwin" else processor

    # Check for Mac sysctl model info
    if system == "Darwin":
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                cpu_brand = res.stdout.strip()
            # Check Apple chip name
            res_chip = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True, check=False)
            if res_chip.returncode == 0:
                chip_model = res_chip.stdout.strip()
        except Exception:
            pass

    cpu_count = os.cpu_count() or 1

    # GPU / MPS / CUDA detection
    mps_available = False
    mps_built = False
    cuda_available = False
    torch_version = "N/A"
    if torch is not None:
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        cuda_available = torch.cuda.is_available()
        torch_version = getattr(torch, "__version__", "N/A")

    gpu_name = "None"
    gpu_type = "None"
    if mps_available:
        gpu_type = "Apple Silicon GPU (Metal Performance Shaders - MPS)"
        gpu_name = f"{cpu_brand} Unified Memory GPU" if "Apple" in cpu_brand else "Apple M-Series GPU"
    elif cuda_available:
        gpu_type = "NVIDIA CUDA"
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "system": system,
        "machine": machine,
        "cpu_brand": cpu_brand,
        "cpu_cores": cpu_count,
        "chip_model": chip_model,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "cuda_available": cuda_available,
        "gpu_type": gpu_type,
        "gpu_name": gpu_name,
        "torch_version": torch_version,
    }


def synchronize_device(device_str: str) -> None:
    """Synchronizes GPU execution queues for accurate timing benchmarks."""
    if device_str == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    elif device_str == "cuda":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
