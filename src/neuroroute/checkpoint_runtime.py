"""Host-side runtime identity for MATCH development checkpoint v2.

GPU identity means model/count/backend version, not machine name or GPU UUID:
an allocation may move to equivalent hardware. Matching this metadata is a
necessary guard, not proof of cross-process numerical equivalence.
"""

from importlib import metadata
import os
from pathlib import Path
import platform

import jax

from .reproducible_training import numerical_runtime


def checkpoint_runtime():
    # Reuse the project's explicit double-flag/highest contract. This must be
    # called outside JIT, after flags were set before backend initialization.
    runtime = numerical_runtime()
    packages = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
                "mujoco", "mujoco-mjx", "flax", "optax", "brax",
                "numpy", "scipy", "ml-dtypes")
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        if name.startswith("nvidia-"):
            versions[name] = distribution.version
    runtime["packages"] = versions
    runtime["python"] = platform.python_version()
    runtime["backend_versions"] = sorted({
        str(device.client.platform_version) for device in jax.devices()})
    runtime["numerical_environment"] = {name: os.environ.get(name) for name in
        ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "JAX_ENABLE_X64")}
    # Driver build is stable across equivalent nodes; do not bind to node IDs.
    driver = Path("/proc/driver/nvidia/version")
    runtime["nvidia_driver"] = driver.read_text() if driver.exists() else None
    return runtime
