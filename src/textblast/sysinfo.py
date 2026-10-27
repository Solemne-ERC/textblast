"""Machine facts used for defaults."""

from __future__ import annotations

import os
import subprocess
import sys


def default_workers() -> int:
    """Performance cores on Apple silicon (efficiency cores are slow for this work), otherwise all cores."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, timeout=5, check=True
            )
            return max(1, int(out.stdout.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return max(1, os.cpu_count() or 1)
