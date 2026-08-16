from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    demo = ROOT / "demo" / "run_live_demo.py"
    return subprocess.call([sys.executable, str(demo), "--sample", "synthetic"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
