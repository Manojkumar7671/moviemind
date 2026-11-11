"""tests/test_memory_rss.py

Asserts that uvicorn app.main:app startup RSS is <= 450 MB on macOS.
Marked to skip in CI environments where resources or platforms differ.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    os.getenv("CI") == "true" or sys.platform != "darwin",
    reason="Startup RSS test skipped in CI (requires local macOS environment)",
)
def test_uvicorn_startup_rss_under_450mb():
    port = 8799
    venv_python = ROOT_DIR / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [python_exe, "-m", "uvicorn", "app.main:app", "--port", str(port), "--host", "127.0.0.1"]
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    rss_mb = 0.0
    start_time = time.time()
    healthy = False

    try:
        while time.time() - start_time < 25.0:
            time.sleep(0.3)
            try:
                url = f"http://127.0.0.1:{port}/healthz"
                with urllib.request.urlopen(url, timeout=1.0) as res:
                    if res.status == 200:
                        healthy = True
                        break
            except Exception:
                pass

        assert healthy, f"uvicorn app.main:app did not become healthy within 25s on port {port}"

        time.sleep(0.5)
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(proc.pid)]).decode().strip()
        rss_mb = round(int(out) / 1024.0, 2)
        print(f"\nMeasured startup RSS: {rss_mb:.2f} MB")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    assert rss_mb <= 450.0, f"Startup RSS {rss_mb:.2f} MB exceeded 450.0 MB threshold"
