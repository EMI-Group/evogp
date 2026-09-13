#!/usr/bin/env python3
"""r6 GPU-work lock wrapper (coordinator-provided shared scaffolding).

Why: the `flock(1)` binary hangs in this sandbox, so every GPU job in round 6
must be serialized with this Python `fcntl.flock(LOCK_EX)` wrapper on the
shared benchmark lock file instead.

Usage:
    python run_under_lock.py <cmd...>
    python run_under_lock.py timeout -s INT -k 30 600 <cmd...>

Rules (house convention, see r6/CONTEXT.md):
  * exactly ONE lock acquisition per command -- never nest this wrapper;
  * keep `timeout` INSIDE the lock (pass it as part of <cmd>), so lock-wait
    time is never charged to a measurement;
  * the lock file is shared with r4/r5 jobs: /tmp/gpbench_critical.lock.
"""
import fcntl
import os
import subprocess
import sys
import time

LOCK_PATH = "/tmp/gpbench_critical.lock"
WAIT_SECONDS = 7200.0


def main(argv):
    if not argv:
        sys.exit(__doc__)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                sys.exit(f"ERROR: lock timeout after {WAIT_SECONDS:.0f}s on {LOCK_PATH}")
            time.sleep(5.0)
    try:
        return subprocess.run(argv).returncode
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
