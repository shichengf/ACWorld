"""Harbor sidecar collect hook. This control is not exposed over HTTP."""
import json
import os
from pathlib import Path
import time


def main():
    root = Path(os.environ.get('ACWORLD_EVIDENCE_DIR', '/acworld-evidence'))
    if not root.is_dir():
        raise SystemExit('ACWorld evidence directory is missing')
    (root / 'stop-requested').touch(exist_ok=True)
    deadline = time.monotonic() + 100
    while time.monotonic() < deadline:
        result = root / 'result.json'
        if result.exists():
            value = json.loads(result.read_text())
            if value.get('finalized') is not True:
                raise SystemExit('ACWorld result has not been finalized')
            print(json.dumps({'status': value['status'], 'finalized': True}))
            return
        time.sleep(0.1)
    raise SystemExit('ACWorld finalization timed out')


if __name__ == '__main__':
    main()
