"""Private oracle smoke check. This file contains no task answers by itself."""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request


def request(path, payload=None, request_id=None):
    headers = {'Content-Type': 'application/json'}
    if request_id is not None:
        headers['X-ACWorld-Request-ID'] = request_id
    req = urllib.request.Request('http://acworld:8765/' + path,
          data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=180) as response:
        return json.load(response)


def main():
    rows = json.loads(Path(sys.argv[1]).read_text())
    index = 0
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        observed = request('observe')
        if observed['status'] == 'finished':
            if index != len(rows):
                raise SystemExit('Reference replay ended before all decisions were used')
            return
        if observed['status'] != 'awaiting_decision':
            time.sleep(0.05)
            continue
        if index >= len(rows):
            raise SystemExit('Unexpected additional decision request')
        row = rows[index]
        for key in ('system', 'user'):
            value = observed['business_request'][key + '_prompt']
            if hashlib.sha256(value.encode()).hexdigest() != row[key + '_sha256']:
                raise SystemExit('Reference input changed: ' + key)
        request('submit', row['decision'], observed['request_id'])
        index += 1
    raise SystemExit('Reference replay timed out')


if __name__ == '__main__':
    main()
