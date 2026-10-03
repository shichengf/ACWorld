"""Recreate the paper's capability means from its 2,000 recorded scores."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
with (ROOT / 'scores.csv').open(newline='') as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 2000 or len({(r['model'], r['task_id']) for r in rows}) != 2000:
    raise SystemExit('Expected 2,000 distinct model/task records')
models = list(dict.fromkeys(r['model'] for r in rows))
if len(models) != 10:
    raise SystemExit('Expected ten models')
with (ROOT / 'model-summary.csv').open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['model', 'overall', 'buyer', 'merchant'] + [f'T{i}' for i in range(1, 11)] + ['full', 'partial', 'zero'])
    for model in models:
        selected = [r for r in rows if r['model'] == model]
        if len(selected) != 200:
            raise SystemExit('Incomplete model: ' + model)
        groups = [selected] + [[r for r in selected if r['role'] == role] for role in ('buyer', 'merchant')]
        groups += [[r for r in selected if r['family'] == f'T{i}'] for i in range(1, 11)]
        if list(map(len, groups)) != [200, 116, 84] + [20] * 10:
            raise SystemExit('Unexpected family/role coverage')
        means = [f'{round(sum(float(r["score"]) for r in g) / len(g), 6) * 100:.1f}' for g in groups]
        counts = [sum(r['strict_success'] == 'True' for r in selected),
                  sum(r['strict_success'] != 'True' and float(r['score']) > 1e-9 for r in selected),
                  sum(float(r['score']) <= 1e-9 for r in selected)]
        writer.writerow([model] + means + counts)
print('Wrote 130 means for ten models from 2,000 task scores.')
