# ACWorld paper score tables

This archive contains all 2,000 task scores for the ten models in the paper's capability-coverage track. Each model has 200 results, comprising 116 Buyer tasks and 84 Merchant tasks. The task and scorer configuration is available in ACWorld `v1.0.0`, also tagged `kdd2027-submission`, at commit `169602300bb0b9c51c69c0c8afa54c0756a3f7d3`.

- `scores.csv`: model names and identifiers, task IDs, families, roles, scores, full-credit indicators, and recorded model call counts.
- `model-summary.csv`: overall, role, and family means, with full, partial, and zero-credit counts.
- `summarize.py`: aggregation script using the Python standard library.
- `API_CONFIGURATION.md`: evaluation dates and shared client settings.
- `SHA256SUMS`: SHA-256 checksums for the other files in this archive.
- `LICENSE`: license for the included code and documentation.

Run `python3 summarize.py` from the extracted folder to regenerate `model-summary.csv`. Means use the paper calculation: mean task score rounded to six decimals, multiplied by 100, then displayed to one decimal place. This reproduces the 130 overall, role, and family means from Tables 4 and 13 for the capability-coverage track. Direct one-decimal rounding can produce different displayed values. The large-catalog track is outside this archive's scope.

The script aggregates the recorded task scores without model API calls. Readers can inspect individual results and check their aggregation into the paper tables. Verify the archive against its published `.sha256` file and its contents against `SHA256SUMS`. Cite arXiv:2608.02441v1.
