# API configuration

All ten models were evaluated during 21–27 July 2026 (UTC) through OpenRouter using the same client settings apart from the model identifier. The exact identifiers used in the recorded results are provided in the `model_id` column of `scores.csv`.

| Setting | Value |
| --- | --- |
| API endpoint | `POST https://openrouter.ai/api/v1/chat/completions` |
| Temperature | Provider default |
| Top-p | Provider default |
| Reasoning settings | Provider defaults |
| Output-token limit | Provider default |
| Request timeout | 240 seconds |
| Model call budget | 48 calls per logical task run |
| Provider routing | `sort=latency`, `allow_fallbacks=true`, `require_parameters=false` |

Across the 2,000 scored task records, the `model_calls` column sums to 7,881 model calls.
