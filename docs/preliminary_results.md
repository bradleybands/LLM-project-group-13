# Preliminary retrieval results — 6 October 2026

These are measured results on the **unreviewed draft**, not a validated benchmark. The 15 specific questions contain exact ad titles/employers, making them easier than natural user questions. All three indexes use the same 527-ad snapshot. Each retrieval returns up to six chunks, with at most two per ad. Hit@6 means at least one chunk from the labelled ad was returned.

| Chunk tokens | Retrieval | Specific-question hit@6 | Questions |
|---|---|---|---|
| 200 | keyword | 66.7% | 15 |
| 200 | semantic | 100.0% | 15 |
| 200 | hybrid | 93.3% | 15 |
| 320 | keyword | 66.7% | 15 |
| 320 | semantic | 100.0% | 15 |
| 320 | hybrid | 93.3% | 15 |
| 400 | keyword | 66.7% | 15 |
| 400 | semantic | 100.0% | 15 |
| 400 | hybrid | 93.3% | 15 |

All eight draft location-filter questions returned at least one accepted ad in every configuration. This only demonstrates that examples were found under those filters; it does not mean every matching ad was retrieved. CV and unsupported questions have no gold ad IDs and are excluded from these hit scores.

These results do not measure generation faithfulness, English translation accuracy, refusal correctness or hiring suitability. Those require live model calls and manual source-based scoring. Use the CSVs and run.json files in evaluation/results/ to inspect individual outcomes, dataset hashes and settings. Review/rewrite evaluation/questions.json and rerun before making report conclusions.
