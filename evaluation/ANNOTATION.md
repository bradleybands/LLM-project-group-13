# Reviewing the question set

`questions.json` is a machine-generated draft attached to real ad IDs. It is not a fully labelled benchmark.

1. Open the gold ad in the local dataset or original URL. Record requirements in `expected_facts`, including exact qualifications, skill/language requirements and any important absence of information.
2. Rewrite exact-title questions into natural English paraphrases for a harder retrieval benchmark. Keep some precise questions as a baseline. Include English questions whose gold source is Swedish.
3. Validate `gold_ad_ids`. For filtered examples, all matching ads are accepted examples; this differs from requiring every ad in the answer. CV questions currently have no gold label: inspect roles and label appropriate examples with explicit supporting requirements.
4. Verify the unanswerable cases. A question is unanswerable only relative to the evidence and requested claim, not because it happens to return no vector matches.
5. Set `reviewed: true` only after reviewing the source. Preserve original drafts separately if useful and freeze the revised set across experiments.

For generation, classify each factual assertion as supported, unsupported or not assessable against the cited source and expected facts. Do not count an absence of CV evidence as demonstrated inability. Treat a fabricated salary, company, requirement or hiring guarantee as unsupported. Count refusal correctness separately from faithfulness: a correct refusal need not contain factual claims or citations.

Use synthetic CVs only in shared experiments. Generation CSVs store the answers and source excerpts for scoring. Report automatic numeric citation validation separately from manual evidence support.
