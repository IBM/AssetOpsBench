# FMEA evaluation

The `fmea` scorer evaluates structured FMEA catalogue scenarios without an
LLM call. Its primary score is semantic F1, and a scenario passes when its F1
is strictly greater than `0.70`. The rubric-weighted composite and strict exact
success remain available as secondary diagnostics.

For `asset_modes` scenarios, the scorer measures failure-mode set recall,
failure-mode set precision, output-contract compliance, and exact
mode/description pairing. Asset names are compared case-insensitively, but
failure-mode names remain exact because the task requires catalogue spellings.

For catalogue-analysis scenarios, nested objects are decomposed into atomic
key-value facts. Mappings are order-independent, while arrays such as rankings
are compared positionally. Numeric JSON values are compared by value, so `7`
and `7.0` are equivalent.

The scorer also reports whether the trajectory used both required catalogue
tools and whether it consulted out-of-scope domain servers. These process
fields are diagnostics and do not change the current answer score.

The aggregate `score_avg` is semantic macro-F1. Per-scenario details include
`semantic_f1`, `f1_pass_threshold`, `rubric_weighted_score`, and
`strict_success`.

Run the MiniMax FMEA evaluation with:

```bash
uv run evaluate \
  --trajectories "../stirrup_agent/tokenrouter-MiniMax-M3 \
  --scenarios "../scenarios_data" \
  --scenario-ids fmea_all \
  --scorer-default fmea \
  --reports-dir "../reports/stirrup_agent/tokenrouter-MiniMax-M3-fmea"
```

`reference_answer.json` currently stores summary baseline metrics rather than
the baseline output. The scorer therefore records its reported F1 but does not
apply the stated RACE normalization; that requires running the actual baseline
answer through the same scorer.
