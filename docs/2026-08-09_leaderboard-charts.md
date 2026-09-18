# Leaderboard chart generation

AssetOpsBench can generate publication-ready leaderboard charts directly from
the structured evaluation report. Chart values are derived from individual
`ScenarioResult` records in `EvalReport.results`; they are not hard-coded or
scraped from README images or papers.

## Install

Matplotlib is optional so evaluation users who do not create charts do not
need to install it:

```bash
uv sync --dev --group visualization
```

You can also select the group for one command with `uv run --group
visualization ...`.

## Generate charts

```bash
uv run --group visualization evaluate \
  --trajectories traces/trajectories \
  --scenarios groundtruth/*.json \
  --scorer-default llm_judge \
  --judge-model litellm_proxy/azure/gpt-5.4 \
  --reports-dir reports \
  --charts
```

Chart rendering runs after evaluation from the completed report and does not
make an additional LLM call. Output is deterministic by runner name:

```text
reports/
├── _aggregate.json
└── charts/
    ├── leaderboard-<runner>.svg
    └── leaderboard-<runner>.png
```

SVG is the preferred README and publication format. PNG is provided for quick
preview. Runner names that are not filesystem-safe receive a stable sanitized
name and hash suffix.

## Aggregation rules

The compact leaderboard includes task completion, data retrieval accuracy,
and generalized result verification. For each runner, model, and criterion,
the success percentage is the number of `true` outcomes divided by the number
of results containing an actual Boolean value for that criterion.

- Only results produced by `llm_judge` are included.
- Missing, null, numeric, and string values are excluded from the denominator.
- Models and runners are aggregated independently.
- The optional raw `hallucinations` criterion is inverted and presented as
  “Hallucination-free,” so its positive meaning is explicit.

The historical README leaderboard predates this report pipeline and has no
checked-in `EvalReport` that can reproduce its published values. It remains
unchanged; future leaderboard figures can be generated reproducibly with this
command.

## Accessibility measures

The renderer follows Carbon's official
[categorical palette](https://carbondesignsystem.com/data-visualization/color-palettes/)
in its documented order because models are discrete categories. Each model
also receives a hatch pattern and dark outline, and its style is kept
consistent across every runner in the same report. Charts use a zero baseline,
direct percentage labels, a percent scale, readable legends, and `N/A` gaps
instead of treating missing criteria as failures. SVG text remains selectable,
and each SVG includes image semantics plus a description containing the plotted
models, percentages, and Boolean counts. PNG metadata carries the same data.

These measures improve color-vision, grayscale, and reduced-size readability;
they are not a claim of complete accessibility conformance. When publishing a
chart, provide meaningful surrounding text or alternative text and retain the
canonical JSON report for readers who need the underlying values.
