# Recorded benchmark runs

| Run | Suite | Execution models | Judge | Results |
|---|---|---|---|---|
| [2026-09-30 · Transformer · k = 3](2026-09-30-transformer-k3/README.md) | 52 open-form scenarios × 3 repetitions | Opus 5.5, GPT-6 Astra, GLM 5.3 (low), GPT-6.1 Sol, Fable 5.1 | Independent Fable 5.1 sessions | [HTML](2026-09-30-transformer-k3/comparison.html) · [CSV](2026-09-30-transformer-k3/cases.csv) · [Criterion averages](2026-09-30-transformer-k3/criterion-averages.csv) · [Scenario means](2026-09-30-transformer-k3/scenario-averages.csv) |
| [2026-09-30 · Transformer](2026-09-30-transformer/README.md) | 50 positive + 2 negative, open form | Opus 5.5, GPT-6 Astra, GLM 5.3 (low), GPT-6.1 Sol, Fable 5.1 | Independent Fable 5.1 sessions | [HTML](2026-09-30-transformer/comparison.html) · [CSV](2026-09-30-transformer/cases.csv) · [Criterion averages](2026-09-30-transformer/criterion-averages.csv) |

Each run records scenario and rubric hashes, model/harness settings, measured invocation timing, independent grading, metrics availability and retained execution/judge evidence. A run's README explains its aggregation and environment limits. Published evidence is portable; credentials and local path prefixes are removed. Reporting scripts rebuild artifacts from saved measurements without model calls.
