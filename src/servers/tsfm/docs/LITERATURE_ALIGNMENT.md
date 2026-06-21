# Literature alignment — are we building the right framework?

A check of the TSFM MCP server design against recent (2024–2026) work on agentic time
series, time-series foundation models (TSFM), AutoML-TS, automated feature engineering, and
MCP. **Verdict: the direction is well-aligned with the 2025–2026 SOTA, and differentiated in
several respects.** The recent work also predicts the gaps to close next.

## 1. What the literature confirms we got right

**The reasoning split (server = passive evidence + grader; agent = decisions).** The 2025
agentic-TS survey defines tools as *"passive: they return information or computations but do not
initiate new reasoning"* and an agent as one that *"selects the next action … in pursuit of a
goal."* That is exactly our split. Our iterate-on-graded-evidence loop is the survey's
**branch-structured reasoning** (fork / aggregate / prune / cycle), its named frontier pattern.
Closest published analogue to our recipe engine: **TS-Reasoner**, which *"decomposes time series
tasks into workflows of specialized operators and refines them with execution feedback."*

**Zero-shot default + keeping classical models + ensembling.** 2025–2026 benchmarks find TSFMs
do *not* universally beat classical methods — *"classical methods remain surprisingly
challenging to beat at very high frequencies,"* and *"ensemble techniques and regression
stacking yield better results than foundation models alone."* Our foundation+classical
mix-and-match with ensembles is the hybrid the evidence favors. The survey notes ensembling is
*"rarely employed"* in agentic systems → our default ensemble+conformal path is white space.

**Per-model parameter reasoning.** A recurring empirical finding: *"context length behaves much
more like a hyperparameter than a dataset parameter … a common trap is assuming larger context
is always better."* This directly justifies our `param_space` layer (context_length is the
canonical parameter it makes the agent reason about from `profile_series` evidence).

**GIFT-Eval.** Confirmed as the de-facto standard (*"55 datasets … 97 distinct test cases"*).
Benchmarking-challenges work warns *"with just four cherry-picked test sets, 46% of models could
appear as state-of-the-art"* → validates our geometric-mean-over-97-configs + mean-rank scoring
over single-dataset numbers.

**MCP substrate + FLOps features.** MCP is now the de-facto integration standard (OpenAI,
Google adopted in 2025), so building as an MCP server is the correct bet; ScaleMCP's dynamic
tool sync echoes our "models are data, not tools" (tiny tool set, large catalog). tsfresh's
hypothesis-test relevance filtering and catch22 are the established automated-feature stack our
FLOps selector sits on (sktime).

## 2. Where we are differentiated (survey-absent, not behind)

A read of the agentic-TS survey found **no** system that (a) treats models/features as
data/catalog-cards, (b) uses **MCP** as the substrate, (c) does **AutoML/FLOps-style automated
feature + model selection over a TSFM catalog**, or (d) makes **ensemble + conformal a default
path**. CoLLM / CAPTime do agentic *model routing*; none combine catalog-as-pointer-index +
recipe composition + GIFT-Eval grading. Our HuggingGPT-style "models are data" framing is
orthogonal to what is published — ahead of, not behind, the cataloged SOTA.

## 3. Gaps to close next (roadmap, literature-justified)

| Priority | Gap | Why (literature) | Our hook |
|---|---|---|---|
| **1** | **Exogenous / covariate forecasting** | Chronos-2 & MOIRAI-2 (Oct 2025) added covariate support as a headline feature | the pending FLOps "exogenous-feature path for forecasting" — build next |
| 2 | **Shift-aware / streaming evaluation** | named open problem; "plan for shift and streaming, treat cost/latency as budgets" | extend `gifteval` with shift/streaming stress splits |
| 3 | **Calibration metrics** | "Are TSFMs well-calibrated?" — accuracy isn't enough | we have conformal intervals; add coverage/calibration scoring |
| 4 | **Retrieval-augmented forecasting (RAF)** | retrieving similar historical segments is emerging | extend our `similarity_search` task into RAG-style grounding |
| 5 | **Reproducibility / audit** | survey 7.1: versioned splits, fixed seeds, audit trails | already partly covered by provenance + `export_state` (#394) |

Optional (further out): multi-agent manager→specialist orchestration (the survey's dominant
multi-agent pattern) — not required, our single-agent + server is sufficient and simpler.

## 4. Recommended next build
**Exogenous/covariate path** (strongest external validation), then **shift-aware GIFT-Eval**.

## Sources
- A Survey of Reasoning and Agentic Systems in Time Series with LLMs — arXiv 2509.11575
- From Prompts to Agents: A Comprehensive Survey of LLM-Driven Time Series Analysis
- Empowering Time Series Forecasting with LLM-Agents — arXiv 2508.04231
- Bridging the Last Mile of Time Series Forecasting with LLM Agents — arXiv 2606.02497
- Benchmarking Foundation Models for Time-Series Forecasting: Zero/Few/Full-Shot — MDPI 11(1)32
- Challenges and Requirements for Benchmarking TSFMs — arXiv 2510.13654
- How Foundational are Foundation Models for Time Series Forecasting? — arXiv 2510.00742
- Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated? — arXiv 2510.16060
- Breaking Silos: Adaptive Model Fusion Unlocks Better TS Forecasting — arXiv 2505.18442
- AutoForecast: Automatic Time-Series Forecasting Model Selection — CIKM '22
- Automatic Feature Engineering for Time Series Classification — arXiv 2308.01071
- ScaleMCP: Dynamic and Auto-Synchronizing MCP Tools for LLM Agents — arXiv 2505.06416
- What is Model Context Protocol (MCP)? — IBM

_Compiled June 2026._
