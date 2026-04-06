# Experimental Data

## L4 GPU — Qwen2.5-8B

Average latency over 23 scenarios

```
Plan:       28.294s  (65.2%)
Execute:    10.373s  (23.9%)
Summarize:   4.743s  (10.9%)
Total:      43.410s
```

---

## A100 — Gemma 4 26B (reasoning disabled)

### Run 1: multi-agent scenarios 501-520 (data missing in CouchDB)

```
Summary over 19 scenarios (19 graded)
Plan:       7.198s  (54.5%)
Execute:    5.444s  (41.2%)
Summarize:  0.571s   (4.3%)
Total:      13.213s
── Accuracy per dimension ──
  task_completion:                  0% passed
  data_retrieval_accuracy:          0% passed
  generalized_result_verification:  0% passed
  agent_sequence_correct:           5% passed
  clarity_and_justification:       95% passed
  hallucinations:                 100% hallucinated
```

Note: All failures caused by missing IoT data (CouchDB only has Chiller 6 / June 2020).

---

### Run 2: chiller_utterance scenarios (single-day queries, data exists)

Model: `litellm_proxy/google/gemma-4-26B-A4B-it`, runs=1

```
Summary: 15 scenarios with latency data, 14 graded
Skipped: 4 (404, 405, 411, 417) — context window overflow
Malformed grade: 1 (412) — judge returned function-call format instead of JSON
Missing: 1 (420) — not captured in output

Plan:       6.636s  (31.5%)
Execute:    5.843s  (27.7%)
Summarize:  8.598s  (40.8%)
Total:      21.078s

── Accuracy per dimension (n=14) ──
  task_completion:                 64% passed   (9/14)
  data_retrieval_accuracy:         93% passed  (13/14)
  generalized_result_verification: 57% passed   (8/14)
  agent_sequence_correct:          71% passed  (10/14)
  clarity_and_justification:       64% passed   (9/14)
  hallucinations:                  21% hallucinated (3/14)
```

Per-scenario breakdown:

| id  | TC | DRA | GRV | ASC | CAJ | H     | total(s) | note |
|-----|----|-----|-----|-----|-----|-------|----------|------|
| 401 | T  | T   | T   | T   | T   | F     | 7.2      |      |
| 402 | T  | T   | F   | T   | F   | F     | 5.1      | missing 1 RT conversion |
| 403 | T  | T   | T   | T   | T   | F     | 13.8     |      |
| 404 | -  | -   | -   | -   | -   | -     | SKIP     | context overflow |
| 405 | -  | -   | -   | -   | -   | -     | SKIP     | context overflow |
| 406 | T  | T   | T   | F   | T   | F     | 10.9     |      |
| 407 | T  | T   | T   | T   | T   | F     | 34.4     | slow summarize |
| 408 | F  | T   | F   | T   | F   | T     | 16.5     | energy calc wrong (65k vs 18k kWh) |
| 409 | F  | T   | F   | F   | F   | F     | 18.8     | calculation step failed |
| 410 | T  | T   | T   | T   | T   | F     | 17.2     |      |
| 411 | -  | -   | -   | -   | -   | -     | SKIP     | context overflow |
| 412 | -  | -   | -   | -   | -   | -     | 17.4     | malformed judge output |
| 413 | T  | T   | T   | T   | T   | F     | 50.2     | slow execute (31s) |
| 414 | F  | T   | F   | F   | T   | F     | 21.0     | noisy data confused model |
| 415 | T  | T   | T   | T   | T   | F     | 18.8     |      |
| 416 | F  | F   | F   | F   | F   | T     | 19.6     | values close but judge harsh |
| 417 | -  | -   | -   | -   | -   | -     | SKIP     | context overflow |
| 418 | F  | T   | F   | T   | F   | T     | 31.5     | model said 151F is plausible |
| 419 | T  | T   | T   | T   | T   | F     | 33.6     | slow summarize |
| 420 | -  | -   | -   | -   | -   | -     | -        | not captured |
