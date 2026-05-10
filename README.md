# HPML Final Project: [Project Title]

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** [Team Name]
- **Members:**
  - Full Name 1 (UNI) — *role / area of contribution*
  - Full Name 2 (UNI) — *role / area of contribution*
  - Full Name 3 (UNI) — *role / area of contribution*
  - Full Name 4 (UNI) — *role / area of contribution*

## Submission

- **GitHub repository:** [https://github.com/&lt;org&gt;/&lt;repo&gt;](https://github.com/org/repo)
- **Final report:** [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](deliverables/HPML_Final_Presentation.pptx)
- **Experiment-tracking dashboard:** [link to public Wandb / MLflow / TensorBoard / Comet / Neptune dashboard]

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

A 2–4 sentence description of the workload, the system being optimized, and *why* the optimization matters. State whether you are targeting **training**, **inference**, or **both**, and identify the bottleneck (compute, memory bandwidth, I/O, communication, etc.) you set out to address.

---

## 2. Model/Application Description

Briefly describe the model(s) and stack you used:

- **Model architecture:** e.g., Llama-3.1 8B, ResNet-50, Stable Diffusion XL.
- **Framework:** PyTorch 2.x / JAX / TensorFlow / vLLM / TGI.
- **Dataset:** name, size, license, and link.
- **Custom layers or modifications:** anything you changed from the upstream reference implementation.
- **Hardware target:** NVIDIA A100 / H100 / Jetson Orin / Cloud TPU v5e / Apple M-series / IBM AIU, etc.

---

## 3. Final Results Summary

Replace the numbers below with your measured values. Add or remove rows to fit your study.

| Metric                       | Baseline | Optimized | Δ (Improvement) |
| ---------------------------- | -------- | --------- | --------------- |
| Top-1 Accuracy / Task Metric | XX.XX%   | XX.XX%    | ±X.XX pp        |
| Inference Latency (p50)      | XX.XX ms | XX.XX ms  | XX% faster      |
| Inference Throughput         | XXX tok/s| XXX tok/s | XX× higher      |
| Training Time / Epoch        | XX s     | XX s      | XX% faster      |
| Peak GPU Memory              | XX GB    | XX GB     | XX% less        |
| Model Size on Disk           | XX MB    | XX MB     | XX% smaller     |
| Energy / Sample (optional)   | X.XX J   | X.XX J    | XX% less        |

**Hardware:** [e.g., 1× NVIDIA A100 80GB SXM, CUDA 12.4, PyTorch 2.5, Ubuntu 22.04]

**Headline result (one sentence):** *e.g., "Applying LoRA + 4-bit quantization reduced fine-tuning memory from 38 GB to 9 GB and cut wall-clock training time per epoch by 2.7× on a single A100, with no measurable accuracy degradation on the GLUE benchmark."*

---

## 4. Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/                # YAML / JSON configs for every reported experiment
├── deliverables/           # Final report (PDF) and final presentation (PPT/PDF) — same files uploaded to CourseWorks
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
├── scripts/
│   ├── download_dataset.sh
│   ├── run_baseline.sh
│   └── run_optimized.sh
├── src/
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # Model definitions / wrappers
│   ├── train.py            # Training entry point
│   ├── eval.py             # Evaluation entry point
│   └── profile.py          # Profiling entry point
├── notebooks/              # Exploratory & analysis notebooks
├── results/                # Logs, figures, profiler traces (small files only)
└── docs/                   # Optional: extended methodology, design notes
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
# Clone
git clone https://github.com/<org>/<repo>.git
cd <repo>

# (Recommended) create a clean Python environment
python -m venv .venv && source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

**System requirements:** Python 3.10+, CUDA 12.x, ≥ 24 GB GPU memory for [model X]. See `requirements.txt` for pinned package versions.

### B. Experiment Tracking Dashboard

Public experiment-tracking dashboard with training and evaluation metrics, system profiling, and baseline vs. optimized comparisons:

> **🔗 Dashboard:** [https://wandb.ai/&lt;team&gt;/&lt;project&gt;](https://wandb.ai/team/project)
>
> *Platform used:* [Weights & Biases / MLflow / TensorBoard / Comet / Neptune / other]

Verify the link opens in an incognito browser. The dashboard includes a curated **report** that walks through the optimization story. If your platform does not support public links (e.g., self-hosted MLflow), a static export is committed under `results/dashboard/` instead.

### C. Dataset

```bash
bash scripts/download_dataset.sh
# or follow the manual instructions in docs/data.md
```

The dataset is *not* committed to the repository. The script fetches it from [source] (license: [license]) and stores it under `data/`.

### D. Training

To reproduce the baseline:

```bash
python src/train.py --config configs/baseline.yaml
```

To reproduce the optimized run:

```bash
python src/train.py --config configs/optimized.yaml
```

### E. Evaluation

```bash
python src/eval.py --weights checkpoints/best_model.pth --config configs/optimized.yaml
```

### F. Profiling

To regenerate the profiler traces referenced in the report:

```bash
python src/profile.py --config configs/optimized.yaml --output results/trace.json
# View in chrome://tracing or perfetto.dev
```

### G. Quickstart: Reproduce the Headline Result

The following sequence reproduces the headline number in Section 3 end-to-end (≈ XX minutes on a single A100):

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Download dataset
bash scripts/download_dataset.sh

# 3. Run optimized training (or skip if checkpoint provided in releases)
bash scripts/run_optimized.sh

# 4. Evaluate
python src/eval.py --weights checkpoints/best_model.pth
```

---

## 6. Results and Observations

A short narrative (3–6 bullets) summarizing what you found. Include 1–2 representative figures from `results/` directly in this README so a reader gets the gist without opening Wandb.

- *Optimization 1 (e.g., torch.compile + bfloat16):* X% latency reduction, attributable to [reason].
- *Optimization 2 (e.g., FlashAttention-2):* Y% memory reduction at long context lengths.
- *Optimization 3 (e.g., paged KV cache):* Z× throughput gain at batch size 32.
- *What did not work:* [briefly note any optimization that failed or regressed performance, and why you think it failed].

![Baseline vs Optimized latency](results/figures/latency_comparison.png)

---

## 7. Notes

- Source files live under `src/`, configuration under `configs/`, and scripts under `scripts/`.
- Trained checkpoints are stored in [GitHub Releases / Hugging Face Hub / external bucket] — see `docs/checkpoints.md`.
- All secrets (API keys, Wandb tokens) are loaded from environment variables. See `.env.example`.

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [ ] Yes, we used AI assistance as described below.

**Tool(s) used:** *e.g., ChatGPT, Claude, GitHub Copilot, Cursor*

**Specific purpose:** *e.g., debugged a CUDA OOM error, clarified SM occupancy, polished prose in the report's introduction*

**Sections affected:** *e.g., src/profile.py setup, README §6 results narrative, report §V Discussion*

**How we verified correctness:** *e.g., re-ran every reported experiment ourselves; confirmed profiler-trace interpretations against the raw traces in results/; rewrote AI-suggested code in our own words and confirmed it produces the same numbers as the version we hand-wrote.*

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{teamname2026hpml,
  title  = {[Project Title]},
  author = {Last1, First1 and Last2, First2 and Last3, First3},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/<org>/<repo>}
}
```

### Contact

Open a GitHub Issue or email *[team-contact@columbia.edu]*.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
