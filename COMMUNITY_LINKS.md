# AssetOpsBench community use & useful links

Explore how people use, extend, evaluate and discuss **AssetOpsBench, FailureSensorIQ, DiagnosticIQ, PHMForge and AssetArena**. This directory brings the links together; each entry identifies the kind of evidence it provides.

[Dataset reuse](#dataset-reuse--research) · [Models & training](#models--training) · [Platforms & leaderboards](#platforms--leaderboards) · [Projects](#projects--evaluation-tools) · [Industry & coverage](#industry--community-coverage) · [Tutorials & events](#tutorials--events)

## Dataset reuse & research

| Resource | How it builds on the family |
| :--- | :--- |
| [Samyama / VaidhyaMegha graph extension](https://github.com/samyama-ai/assetops-kg) · [Derived dataset](https://huggingface.co/datasets/VaidhyaMegha/assetops-kg) | AOB data adapted into an industrial knowledge graph; released data, implementation and experiments. |
| [Chain of Physics](https://arxiv.org/abs/2611.14263) | Independent FailureSensorIQ evaluation across four asset classes and three LLMs using engineering-informed graph reasoning. November 2026 preprint. |
| [Damarla–Zhu industrial failure–sensor reasoning](https://github.com/seshu-damarla/industrial-failure-sensor-reasoning) · [Paper](https://arxiv.org/abs/2607.24824) | FailureSensorIQ retrieval, prompting and QLoRA experiments. |
| [Industrial-Instruction](https://arxiv.org/abs/2608.22817) | Evaluates industrial models on original and perturbed FailureSensorIQ questions; its training corpus is separately sourced. |
| [multi-CaMeL](https://arxiv.org/abs/2610.05640) | External AOB evaluation of agent security and utility. |
| [FedProC data loader](https://github.com/nclabteam/FedProC/blob/main/data_factory/AssetOpsBench.py) | Integrates AOB telemetry into a federated time-series framework. |
| [shopfloor-agent](https://github.com/GBR-RL/shopfloor-agent) | Uses AOB sample operational data to build a plant database and a separate agent evaluation. |
| [Transformer scenario dataset](https://huggingface.co/datasets/rohithkanathur/assetopsbench-transformer-scenarios-dataset) | University-project extension releasing 50 transformer scenarios. |
| [AI4Manufacturing EMB-D29](https://huggingface.co/datasets/AI4Manufacturing/EMB-D29) | Public card describes 1,138 retrieval records reformatted for supervised fine-tuning, crediting FailureSensorIQ's embedding pipeline. Files are gated; no trained model is established by the card. |

## Models & training

| Resource | Released artifact or training evidence |
| :--- | :--- |
| [Qwen3-4B FailureSensorIQ LoRA](https://huggingface.co/predictive-maintenance/qwen3-4b-failuresensoriq-lora) | Public adapter; card documents training on 1,242 labeled competition questions. Read its evaluation protocol alongside reported scores. |
| [Qwen2.5 FailureSensorIQ adapter](https://huggingface.co/dseshukumar/failuresensoriq-qwen25-qlora) | Public weights uploaded November 16, 2026. Configuration identifies Qwen2.5-7B-Instruct and rank-16 LoRA. No model card was available when checked; training provenance, quantization and scores remain unverified. |
| [ASTRIS MiniLM reranker](https://huggingface.co/DestinyOtto/astris-minilm-l6-assetops-reranker) · [Training code](https://github.com/Otto-Destiny/assetops-minilm-l6-reranker) | Ranking model fine-tuned using selected AOB examples, with released weights and comparison artifacts. |
| [Pesquisa_IC_IA](https://github.com/viniciusjsdev/Pesquisa_IC_IA/tree/main/FineTuning) | FailureSensorIQ and mixed FinQA/FailureSensorIQ QLoRA training manifests, code and evaluation artifacts. |
| [Industrial-Instruction Qwen model](https://huggingface.co/Parssky/industrial-instruction-qwen4b) · [Claude-data variant](https://huggingface.co/Parssky/industrial-instruction-qwen4b-claude) | Released models evaluated on FailureSensorIQ; trained on the authors' industrial-document corpus. |

## Platforms & leaderboards

| Resource | What to explore |
| :--- | :--- |
| [PHMForge on Harbor](https://hub.harborframework.com/datasets/phmforge/phmforge) | 123 packaged tasks adapted from 99 scenarios. General-purpose agents work from data; this differs from the MCP-paper evaluation. |
| [Kaggle Enterprise Operations](https://www.kaggle.com/benchmarks/ibm-research/enterprise-ops) | IBM–Kaggle benchmark suite. The AOB-labeled component currently evaluates FailureSensorIQ-style MCQA. |
| [FailureSensorIQ HF leaderboard](https://huggingface.co/spaces/cc4718/FailureSensorIQ) | Original project leaderboard Space. |
| [DiagnosticIQ on Kaggle](https://www.kaggle.com/datasets/devindesilva1710/diagnosticiq-paper-rules) | Public release of 6,690 maintenance-action questions. |
| [Benchmark Almanac](https://github.com/brianlzhou/benchmark-almanac) | Separate AOB and FailureSensorIQ importers and stored results; published-result aggregation, not independent model runs. |
| [whatmodel](https://github.com/jakemanger/whatmodel) · [BenchmarkList DiagnosticIQ](https://benchmarklist.com/benchmarks/diagnosticiq_a_benchmark_for_llm_based_industrial_maintenance_action_recommendation_from_symbolic_rules/) | Downstream reuse of published DiagnosticIQ comparisons. |
| [BenchLM FailureSensorIQ](https://benchlm.ai/benchmarks/failuresensoriq) · [ModelScale](https://modelscale.dev/leaderboards/benchmarks/external-failure-sensor-iq) | External result displays; ModelScale identifies BenchLM as a results supplier. Do not count the displays as additional experiments. |

## Projects & evaluation tools

| Resource | Connection |
| :--- | :--- |
| [Lost-in-the-Middle Kaggle notebook](https://www.kaggle.com/code/orecord/lost-in-the-middle-benchmark) | FailureSensorIQ robustness experiments with position, distractors and paraphrasing; saved outputs are available. |
| [PHMForge research adapter](https://github.com/zyqzyq007/phmforge-research-adapter) | Partial community adaptation; consult its validation record and dataset substitutions. |
| [PHMForge-SME](https://github.com/Shawon559/PHMForge-SME) | Adapts the evaluation methodology to local-model bearing diagnosis using Paderborn data. |
| [Faraday Industrial Benchmark](https://github.com/Faraday-Cage-AI/faraday-industrial-benchmark) | Preserves an attributed AOB code/data snapshot; native Faraday results are a separate evaluation. |
| [OpenTalon trajectory schema](https://github.com/opentalon/opentalon/blob/main/internal/scenarios/trajectory.go) | Explicitly implements an AOB-inspired expected tool-call plan and rubric structure; methodological reuse, not verified dataset ingestion. |
| [University projects](docs/university-projects.md) | Named academic collaborations, scenario extensions, model optimization and agent research. |

## Industry & community coverage

| Resource | Type of connection |
| :--- | :--- |
| [LG AI Research: AEGIS](https://www.lgresearch.ai/blog/view?seq=633) | Dedicated AOB discussion and prospective evaluation interest; no completed AEGIS-on-AOB run is reported. |
| [Google Research publication](https://research.google/pubs/convergence-of-llm-integration-and-edge-intelligence-in-decentralized-iot-ecosystems/) | Official-site publication abstract recommends domain-grounded benchmarks such as AOB; not evidence of a Google dataset run. |
| [Good Start Labs transfer study](https://every.to/playtesting/we-trained-an-ai-on-a-board-game-it-became-a-better-customer-support-agent-299b5938-09dd-4881-803f-aea21f0d461f) | Company-reported evaluation on AOB following Diplomacy fine-tuning. |
| [FlyDIY / 飞速创软 case study](https://flydiy.cn/cases/assetopsbench) | Company-reported ontology extension of AOB. Its stated operational targets are not established measured outcomes. |
| [GOMAX benchmark](https://github.com/GOMAX-LAB/gomax-bench-1) | Another industrial benchmark compares its scope with AOB; research positioning rather than dataset reuse. |
| [MostStarred LLM-testing report](https://moststarred.com/reports/llm-testing) | Includes AOB in its analysis of 95 repositories and highlights industrial-maintenance testing. |
| [Viji R: Claude for Manufacturing](https://www.linkedin.com/pulse/claude-manufacturing-new-pathway-industry-viji-r-cmwue/) | Broader manufacturing discussion citing AOB research. |
| [Conversion System property-management article](https://conversionsystem.com/blog/ai-agent-property-management) | Commercial guidance citing DiagnosticIQ and the Samyama extension; not evidence of client deployment or dataset training. |

## Tutorials & events

- [KDD hands-on tutorial](https://shuxinlin.github.io/assetopsbench-kdd-tutorial-2026/) — industrial agents and MCP.
- [Evaluation Science / Benchmarks to Evidence](https://bench-to-evidence.github.io/) — evaluation research and community activities.
- [Coverage, newsletters, tutorials and event directory](COVERAGE_AND_EVENTS.md) — the detailed source collection, including conference discussions and announcements.
- [Resource catalog](RESOURCE_CATALOG.md) — official datasets, tools, pipelines and learning materials.

## Evidence & updates

[Detailed research-use register](RESEARCH_USE.md) · [Citation audit](CITATION_AUDIT.md) · [Metric definitions](METRICS.md) · [Back to README](README.md)

Directory assembled November 20, 2026, from the linked public sources and the existing evidence registers. New discoveries are not necessarily newly published work. Public cards, code and author-reported results establish different kinds of use; downstream workloads were not rerun. AssetArena-specific evidence will be added as it is verified, without conflating similarly named products.

To suggest an addition, open a PR with a public source, the relevant benchmark family, and a short description of what was actually used or released.
