# AssetOpsBench at a glance

Snapshot: November 9, 2026. These metrics describe the scope explicitly listed below.

| Headline | Value | Counting rule and source |
| :--- | ---: | :--- |
| **Research papers** | **15** | One entry per paper in the README's core research table: published work, acceptance-announced work, and preprints. Excludes tutorials, bridge/workshop contributions in the separate list, and community papers in the extensions table. This is a count of the curated table, not every paper ever associated with the ecosystem. [Research portfolio](README.md#research-portfolio). |
| **Benchmark families** | **6** | AssetOpsBench, FailureSensorIQ, DiagnosticIQ, PHMForge, IoTBench, and IndustryAssetEQA. Families are not mutually exclusive datasets; do not add their scenario counts without checking overlap. [Benchmark directory](README.md#a-family-of-industrial-ai-benchmarks). |
| **MCP servers** | **5 + 1** | Five domain servers plus a shared utility server, per the [server reference](https://github.com/IBM/AssetOpsBench/blob/main/docs/mcp-servers.md). This is not the count of individual tools. |
| **Competition editions** | **2** | CODS 2025 and IJCAI 2026. Individual tracks are not counted as additional editions. [CODS](https://ikdd.acm.org/cods-2025/data-challenge.php) · [IJCAI](https://2026.ijcai.org/competitions/). |
| **Dataset downloads last month** | **2,654** | Sum of Hugging Face's `downloads` field for the three repositories below. This counts repository download activity, not unique people, installations, or lifetime downloads. |
| **Universities represented** | **4** | Columbia, University at Albany, Binghamton, and University of Technology Sydney in the listed academic-project directory. This is a scoped count of that directory, not all academic collaborators or institutional partnerships. [Project directory](docs/university-projects.md). |

## Dataset download snapshot

| Dataset repository | Downloads last month | Source |
| :--- | ---: | :--- |
| AssetOpsBench | 1,108 | [Dataset page](https://huggingface.co/datasets/ibm-research/AssetOpsBench) · [Public API](https://huggingface.co/api/datasets/ibm-research/AssetOpsBench) |
| FailureSensorIQ | 366 | [Dataset page](https://huggingface.co/datasets/ibm-research/FailureSensorIQ) · [Public API](https://huggingface.co/api/datasets/ibm-research/FailureSensorIQ) |
| PHMForge | 1,180 | [Dataset page](https://huggingface.co/datasets/ibm-research/PHMForge) · [Public API](https://huggingface.co/api/datasets/ibm-research/PHMForge) |
| **Total** | **2,654** | Snapshot captured November 9, 2026. |

The underlying public API values are preserved in [metrics-snapshot.json](assets/metrics-snapshot.json). DiagnosticIQ and other resources are not included in this download total. Hugging Face labels this statistic “Downloads last month”; do not relabel it as a calendar-month or lifetime total.

## Reusable headline assets

- [Full ecosystem diagram, SVG](assets/assetopsbench-ecosystem.svg) · [PNG](assets/assetopsbench-ecosystem.png)
- [Standalone metrics strip, SVG](assets/assetopsbench-metrics.svg) · [PNG](assets/assetopsbench-metrics.png)

The concise diagram keeps headline metrics separate from the ecosystem layers.

The diagram uses a layered ecosystem layout inspired by the supplied AutoGen reference. Its SVG has selectable text. The PNG versions are for presentations and places that do not render SVG.

## Additional headline candidates

**GitHub stars and forks** can work as secondary community signals, with live repository badges or a dated snapshot. **Citations** require a named index and retrieval date. **Academic projects**, **participants**, and **submissions** need stable counting rules that distinguish projects from papers and teams from user accounts. **Media mentions** should distinguish original coverage from syndicated copies and directory listings.

These are useful future metrics, but should be added only when their sources and definitions can be maintained alongside the headline strip.

To refresh the visuals, edit the source SVGs and this file, update the saved metric snapshot, and export the SVGs to PNG. The assets are static and do not fetch network data automatically.

## Citation discovery snapshot

See the [Google Scholar audit](CITATION_AUDIT.md) for dated record-level counts and deduplication limits. These raw counts are excluded from the headline metrics because versions overlap and some indexed citing items are not papers.

## Verified reuse highlights

Reviewed November 10, 2026. These are selected examples from the [research-use register](RESEARCH_USE.md), not totals of adopters or independently reproduced results. They do not change the dated November 9 headline strip above.

| README highlight | Verified scope and limits |
| :--- | :--- |
| **Training and fine-tuning** | The [FSIQ LoRA adapter](https://huggingface.co/predictive-maintenance/qwen3-4b-failuresensoriq-lora), [Pesquisa_IC_IA QLoRA project](https://github.com/viniciusjsdev/Pesquisa_IC_IA/blob/main/FineTuning/docs/modelos_disponiveis.md), [ASTRIS reranker](https://huggingface.co/DestinyOtto/astris-minilm-l6-assetops-reranker), and [Columbia router](https://github.com/jasonlee-1024/AssetOpsBench) document training on family data or derived labels. These are distinct examples, not a count of all models. Pesquisa weights were checked through Git LFS pointers; the router's checkpoint requires university login. No blanket claim of public weights is made for all four. |
| **Public QLoRA implementation** | [Damarla–Zhu code](https://github.com/seshu-damarla/industrial-failure-sensor-reasoning/tree/70feb3b5afbd0d6e7b6d083d6069e32b295d9811) contains FailureSensorIQ artifacts and a notebook training path; their paper reports retrieval, fine-tuning, and perturbation experiments. Code was inspected, not executed; no additional publicly released adapter is implied. |
| **Model-routing research** | [Han–Park code and results](https://github.com/Sangdon-Park/selfish-escalation/tree/6757a7611f388e8c0061e7da0b128d939d8229bf) identify the 2,667-question FailureSensorIQ single-answer configuration and publish measurements, fold-analysis outputs, and reproduction scripts. This is research-code evidence, not verified publication acceptance or independent replication. |
| **143 test tasks** | shopfloor-agent's own test split; excludes its 37 development tasks. Its AOB sample-data loader is pinned and checksummed. This is a derivative evaluation, not 143 additional canonical AOB scenarios. |
| **22 visual scenarios / 2 released models** | One Columbia Team 23 extension and its two HF checkpoint repositories (Qwen and Llama). Cards and weight-file listings were inspected. Quantization/calibration is not supervised fine-tuning on AOB. The linked HF VLM-22 dataset was inaccessible; the README links the checked GitHub project instead. |

Selected revisions and inspection limits are saved in [research-use-evidence.json](assets/research-use-evidence.json), the [fork/model audit](assets/fork-model-audit.json), and the [public-reference snapshot](assets/reference-discovery-snapshot.json). Model training and downstream evaluations were not rerun. Company competition use, reported transfer evaluation, and additional studies remain linked in [RESEARCH_USE.md](RESEARCH_USE.md).
