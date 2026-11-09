# Research impact: datasets in use

Reviewed November 9, 2026. This register connects the industrial AI benchmark family to papers that evaluate on its data, extend its artifacts, or cite its research. A separate coverage category records industry discussion and prospective use.

## External experimental use

| Research | People | Resource used | Evidence of use |
| :--- | :--- | :--- | :--- |
| [Can CaMeLs Talk?](https://arxiv.org/abs/2610.05640), October 2026 preprint | James Peters-Gill, Avi Semler, Henning Bartsch, Ilia Shumailov, Christian Schroeder de Witt | **AssetOpsBench** | [Section V-A](https://arxiv.org/html/2610.05640v1#S5.SS1) evaluates task utility under alternative agent-security configurations. The paper lists Oxford affiliations for Semler and Schroeder de Witt. It adapts the task set, excluding tasks the authors identify as infeasible; compare within that setup. Attack-success results are measured on **MultiAgentDojo**, not AssetOpsBench. |
| [Retrieval-based and Fine-tuned LLM Approaches for Industrial Asset Health Monitoring and Decision Support](https://arxiv.org/abs/2607.24824), July 2026 preprint | Seshu Kumar Damarla, Xiuli Zhu | **FailureSensorIQ** | The abstract explicitly reports experiments comparing retrieval methods, prompting, QLoRA fine-tuning, and perturbation robustness on FailureSensorIQ. This is dataset reuse, beyond a bibliography mention. |

| [Harnessing collective intelligence of multi-agent LLM systems for sensor failure reasoning in smart manufacturing](https://doi.org/10.1016/j.jii.2025.101012), Journal of Industrial Information Integration 49 (2026) | Wei Gong, Shuang Qiao, Chenhong Cao, Shilei Tan, Junliang Ye, Haoxiang Liu, Si Chen, Xuesong Wang | **FailureSensorIQ** | The publisher's dataset-description section explicitly identifies FailureSensorIQ as the experimental benchmark for its modular multi-agent reasoning framework. |
| [Industrial-Instruction](https://arxiv.org/abs/2608.22817), August 2026 preprint | Parsa Bakhtiari, Hassan Bashiri, Alireza Khalilipour, Masoud Nasiripour, Moharram Challenger | **FailureSensorIQ** | [Section 5.2](https://arxiv.org/html/2608.22817v1#S5.SS2) compares fine-tuned and retrieval-based models on original and perturbed questions. Affiliations include Hamedan University of Technology, University of Antwerp, and Flanders Make. Its training corpus uses public Panasonic documents; this does not establish Panasonic participation. |

These examples establish use in research. Their findings do not establish production readiness or endorsement of the benchmark's entire methodology.

## Experimental reuse and extensions connected to the ecosystem

| Research | Connection and resource | What it enables researchers to investigate |
| :--- | :--- | :--- |
| [DynAMO](https://arxiv.org/abs/2606.19382) | Kanishk Kushwaha, Vikrant Vinod Bansode, Harsh Vardhan, Dhaval C. Patel; shares an AssetOpsBench author | Evaluates sequential and dependency-aware parallel workflows on AssetOpsBench. [Public code](https://github.com/kushwaha001/DynAMO). |
| [Transduction is All You Need for Structured Data Workflows](https://arxiv.org/abs/2508.15610) | IBM Agentics research by Alfio Gliozzo and colleagues; shares a FailureSensorIQ co-author | [Section 5.2](https://arxiv.org/html/2508.15610v3#S5.SS2) evaluates structured workflows and perturbation robustness on FailureSensorIQ. This is verified reuse, with shared authorship; classify separately from external use. |
| [Knowledge Graphs as the Missing Data Layer for LLM-Based Industrial Asset Operations](https://arxiv.org/abs/2605.26874) | Madhulatha Mandarapu and Sandeep Kunkunuru; AssetOpsBench extension | Compares a graph-based data layer with the original environment. [Public implementation](https://github.com/samyama-ai/assetops-kg). |
| [Internalizing Tool Knowledge in Small Language Models via QLoRA Fine-Tuning](https://arxiv.org/abs/2605.17774) | Yuval Shemla, Ayal Yakobe, Tanmay Agarwal; Columbia project | Uses AssetOpsBench to study planning without repeatedly supplying a full tool catalog. |
| [Beyond Final Answers: Auditing Trajectory-Level Hallucinations in Multi-Agent Industrial Workflows](https://arxiv.org/abs/2605.24219) | Harshada Badave and colleagues, including AssetOpsBench team members | Introduces **Trajel**, built from annotated AssetOpsBench traces, for trajectory-level hallucination research. This is a related trace-dataset extension, not independent validation. |

Additional extensions and named project roles are documented in the [university-project directory](docs/university-projects.md).

## External citations and comparison with related work

| Research | People | Verified connection |
| :--- | :--- | :--- |
| [Intent-Driven Network Management with Multi-Agent LLMs: The Confucius Framework](https://minlanyu.seas.harvard.edu/writeup/sigcomm25.pdf), SIGCOMM 2025 | Zhaodong Wang and colleagues, including **Soudeh Ghorbani** and **Minlan Yu**; paper affiliations include Meta, Johns Hopkins, Harvard, and Stony Brook | Related-work discussion names AssetOpsBench; reference 40 cites its paper. This establishes a research citation, not that Meta or these universities deployed or evaluated on AOB. |
| [Can LLM Agents Efficiently Identify Appliance Faults?](https://arxiv.org/abs/2610.19291), October 2026 preprint | Lucas H. McCabe, Sarah Lukens, **H. Howie Huang**; paper affiliations include LMI Consulting and George Washington University | The [related-work section](https://arxiv.org/html/2610.19291v1#S4) discusses AssetOpsBench, FailureSensorIQ, and PHMForge, and the introduction cites DiagnosticIQ. Its experiments use CoffeeBench. This is comparative discussion and citation, not reuse of the four datasets. |

## Industry discussion: LG AI Research

**Dongmin Kim's March 18, 2026 [LG AI Research blog on AEGIS](https://www.lgresearch.ai/blog/view?seq=633)** includes a dedicated AssetOpsBench section and discusses the AssetOpsBench-Live AAAI demonstration. It highlights industrial data integration and assessment of agent decisions, and says LG is considering AssetOpsBench for evaluating AEGIS's future scalability.

This is a concrete industry-research mention and statement of interest. The post does **not** report a completed AEGIS evaluation on AssetOpsBench. The [Korean edition](https://www.lgresearch.ai/blog/view?seq=632) is the same coverage in another language, not a second independent article.

## Evidence across the benchmark family

| Family member | Verified downstream evidence in this audit |
| :--- | :--- |
| **AssetOpsBench** | External experimental use in multi-CaMeL; graph and small-model extensions; Confucius and CoffeeBench citations; LG research discussion. |
| **FailureSensorIQ** | External multi-agent reasoning, Industrial-Instruction, and retrieval/fine-tuning experiments; Agentics reuse with shared authorship; CoffeeBench comparative discussion. |
| **DiagnosticIQ** | CoffeeBench citation in its introduction/reference list. No separate external experimental reuse was verified in this audit. |
| **PHMForge** | CoffeeBench related-work discussion. The [instructor account](https://kaoutarelmaghraoui.com/blog/hpml-industrial-mentorship-arxiv/) also documents its educational/research origin; that is project-connected evidence. |
| **IoTBench / ReActXen** | The [original evaluation](https://aclanthology.org/2025.emnlp-industry.24/) and an [external Japanese paper introduction](https://note.com/tmago_business/n/nd61d5072b7d1) were located. A separate external experiment was not verified. |
| **IndustryAssetEQA** | [Published study and released branch](https://aclanthology.org/2026.acl-industry.49/), plus [external explanatory coverage](https://www.emergentmind.com/topics/industryasseteqa). A separate external experimental reuse was not verified. |

Evidence will differ across releases and publication dates. “Not verified in this audit” is not a claim that no such use exists.

## Reporting impact consistently

Use **evaluated on** for documented experiments, **extended** for new artifacts or methods built on a release, **cited** for references or related-work discussion, and **discussed** for coverage. Record the study version, exact resource, authors, author overlap, and evidence location. Use a named bibliographic index and retrieval date for any future citation totals; this register is a curated set of examples, not a citation count.

## Company and industry coverage

| Publisher or affiliation | Evidence | What can be said |
| :--- | :--- | :--- |
| **LG AI Research** | [AEGIS research blog](https://www.lgresearch.ai/blog/view?seq=633) | Dedicated discussion and prospective evaluation interest. |
| **EQUA AI** | Founder Srikant Naidu's [autonomy guide](https://equa.work/resources/grading-the-autonomy-claim) and [work-order guide](https://equa.work/resources/work-order-history-what-it-can-do) | Company-authored practitioner discussion citing AOB as evaluation evidence. |
| **Lancartech** | [Industrial-agent retrospective](https://lancartech.co.id/en/blog/assetopsbench-ibm-retrospektif-evaluasi-agent-industri) | Company-site commentary based on IBM's earlier account. |
| **HysenLabs** | [AssetOpsBench review](https://hysenlabs.com/projects/ibm-assetopsbench) | External editorial analysis of the repository and adoption considerations. |
| **Meta and LMI Consulting affiliations** | Confucius and CoffeeBench papers above | Author affiliations on citing research, not company adoption announcements. |

This is a verified selection, not a census of company mentions. Commercial summaries and directory listings belong in the [coverage inventory](COVERAGE_AND_EVENTS.md); none of these records alone establishes a customer relationship, partnership, or deployment. The [citation audit](CITATION_AUDIT.md) preserves Scholar records and follow-up leads without inflating the README.
