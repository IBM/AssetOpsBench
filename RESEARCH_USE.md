# Research impact: datasets in use

Reviewed November 9, 2026. This register connects the industrial AI benchmark family to papers that evaluate on its data, extend its artifacts, or cite its research. The emphasis is documented reuse: experiments, model training, adapters, and derivative benchmarks. Separate categories record citations, industry discussion, and prospective use.

## External experimental use in papers

| Research | People | Resource used | Evidence of use |
| :--- | :--- | :--- | :--- |
| [Can CaMeLs Talk?](https://arxiv.org/abs/2610.05640), October 2026 preprint | James Peters-Gill, Avi Semler, Henning Bartsch, Ilia Shumailov, Christian Schroeder de Witt | **AssetOpsBench** | [Section V-A](https://arxiv.org/html/2610.05640v1#S5.SS1) evaluates task utility under alternative agent-security configurations. The paper lists Oxford affiliations for Semler and Schroeder de Witt. It adapts the task set, excluding tasks the authors identify as infeasible; compare within that setup. Attack-success results are measured on **MultiAgentDojo**, not AssetOpsBench. |
| [Retrieval-based and Fine-tuned LLM Approaches for Industrial Asset Health Monitoring and Decision Support](https://arxiv.org/abs/2607.24824), July 2026 preprint | Seshu Kumar Damarla, Xiuli Zhu | **FailureSensorIQ** | The abstract explicitly reports experiments comparing retrieval methods, prompting, QLoRA fine-tuning, and perturbation robustness on FailureSensorIQ. This is dataset reuse, beyond a bibliography mention. |
| [Harnessing collective intelligence of multi-agent LLM systems for sensor failure reasoning in smart manufacturing](https://doi.org/10.1016/j.jii.2025.101012), Journal of Industrial Information Integration 49 (2026) | Wei Gong, Shuang Qiao, Chenhong Cao, Shilei Tan, Junliang Ye, Haoxiang Liu, Si Chen, Xuesong Wang | **FailureSensorIQ** | The publisher's dataset-description section explicitly identifies FailureSensorIQ as the experimental benchmark for its modular multi-agent reasoning framework. |
| [Industrial-Instruction](https://arxiv.org/abs/2608.22817), August 2026 preprint | Parsa Bakhtiari, Hassan Bashiri, Alireza Khalilipour, Masoud Nasiripour, Moharram Challenger | **FailureSensorIQ** | [Section 5.2](https://arxiv.org/html/2608.22817v1#S5.SS2) compares fine-tuned and retrieval-based models on original and perturbed questions. Affiliations include Hamedan University of Technology, University of Antwerp, and Flanders Make. Its training corpus uses public Panasonic documents; this does not establish Panasonic participation. |

These examples establish use in research. Their findings do not establish production readiness or endorsement of the benchmark's entire methodology.

## Company-reported experiments and competition work

| Organization / people | Documented use | Qualification |
| :--- | :--- | :--- |
| **Good Start Labs — Alex Duffy** | His February 3, 2026 [first-person Every article](https://every.to/playtesting/we-trained-an-ai-on-a-board-game-it-became-a-better-customer-support-agent-299b5938-09dd-4881-803f-aea21f0d461f), linked from the [company research page](https://goodstartlabs.com/research), reports evaluating a Diplomacy-fine-tuned Qwen3-235B on AOB | Qualitative improvement claim; no AOB-specific score, protocol, or uncertainty provided. The article's “over 10 percent” figure applies to other games, not AOB. This establishes reported evaluation, not training on AOB. |
| **GyanData — Smart Maintenance Crew** | [Company announcement](https://www.linkedin.com/posts/gyandata-pvt-ltd_enterpriseai-agenticsystems-aiengineering-activity-7410203928124026880-RBfF) reports its challenge win; the [organizer awards page](https://ikdd.acm.org/cods-2025/awards.php) independently names the matching winning team | Team members: Sai Singireddy, Yasaswini Singanapalli, Shashank T, K. Vamsi Krishna, G. Aditya Ramesh, Soma Prasad. Competition use; not production-readiness evidence. |

## Models, notebooks, and software built on the releases

| Downstream work | Resource and use | Evidence and limits |
| :--- | :--- | :--- |
| **VaidhyaMegha / Samyama — Madhulatha Mandarapu and Sandeep Kunkunuru** | **AssetOpsBench extension and experiments:** adds a typed graph layer and graph-native scenarios | [Paper](https://arxiv.org/html/2605.26874v1), [implementation](https://github.com/samyama-ai/assetops-kg). The paper identifies VaidhyaMegha Private Limited and samyama.ai. A [derived graph dataset](https://huggingface.co/datasets/VaidhyaMegha/assetops-kg) explicitly attributes AOB; its card describes equipment, sensors, work orders, and events. Its adapted scenario set and graph-specific tasks are its own setup; results are not a replacement for the canonical leaderboard. |
| **predictive-maintenance Qwen3-4B LoRA adapter** | **FailureSensorIQ-derived IJCAI data used for training** | [Model card and released weights](https://huggingface.co/predictive-maintenance/qwen3-4b-failuresensoriq-lora). The card reports training on 1,242 labeled validation questions. It explicitly says the adapter has no external held-out score; the cited Kaggle score belongs to another, non-fine-tuned system. Revision `0cbd270d2c71129165e7da98f3adafd825e2ef48` includes adapter weights and configuration. |
| **ASTRIS MiniLM-L6 reranker — DestinyOtto** | **Selected AOB examples used to fine-tune retrieval ranking**, including FailureSensorIQ-style QA, failure mapping, rules, PHM, and asset scenarios | [Model card, weights, and base-versus-fine-tuned metric artifacts](https://huggingface.co/DestinyOtto/astris-minilm-l6-assetops-reranker). Revision `7008d3ad0de4dc0eb986ad2fe9ebb73ddb7dfa1d` includes weights and comparison files. Training and evaluation are publisher-reported, not independently reproduced here. The example inference ID differs from the repository ID; follow the actual released files. |
| **Or4k2! / orecord — Lost-in-the-Middle-Benchmark** | **FailureSensorIQ reuse in a Kaggle robustness experiment:** question position, distractor noise, and optional paraphrasing | [Public notebook](https://www.kaggle.com/code/orecord/lost-in-the-middle-benchmark), version 14, inspected November 9. Section 3 explicitly loads `ibm-research/failuresensoriq`. Saved outputs show 462/600 evaluation jobs and an inconclusive positional effect; describe the experiment, not a definitive performance claim. |
| **Siddharth Srinivasan — Industrial Mind & Code** | **FailureSensorIQ evaluated on local small models** | [First-person experiment](https://www.industrialmindandcode.ai/blog/small-language-models-fault-diagnosis.html) describes 2,667 single-answer questions and separately measures accuracy and output usability. This is an independent blog report, not a peer-reviewed result or live-plant trial. |
| **FedProC — Khoa Nguyen and Taehong Kim** | **AOB telemetry integrated into a federated time-series framework** | [Dataset guide](https://fedproc.readthedocs.io/en/latest/datasets/) and [loader code](https://github.com/nclabteam/FedProC/blob/main/data_factory/AssetOpsBench.py) support chiller, hydraulic-pump, metro-pump, and motor data. Verified software integration; no AOB-specific result table was established in this audit. Underlying datasets retain their own provenance. |
| **Faraday Industrial Benchmark — Faraday-Cage-AI** | **A separately developed industrial benchmark preserves an attributed AOB snapshot** | [Manifest](https://github.com/Faraday-Cage-AI/faraday-industrial-benchmark/blob/main/third_party/manifest.json) pins upstream commit `8ea62868134ba3e9a83fabe91add339a5ec69234` and records included code, scenarios, sample data, and documentation. [Design lineage](https://github.com/Faraday-Cage-AI/faraday-industrial-benchmark/blob/main/docs/design-lineage.md) also acknowledges AOB and PHMForge. Its native results are separate from the compatibility snapshot; do not attribute them to AOB. |
| **PHMForge research adapter — zyqzyq007** | **Community adaptation of PHMForge-style tasks and tools** | [Implementation](https://github.com/zyqzyq007/phmforge-research-adapter) and [validation record](https://github.com/zyqzyq007/phmforge-research-adapter/blob/main/docs/VALIDATION.md). Explicitly partial: numerical subtasks, some alternative datasets, and unavailable original targets. It is not a full official reproduction or canonical PHMForge score. |

### Additional implemented experiments

| Project | Verified code-level reuse | Qualification |
| :--- | :--- | :--- |
| **Shane Peckham — MutaGenAI** | [Planning/prompt-evolution recipe](https://github.com/shanepeckham/MutaGenAI/blob/main/examples/cookbook/prompt_evolution_assetops.py) loads AOB scenarios; [experiment scripts](https://github.com/shanepeckham/MutaGenAI/tree/main/examples/experiments/assetops) and [saved log](https://github.com/shanepeckham/MutaGenAI/blob/main/logs/assetops_track1_log.json) use the benchmark to demonstrate evolutionary prompt search | Code also has embedded fallback scenarios and a custom plan scorer. Saved log does not establish which source path was used; do not present its scores as canonical AOB results. No employer adoption claim is made. |
| **jagadish-volvo / thesis-memory-optimization** | [AdaptEvolve experiment](https://github.com/jagadish-volvo/thesis-memory-optimization/blob/main/experiment5_adaptevolve.py) loads FailureSensorIQ and removes choices for free-response model-routing experiments | Modified task and grading; implementation verified, outcome not independently reproduced. The account name alone does not establish Volvo sponsorship or corporate use. |
| **SeaFup / Multiagents-three-modes** | [Scenario loader](https://github.com/SeaFup/Multiagents-three-modes/blob/main/src/assetops_local/scenarios.py) fetches the AOB `scenarios` configuration; [project account](https://github.com/SeaFup/Multiagents-three-modes) compares planning/orchestration modes | Code and local-output organization verified. An inspected archived summary was empty; no score is promoted from it. |
| **Qwer-11-max / SelfGrowthGraphRag** | [Experiment code](https://github.com/Qwer-11-max/SelfGrowthGraphRag/blob/main/noRag/paraQuestion.py) reads a locally stored FailureSensorIQ single-answer CSV for question/option experiments | Local data path demonstrates intended integration, not a verified completed run or a public dataset snapshot. |

These records establish released artifacts or first-person experiment reports. We inspected their documentation and relevant code; we did not run the downstream training or evaluation workloads.

## Experimental reuse and extensions connected to the ecosystem

| Research | Connection and resource | What it enables researchers to investigate |
| :--- | :--- | :--- |
| [DynAMO](https://arxiv.org/abs/2606.19382) | Kanishk Kushwaha, Vikrant Vinod Bansode, Harsh Vardhan, Dhaval C. Patel; shares an AssetOpsBench author | Evaluates sequential and dependency-aware parallel workflows on AssetOpsBench. [Public code](https://github.com/kushwaha001/DynAMO). |
| **Chand Sahil Mansuri; Sadamori Kojaku — teams4industry** | Binghamton calibrated-coordination project; **FailureSensorIQ adapter verified** | [Adapter](https://github.com/skojaku/teams4industry/blob/main/libs/teams/teams/benchmarks/__init__.py) converts single-answer records to a shared task schema. [Experiment snapshot](https://github.com/skojaku/teams4industry/tree/main/exps/2026-0730-calibrated-coordination) includes runners but states saved run logs are absent. Its AOB adapter explicitly raises `BenchmarkNotRunnable`; do not describe a completed AOB evaluation. |
| [ReasonLab / earlier title Simplicity Paradox](https://arxiv.org/abs/2607.14109) | Inder Preet, Shuxin Lin, Dhaval Patel; shared authorship | [Version 1, Section 3.1 and Appendix D](https://arxiv.org/html/2607.14109v1#S3.SS1) includes a 500-question FailureSensorIQ sample in prompting experiments. Current version 2 uses the title **ReasonLab**; this is one work, not two citations. |
| **Exgentic evaluation containers** | IBM-connected evaluation tooling; **AOB packaging** | [Container documentation](https://github.com/Exgentic/eval-containers/tree/main/containers/benchmarks/assetopsbench) specifies a pinned dataset revision and 152 tasks. Packaging is verified; the adapter is a separate evaluation setup, not evidence of independently reproduced canonical scores. |
| [Transduction is All You Need for Structured Data Workflows](https://arxiv.org/abs/2508.15610) | IBM Agentics research by Alfio Gliozzo and colleagues; shares a FailureSensorIQ co-author | [Section 5.2](https://arxiv.org/html/2508.15610v3#S5.SS2) evaluates structured workflows and perturbation robustness on FailureSensorIQ. This is verified reuse, with shared authorship; classify separately from external use. |
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
| **AssetOpsBench** | Good Start Labs reported transfer evaluation; GyanData competition agents; multi-CaMeL experiments; Samyama graph extension and derived dataset; ASTRIS training; FedProC telemetry integration; Faraday snapshot; connected small-model and orchestration extensions; LG discussion. |
| **FailureSensorIQ** | External multi-agent reasoning, Industrial-Instruction, and retrieval/fine-tuning experiments; LoRA adapter; Kaggle robustness notebook; local-model blog experiment; ASTRIS training; teams4industry adapter; Agentics/ReasonLab connected experiments. |
| **DiagnosticIQ** | CoffeeBench citation; [R2A-Sales design note](https://github.com/qinyh10300/R2A-Sales-Benchmark/blob/main/docs/RULE_TO_ACTION_GAP_CN.md) references its rule-to-action framing without establishing dataset reuse. [Author-maintained code](https://github.com/DevinDeSilva/DiagnosticIQ) was located. No separate external experiment verified. |
| **PHMForge** | Community research adapter with explicit partial scope; Faraday design-lineage acknowledgement; CoffeeBench and ADMITBench discussion. The [instructor account](https://kaoutarelmaghraoui.com/blog/hpml-industrial-mentorship-arxiv/) also documents its educational/research origin; that is project-connected evidence. |
| **IoTBench / ReActXen** | The [original evaluation](https://aclanthology.org/2025.emnlp-industry.24/) and an [external Japanese paper introduction](https://note.com/tmago_business/n/nd61d5072b7d1) were located. [BTS-AgentBench](https://arxiv.org/html/2608.27334v1#S2) cites the IoT paper as related work. A separate external experiment was not verified. |
| **IndustryAssetEQA** | [Published study and released branch](https://aclanthology.org/2026.acl-industry.49/); [embodied-VQA survey, Section 5.7/reference 91](https://www.mdpi.com/2673-2688/7/9/367). External survey citation verified; separate external experimental reuse not verified. |

Evidence will differ across releases and publication dates. “Not verified in this audit” is not a claim that no such use exists.

## Reporting impact consistently

Use **evaluated on** for documented experiments, **extended** for new artifacts or methods built on a release, **cited** for references or related-work discussion, and **discussed** for coverage. Record the study version, exact resource, authors, author overlap, and evidence location. Use a named bibliographic index and retrieval date for any future citation totals; this register is a curated set of examples, not a citation count.

## Company and industry coverage

| Publisher or affiliation | Evidence | What can be said |
| :--- | :--- | :--- |
| **Good Start Labs** | [Alex Duffy’s experiment report](https://every.to/playtesting/we-trained-an-ai-on-a-board-game-it-became-a-better-customer-support-agent-299b5938-09dd-4881-803f-aea21f0d461f) | Company-reported evaluation on AOB after game training; quantitative AOB results not supplied. |
| **GyanData** | [Organizer award](https://ikdd.acm.org/cods-2025/awards.php) and [company post](https://www.linkedin.com/posts/gyandata-pvt-ltd_enterpriseai-agenticsystems-aiengineering-activity-7410203928124026880-RBfF) | Verified winning team and company-reported competition participation. |
| **VaidhyaMegha / Samyama** | [Graph extension](https://arxiv.org/html/2605.26874v1) | Company-affiliated authors built and evaluated an AOB extension. |
| **AI Sequrity Company** | [multi-CaMeL author block and Section V-A](https://arxiv.org/html/2610.05640v1#S5.SS1) | Ilia Shumailov is a co-author of a study using AOB for utility evaluation. Paper-level experimental use; no corporate deployment claim. |
| **Refiant, Inc.** | [ADMITBench, Section 3.2](https://arxiv.org/html/2608.03866v1#S3.SS2) | Yash Misra and Siddharth Gutta are Refiant-affiliated co-authors; cites AOB and PHMForge while describing its own framework. |
| **Solution Seeker AS** | [Virtual-flow-meter paper](https://arxiv.org/html/2610.19860v1) | Kristoffer Hjelmen Nesland and Bjarne Grimstad have company affiliations; introduction cites AOB. Experiments use a separate generated dataset. |
| **Infineon, Signify, Statwolf, AITIA** | [AIMS5.0 paper](https://real.mtak.hu/249438/1/AIMS_SpecIss_ICJ_2026_1_6.pdf), author block, related work and reference 58 | Affiliations correspond to Marta Bonik, Chris Damkat, David Dandolo/Chiara Masiero, and Dániel Ficzere. Citation in jointly authored industry research, not verified AOB execution. |
| **LG AI Research** | [AEGIS research blog](https://www.lgresearch.ai/blog/view?seq=633) | Dedicated discussion and prospective evaluation interest. |
| **EQUA AI** | Founder Srikant Naidu's [autonomy guide](https://equa.work/resources/grading-the-autonomy-claim) and [work-order guide](https://equa.work/resources/work-order-history-what-it-can-do) | Company-authored practitioner discussion citing AOB as evaluation evidence. |
| **Lancartech** | [Industrial-agent retrospective](https://lancartech.co.id/en/blog/assetopsbench-ibm-retrospektif-evaluasi-agent-industri) | Company-site commentary based on IBM's earlier account. |
| **HysenLabs** | [AssetOpsBench review](https://hysenlabs.com/projects/ibm-assetopsbench) | External editorial analysis of the repository and adoption considerations. |
| **Meta and LMI Consulting affiliations** | Confucius and CoffeeBench papers above | Author affiliations on citing research, not company adoption announcements. |

This is a verified selection, not a census of company mentions. Commercial summaries and directory listings belong in the [coverage inventory](COVERAGE_AND_EVENTS.md); none of these records alone establishes a customer relationship, partnership, or deployment. The [citation audit](CITATION_AUDIT.md) preserves Scholar records and follow-up leads without inflating the README.

## Additional company-affiliated research citations

**Siemens AG and Artiquare GmbH:** David Kube and Asaad Almutareb are respectively affiliated in [Foundation-Model-Based Agents in Industrial Automation](https://arxiv.org/html/2605.02592v1); Section 4.1 cites AOB in its review corpus. **Huawei Technologies:** the author block of [Empowering Real-World](https://arxiv.org/html/2510.17491v1) identifies Huawei collaborators; its evaluation discussion cites AOB (reference 159). These establish inclusion in industrial-agent research reviews, not company experiments on AOB.

## Researchers: document the contribution, not just the name

**Ilia Shumailov and Christian Schroeder de Witt** co-authored the external AOB security/utility study; **Moharram Challenger** co-authored Industrial-Instruction's FailureSensorIQ evaluation. **Sadamori Kojaku's** connected project has a verified FailureSensorIQ adapter, with the execution limits described above. **Amit Sheth, Deborah L. McGuinness, and Kaoutar El Maghraoui** have documented family-development or mentorship roles in the [contribution directory](docs/university-projects.md#named-contributions-and-evidence).

For broader research reach, **Christian Holz** co-authored TraceBench, **Dacheng Tao** co-authored FORGE, and **Minlan Yu / Soudeh Ghorbani** co-authored Confucius. Full-text checks establish related-work citations, not experiments on this family's data. Enrico Zio's survey remains a citation lead with publisher metadata verified but full citation context unavailable; it is not promoted as dataset use. See the [audit dispositions](CITATION_AUDIT.md#disposition-of-the-original-20-leads).

Source revisions for the released models and selected integrations are preserved in [research-use-evidence.json](assets/research-use-evidence.json).
