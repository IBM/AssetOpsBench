# Citation discovery and verification

Reviewed November 10, 2026 using Google Scholar's “Cited by” lists and primary papers. This is a dated research-discovery audit, not a unique-citation total. Keyword search results are not citation counts. Scholar may split proceedings, preprints, and changed titles into different records, and may index blogs or patents alongside papers.

## Records inspected

| Resource / Scholar record | Displayed citing records | Discovery link |
| :--- | ---: | :--- |
| AssetOpsBench preprint | 31 | [Citing records](https://scholar.google.com/scholar?cites=3506473653926276354&hl=en) |
| AssetOpsBench KDD proceedings | 6 | [Citing records](https://scholar.google.com/scholar?cites=4319209530128476739&hl=en) |
| FailureSensorIQ | 17 | [Citing records](https://scholar.google.com/scholar?cites=16528683271765767035&hl=en) |
| PHMForge, current title | 1 | [Citing records](https://scholar.google.com/scholar?cites=14992453025155493477&hl=en) |
| PHMForge, earlier title | 3 | [Citing records](https://scholar.google.com/scholar?cites=9384586252328081340&hl=en) |
| DiagnosticIQ | 1 | [Citing records](https://scholar.google.com/scholar?cites=9591820185078742697&hl=en); CoffeeBench |
| IndustryAssetEQA | 1 | [Citing records](https://scholar.google.com/scholar?cites=5941950652704872380&hl=en); embodied-VQA survey |
| ReAct Meets Industrial IoT / IoTBench | 8 | [Citing records](https://scholar.google.com/scholar?cites=7720376694480592636&hl=en); includes duplicate AOB versions and connected papers |

**Do not add these numbers.** Versions overlap, one citing work can reference several family members, and the PHMForge current-title record includes a project-connected mentorship blog. FailureSensorIQ's list also contains multiple versions of individual studies. No aggregate citation badge is proposed from these raw counts.

## Verified examples suitable for the README

The [research-use register](RESEARCH_USE.md) supplies people, author relationships, and evidence locations for:

- External experiments: **Can CaMeLs Talk?**, **Harnessing collective intelligence**, **Industrial-Instruction**, and **Retrieval-based and Fine-tuned LLM Approaches**.
- Connected experiments and extensions: **Agentics**, **DynAMO**, the knowledge-graph extension, the Columbia tool-knowledge project, and **Trajel**. Shared authorship is identified rather than counted as independent validation.
- External citations: **Confucius** and **CoffeeBench**. A citing author's employer is not evidence of company deployment.
- Company discussion: **LG AI Research**, **EQUA AI**, and **Lancartech**; external editorial review from **HysenLabs**. LG's stated evaluation interest remains prospective.

## Disposition of the original 20 leads

Full text was inspected for **16 of the 20 original leads**. Four remain limited by inaccessible full text or unresolved primary records. “Citation” below means a verified reference/discussion; it does not imply dataset reuse.

| Original lead | Disposition after primary-source review | Evidence location |
| :--- | :--- | :--- |
| AI-Augmented SOC | **Survey citation** of AOB; no new AOB experiment | [Sections 3.4/4 and reference 53](https://www.mdpi.com/2624-800X/5/4/95) |
| TraceBench — Tommaso Bendinelli, Artur Dox, Christian Holz | **Related-work citation**; experiments use its physical simulators | [Section 2.1](https://arxiv.org/html/2608.27182v1#S2.SS1) |
| Capable but Unreliable | **Related-work citation**; experiments use Toolathlon | [Section 2 and Section 3.1](https://arxiv.org/html/2602.19008v1#S2). Its description of AOB's tool count is not adopted as a fact here. |
| IndustryBench | **Related-work citation**; its procurement-QA dataset is separate | [Section 2](https://arxiv.org/html/2605.10267v1#S2) |
| ADMITBench — Yash Misra, Javal Vyas, Siddharth Gutta, Mehmet Mercangöz | **Cites AOB and PHMForge**; separate action-admissibility framework | [Section 3.2](https://arxiv.org/html/2608.03866v1#S3.SS2); affiliations identify Refiant and Imperial College London |
| Evidence-Driven Reasoning for Industrial Maintenance | **Connected citation**, not independent reuse; shares AOB authors | [ACL paper](https://aclanthology.org/2026.acl-industry.50.pdf), Section 2 and references; IBM/Imperial affiliations. Reports its own CMMS case studies. |
| Forecast Workflow Bench | **Related-work citation**; separate decision-focused forecast benchmark | [Section 2 and reference 14](https://arxiv.org/html/2609.27385v1#S2) |
| Text-Informed Training Data Filtering | **Citation by company-affiliated authors**, not AOB execution | [Introduction/reference 11](https://arxiv.org/html/2610.19860v1); Solution Seeker AS affiliations; experiments use generated flow-meter data |
| FacProcessTwin | **Related-work citation**; independent manufacturing case study | [Section II/reference 15](https://arxiv.org/html/2606.17666v1#S2); Swinburne authors |
| Imagine Before You Act | **Unresolved full text**; no reuse claim added | [ACM DOI lead](https://doi.org/10.1145/3841560.3841567); primary page unavailable and focused title search did not resolve an accessible manuscript |
| Hard-CAD | **Unresolved full text**; no reuse claim added | [Publisher lead](https://www.sciencedirect.com/science/article/pii/S0097849326002979); publisher returned access denial |
| AIMS5.0 multi-case experiences | **Industry research citation**, not AOB experiment | [PDF](https://real.mtak.hu/249438/1/AIMS_SpecIss_ICJ_2026_1_6.pdf), printed p.138/reference 58; company affiliations on p.135 |
| BTS-AgentBench — J. Y. Kim | **Cites both AOB and ReAct Meets Industrial IoT**; own telemetry corpus | [Section 2 and comparison appendix](https://arxiv.org/html/2608.27334v1#S2) |
| TUM preventive-maintenance thesis — Ruchit Kumar Patel | **Methodological use:** evaluation design explicitly responds to AOB/PHMForge findings; no dataset reuse established | [Thesis](https://mediatum.ub.tum.de/1865598), Section 2.7.2 (printed pp.21–22), references 83–84; TUM supervisors Felix Dietrich and Vladyslav Fediukov |
| FORGE | **FailureSensorIQ comparison and citation**, not reuse in its experiments | [Final EMNLP Findings paper](https://aclanthology.org/2026.findings-emnlp.888.pdf), Table 1/Section 2/references. Includes **Dacheng Tao**. Earlier audit incorrectly called the venue Findings of ACL; corrected to **Findings of EMNLP 2026**. |
| CemEval construction-engineering benchmark | **Publisher abstract verified; full reference/experiment context unresolved** | [Publisher](https://www.sciencedirect.com/science/article/pii/S0926580526003420). Abstract describes its own regulatory dataset; do not infer FailureSensorIQ reuse from Scholar alone. |
| Simplicity Paradox | **Connected experimental reuse** of FailureSensorIQ; not independent | [v1 Section 3.1/Appendix D](https://arxiv.org/html/2607.14109v1#S3.SS1); Inder Preet, Shuxin Lin, Dhaval Patel. [Current v2 title is ReasonLab](https://arxiv.org/abs/2607.14109); one work. |
| From Question Answering to Task Completion | **PHMForge survey citation**, not new benchmark execution | [Section 6.1/reference 183](https://arxiv.org/html/2606.20683v1) |
| PHM foundation-model survey — Te Han, Xiao Wang, Huixing Meng, Enrico Zio | **Publisher metadata verified; full PHMForge citation context unresolved** | [IEEE author/abstract record](https://ieeexplore.ieee.org/abstract/document/11715714/authors), DOI 10.1109/TII.2026.3732729. No claim that Zio ran PHMForge. |
| Locally Runnable LLMs for Bearing Fault Diagnosis — M. H. Shawon, P. Kumar | **Methodological influence and discussion** of PHMForge; experiments use Paderborn, CWRU, HUST | [Section 2.4/reference 4](https://www.mdpi.com/2079-3197/14/9/220). Its reported PHMForge comparison uses different tasks and is not a controlled head-to-head result. |

## New reuse found beyond citation indexes

The [research-use register](RESEARCH_USE.md#models-notebooks-and-software-built-on-the-releases) records the evidence and limitations for the Samyama graph extension, two released fine-tuned models (the FailureSensorIQ LoRA adapter and ASTRIS reranker), Or4k2!'s Kaggle notebook, Siddharth Srinivasan's local-model blog experiment, FedProC telemetry loaders, Faraday's attributed AOB snapshot, and the community PHMForge research adapter.

The connected-work section adds Sadamori Kojaku's **teams4industry** adapter, **ReasonLab**, and **Exgentic** packaging. An implemented loader establishes integration; missing logs or a stub that raises `BenchmarkNotRunnable` do not establish completed evaluation.

Public GitHub code searches inspected the first 100 matching files for AOB and FailureSensorIQ, plus PHMForge and DiagnosticIQ matches. Searches also covered Scholar citing lists for all six families, publisher full text, Kaggle, model/dataset cards, public Hugging Face API searches, blogs, and company sites. These are bounded discovery searches, not exhaustive counts of adoption.

An additional search for exact Hugging Face dataset IDs in Python code located **MutaGenAI**, **thesis-memory-optimization**, **Multiagents-three-modes**, and **SelfGrowthGraphRag**. Their loaders were inspected; task changes, fallback data, custom scorers, and missing output artifacts are recorded in the [additional experiment table](RESEARCH_USE.md#additional-implemented-experiments). Code presence is kept distinct from a verified run. Account names and personal projects are not attributed to employers.

## Other dispositions and exclusions

| Source | What was established | Reporting decision |
| :--- | :--- | :--- |
| [IndustryAssetEQA embodied-VQA survey](https://www.mdpi.com/2673-2688/7/9/367) | Section 5.7/reference 91 discusses IndustryAssetEQA | Verified external citation; no new dataset experiment |
| [R2A-Sales](https://github.com/qinyh10300/R2A-Sales-Benchmark/blob/main/docs/RULE_TO_ACTION_GAP_CN.md) | Section 9.1 acknowledges DiagnosticIQ's rule-to-action framing | Conceptual reference; no DiagnosticIQ data reuse established |
| [Cymatix's AOB assessment](https://github.com/mbachaud/Cymatix-Context/blob/main/docs/benchmarks/2026-07-13-assetopsbench-eval-verdict.md) | Describes cloning AOB and analyzing a 152-scenario snapshot, then declining a planned evaluation | Dataset inspection and critical technical discussion, not a completed benchmark run or endorsement; its technical judgments are the author's |
| [Faraday design lineage](https://github.com/Faraday-Cage-AI/faraday-industrial-benchmark/blob/main/docs/design-lineage.md) | Credits AOB and PHMForge; manifest verifies copied AOB snapshot | Artifact reuse verified; native Faraday results must remain separate |
| [yushengli/PHMForge](https://huggingface.co/datasets/yushengli/PHMForge) | Card explicitly identifies itself as a mirror of the canonical IBM dataset | Mirror is not an independent adopter or experiment |
| DeveloperMindset123/PHMForge_Scenarios | API search locates the repository; card access is restricted | No dataset-use claim from inaccessible material |
| [Qwen3.5 PTL AssetOps model](https://huggingface.co/davidfregoli/qwen3.5-4b-ptl-assetops) | Card describes its own AssetOps API and training, but does not establish IBM AOB provenance | Similar naming is insufficient; excluded from verified family reuse |
| [arifOS bridge](https://github.com/ariffazil/A2B/blob/main/docs/ASSETOPSBENCH_BRIDGE.md) | Design note describes a bridge but lists full AOB integration as future work | Not promoted as completed benchmark use |
| [AlphaEval catalog](https://github.com/GAIR-NLP/AlphaEval/blob/main/BENCHMARKS.md) | Lists AOB in a benchmark comparison table | Catalog inclusion, not a run or dataset derivative |

## Final search pass — November 9, 2026

The final pass searched all six family names plus the `assetops` alias across Hugging Face **models, datasets, and Spaces** (21 public API searches), and checked model dataset tags for the three public IBM datasets. Discovery searches are name/tag-based, not a full-text census of every Hub repository. Author-excluded Scholar searches were reviewed across all four pages of roughly 35 AOB matches and both pages of 13 FailureSensorIQ matches. Author filtering is only a discovery heuristic; overlap and actual reuse were checked separately.

| New source | Verified disposition |
| :--- | :--- |
| [Good Start Labs / Every article](https://every.to/playtesting/we-trained-an-ai-on-a-board-game-it-became-a-better-customer-support-agent-299b5938-09dd-4881-803f-aea21f0d461f) | First-person AOB evaluation report after Diplomacy fine-tuning, corroborated as the company's work by its [research index](https://goodstartlabs.com/research). No AOB numeric lift or reproducible protocol disclosed; not an AOB training-data claim. |
| [GyanData company post](https://www.linkedin.com/posts/gyandata-pvt-ltd_enterpriseai-agenticsystems-aiengineering-activity-7410203928124026880-RBfF) | Competition-use claim with team identity cross-checked against [organizer awards](https://ikdd.acm.org/cods-2025/awards.php); company product-readiness language is not adopted. |
| [VaidhyaMegha/assetops-kg](https://huggingface.co/datasets/VaidhyaMegha/assetops-kg) | Derived graph dataset with explicit AOB source attribution; revision `a3611fb500675d8cc569dfbf6461d5234d5afe70`. Samyama's later paper lists FailureSensorIQ evaluation as future work, so no completed FSIQ experiment is inferred. |
| [Transformer scenario dataset](https://huggingface.co/datasets/rohithkanathur/assetopsbench-transformer-scenarios-dataset) | 50 released extension scenarios; connected Columbia project already listed in the university directory. Revision `e3a0664ca455eb58be5fc5f3298bb06691cbb9bf`. Do not count it as an additional independent research group. |
| zhansingsong/AssetOpsBench; MAOhih/AssetOpsBench | Cards reproduce AOB descriptions/configurations without documenting a new study; no independent experiment claimed. |
| [cc4718/FailureSensorIQ Space](https://huggingface.co/spaces/cc4718/FailureSensorIQ) | Paper-linked leaderboard implementation; card alone does not establish an external user or new experiment. |
| [Authentic but Not Neutral](https://aclanthology.org/2026.realm-1.8.pdf) — Xiaoxuan Li, Bangbang Liu, Ying Zhao, Shanghai AI Laboratory | Appendix E.4/Table 14 uses published AOB-Live trace-cost summary statistics. It explicitly says those traces are not linked to the outcomes needed for rank analysis. This is secondary evidence use, not raw-dataset re-evaluation. |
| [Foundation-Model-Based Agents in Industrial Automation](https://arxiv.org/html/2605.02592v1) | Section 4.1/reference includes AOB; Siemens AG/Artiquare affiliations verified. Survey corpus inclusion, not an AOB experiment. |
| [Empowering Real-World](https://arxiv.org/html/2510.17491v1) | Evaluation discussion/reference 159 cites AOB; Huawei author affiliations verified. Survey citation, not corporate deployment. |
| [DynaContext](https://arxiv.org/html/2608.22014v1) | AOB-Live related-work citation; extraction experiments use separate component benchmarks. |
| [Task-Aware Harness Provisioning](https://arxiv.org/html/2608.17433v1) | AOB/competition-retrospective discussion; experiments use its own constructed infrastructure tasks. |
| [Relay-Bench](https://arxiv.org/html/2607.18438v1) | Cites AOB Stirrup-agent documentation; no AOB dataset experiment established. |
| [Large-scale factor analysis](https://arxiv.org/html/2609.36515v1) | FailureSensorIQ reference verified; exact inclusion of its scores in the analysis was not established from that reference alone. |
| Carbon-m1 / Generating Stable Materials search matches | Search results resolve to a multi-paper IBM conference page that also contains FailureSensorIQ. Co-location is insufficient evidence of citation or reuse; excluded. |
| CertiGATE (SSRN 7529119); learned work-order dispatching (SSRN 7095162) | Scholar discovery leads; primary pages returned access denial. No dataset-use claim promoted. |

The four original unresolved full-text leads remain unresolved. Crossref confirms the titles/DOIs for the world-model and PHM-survey records, but metadata does not establish their experiments. No access restrictions were bypassed. The search is broad and reproducible in scope, not a claim to have found every use on the internet.

## Profile-based citation follow-up — November 9, 2026

The [supplied Google Scholar profile](https://scholar.google.com/citations?hl=en&user=c06_VJAAAAAJ&view_op=list_works&sortby=pubdate) was expanded to 100 publication rows. Family-related citation trails were checked for Trajel (five listed citing records), Evoflux (four), CodeReAct (eight), and the first ten of eleven SPIRAL records. These are discovery records, not additional unique family citations: the profile includes unrelated research, shared-author work, and duplicate publication versions. Profile-wide citation totals are not AssetOpsBench impact metrics.

| Primary source checked | What the evidence supports |
| :--- | :--- |
| [IndustryBench](https://arxiv.org/html/2605.10267v1#S2) | Section 2 cites AOB as related work. Its procurement-QA experiments use IndustryBench's own data; now highlighted near the top of the README as research reach. |
| [Failing Agents Leave Traces](https://aclanthology.org/2026.realm-1.56/) | Amazon-affiliated author Muthaheera Yasmeena Belgur Shamiullah cites Trajel in related work and limitations. Its own telemetry study does not establish reuse of Trajel traces or AOB tasks. |
| [FiMI Banking](https://arxiv.org/html/2609.03960v1) | NPCI AI Research Team cites Evoflux (reference 28). A method citation, not an AOB dataset experiment. |
| [DeMTS](https://arxiv.org/html/2608.14632v1) | Shirui Pan and collaborators cite Trajel in the introduction; no Trajel dataset experiment established. |
| [Uncertainty Quantification for LLM Agents](https://arxiv.org/html/2609.07395v1) | Section 8.1.1/Table 17 discusses Trajel for failure attribution; citation rather than verified trace reuse. |
| [Inherit-MAS](https://arxiv.org/html/2610.02396v1) | Cites Evoflux; experiments use WorkBench and HotpotQA, not AOB. |
| [AdvPlan-Bench](https://arxiv.org/html/2608.00832v1) and [Learning Compositional Meta-Routing](https://arxiv.org/html/2608.00106v1) | Cite SPIRAL as planning-related work; their own task collections do not establish AOB data reuse. |

This November 9 snapshot left several primary-source checks open. Their November 10 dispositions and access limitations are recorded below; no dataset-reuse claim is inferred from an inaccessible lead.

## Remaining-lead review and expanded GitHub scan — November 10, 2026

This follow-up rechecked the outstanding leads from the six-family audit and related-method citation trails. The SPIRAL Scholar list was revisited on both pages: it now displays 12 records, including the previously unchecked **MCP-Cosmos** and a new SSRN path-planning record. This is another dated discovery snapshot, not 12 additional AOB dataset users.

| Follow-up source | Verified disposition |
| :--- | :--- |
| [ToolForge](https://aclanthology.org/2026.emnlp-main.1250.pdf) | Full PDF checked. Section 2 cites SPIRAL; author block includes **Meituan**. Its synthesis/training uses separate virtual-tool data, not AOB datasets. |
| [Cost-Awareness in Tree-Search LLM Planning, v2](https://arxiv.org/html/2505.14656v2) | Current revision checked, resolving the changed-title lead. Section 2 references SPIRAL; its experiments use its own planning setting. The earlier v1 inspection alone could not resolve this citation. |
| [The Evolution of Tool Use in LLM Agents](https://arxiv.org/html/2603.22862) | Section 3.3/reference 223 discusses SPIRAL as reflective planning. Survey citation, not benchmark execution. |
| [MetaRoute-Bench](https://arxiv.org/html/2608.00107v1) and [Adversarial Course-of-Action Generation](https://arxiv.org/html/2609.26059v1) | Related-work discussions cite SPIRAL. Their own routing/planning workloads are separate from the AOB family. |
| [MCP-Cosmos](https://arxiv.org/html/2605.09131v1) | Connected method extension: Giridhar Ganapavarapu and Dhaval Patel evaluate a SPIRAL-Exec configuration with world models on MCP-Bench. Verified method reuse, not a new external AOB dataset experiment. |
| [Imagine Before You Act](https://doi.org/10.1145/3841560.3841567) | The [publisher-deposited Crossref reference list](https://api.crossref.org/works/10.1145/3841560.3841567) explicitly contains AOB. Bibliographic citation verified; full-text experiment context remains inaccessible. No dataset-use claim. |
| [Large-scale factor analysis](https://arxiv.org/html/2609.36515v1) | Rechecked full text; FailureSensorIQ remains a verified reference. Its per-benchmark score inclusion was not resolved, so no new raw-data or score-reuse claim is promoted. |

### Access-limited leads after follow-up

| Lead | Remaining limitation |
| :--- | :--- |
| Imagine Before You Act; Hard-CAD; CemEval; Han/Wang/Meng/Zio PHM foundation-model survey | Original four still lack inspectable full-text experiment context. ACM/Elsevier requests were unavailable; IEEE presented browser verification. Metadata, abstracts, and the Imagine bibliography do not establish dataset reuse. |
| [TrustTrajectory](https://openreview.net/forum?id=u8TM3hZont) | OpenReview still presents browser verification. |
| SSRN: [Vigilance Unbundling](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7406038), [Dependency-Complete Tool Discovery](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7620582), [CertiGATE](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7529119), [learned work-order dispatching](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7095162), [Legality, Not Routing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7556542) | Primary requests failed or returned access denial; citation context and experiments remain unverified. |
| [Semantic-collapse industrial-agent study](https://doi.org/10.1016/j.asoc.2026.116428) | Publisher preview describes a separate Green Button task set; full CodeReAct citation context remains unavailable. |
| [Learning from supervision with semantic and episodic memory](https://doi.org/10.1145/3786335.3813154); [LLMs in Evolutionary Computation](https://ieeexplore.ieee.org/abstract/document/11755706/); [From Impasse to Adaptation](https://ojs.aaai.org/index.php/AAAI-SS/article/view/43399) | Scholar leads checked; primary full text could not be retrieved in this pass. No attribution of AOB data use to their authors or institutions. |

These are checked but unresolved leads, not confirmed negative results. No access challenge was bypassed.

### GitHub: code, notebooks, data attribution, and links

Fourteen completed GitHub code searches covered **all six family names/aliases**, exact Hugging Face IDs for AOB/FailureSensorIQ, Python files, and notebooks. They returned **1,542 file hits / 1,309 distinct repository-path pairs across 293 repositories** before relevance filtering. These are discovery volumes, including unrelated IoTBench/DiagnosticIQ names, official code, repeated versions, catalogs, and mirrors. They are **not adoption counts**, and not every returned file was read in full. Search caps and selected inspected revisions are saved in the [search snapshot](assets/github-discovery-snapshot.json). GitHub's legacy code-search index is incomplete, particularly for forks; the scan is bounded.

New verified evidence in the [research-use register](RESEARCH_USE.md) includes **Pesquisa_IC_IA QLoRA training**, **shopfloor-agent's pinned AOB sample-data evaluation**, **A2B's FailureSensorIQ harness and saved ablations**, **EviPlan's attributed software extension**, **InspectionBench's infrastructure/benchmark extension**, and **BESS-Bench's connected PHMForge-aligned battery scenarios**. A first-person AOB smoke test in `tesla_agent` and SelfGrowthGraphRag's experiment journal are qualified separately. None of these account names establishes an employer's use.

Additional checked dataset links are retained in the [GitHub discovery table](RESEARCH_USE.md#github-dataset-links-and-discovery). Notable exclusions: TORQ's attributed data directories contain placeholders; PdMData explicitly lists AOB/FSIQ outside supported adapters; ContextLab's archived dataset-resolution record is not a completed experiment; RuiSue719's three apps recommend FSIQ without an inspected FSIQ loader; `hierarchical-graphs` mentions FSIQ in notebook reading material; SimulAgent's feasibility report explicitly defers implementation; `justsearch` labels its PHMForge comparison `self_reproduced: false`; `industrial-copilot` uses AI4I data and its AOB coverage note is not an AOB run. InspectionBench's related review package and the mirrored HPML smart-grid repositories are not counted twice.

## Refresh procedure

Open each canonical publication and its preprint record; export or record the citing works; deduplicate by DOI, arXiv ID, and title. Check primary full text for the resource name, methods, tables, and bibliography. Record author overlap before calling use external. Count distinct works separately from distinct citing organizations. Keep coverage, citations, experiments, and extensions as separate measures. Record retrieval date and source whenever publishing a count.
