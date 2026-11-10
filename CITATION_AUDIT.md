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

The connected-work section records **ReasonLab** and **Exgentic** packaging. Listed university projects are distinguished from publicly inspectable software artifacts and completed evaluations.

GitHub discovery searches inspected the first 100 matching files for AOB and FailureSensorIQ, plus PHMForge and DiagnosticIQ matches. Searches also covered Scholar citing lists for all six families, publisher full text, Kaggle, model/dataset cards, public Hugging Face API searches, blogs, and company sites. These are bounded discovery searches, not exhaustive counts of adoption.

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

New verified evidence in the [research-use register](RESEARCH_USE.md) includes **Pesquisa_IC_IA QLoRA training**, **shopfloor-agent's pinned AOB sample-data evaluation**, **A2B's FailureSensorIQ harness and saved ablations**, **EviPlan's attributed software extension**, **InspectionBench's infrastructure/benchmark extension**. A first-person AOB smoke test in `tesla_agent` and SelfGrowthGraphRag's experiment journal are qualified separately. None of these account names establishes an employer's use.

Additional checked dataset links are retained in the [GitHub discovery table](RESEARCH_USE.md#github-dataset-links-and-discovery). Notable exclusions: TORQ's attributed data directories contain placeholders; PdMData explicitly lists AOB/FSIQ outside supported adapters; ContextLab's archived dataset-resolution record is not a completed experiment; RuiSue719's three apps recommend FSIQ without an inspected FSIQ loader; `hierarchical-graphs` mentions FSIQ in notebook reading material; SimulAgent's feasibility report explicitly defers implementation; `justsearch` labels its PHMForge comparison `self_reproduced: false`; `industrial-copilot` uses AI4I data and its AOB coverage note is not an AOB run. InspectionBench's related review package and the mirrored HPML smart-grid repositories are not counted twice.

## Fork and linked-Hugging-Face follow-up — November 10, 2026

The [saved audit](assets/fork-model-audit.json) covers public fork listings for AOB, FailureSensorIQ, ReActXen, the PHMForge implementation, and EmbodiedQA. The latter complements IndustryAssetEQA's branch inside AOB. The public listings returned **372 forks**: 359 AOB, 3 FSIQ, 7 ReActXen, 3 PHMForge, and none for EmbodiedQA; a separate DiagnosticIQ code endpoint was not publicly accessible. Fork totals are discovery scope, not dataset adoption.

- Attempted default-branch comparisons for every listed fork: **364 succeeded; 8 failed**. Six failures had no common ancestor; their READMEs were inspected separately. Two repositories could not be resolved. History mismatches are not treated as evidence of new reuse.
- Listed branches for the **120 forks whose metadata showed a push after creation**, then inspected selected project branches and artifacts. Inherited upstream branches, maintenance fixes, unchanged copies, and duplicate course-team forks are not new adopters. This does not constitute a full review of every branch or commit.
- Ran **21 HF model-discovery queries**: nine names/aliases, three canonical dataset tags, seven publication tags, and two linked-dataset tags. Followed model links from project READMEs; inspected six cards/revisions and their file listings, including the two previously recorded LoRA/reranker releases. Empty name/tag searches do not prove no untagged model exists.

The [research-use register](RESEARCH_USE.md#forks-and-linked-model-releases) adds AOB-derived classifier training, released visual-model quantizations, FSIQ evaluation by the two Industrial-Instruction models, an interrupted GEPA notebook, and hydraulic/water-domain framework adaptations. These are explicitly different uses. In particular, quantization is not AOB supervised fine-tuning, and Industrial-Instruction's FSIQ test set is not its training corpus.

The HF link `amaan784/AssetOpsBench-VLM-22` returned unauthorized; no public dataset-release claim is added. The Team 21 checkpoint requires university login. Empty planning/combined training-data files in the Columbia QLoRA repository were recorded rather than promoted as dataset releases. No new independently verified fine-tuned model on PHMForge, DiagnosticIQ, IoTBench, or IndustryAssetEQA was established in this pass.

## Refresh procedure

Open each canonical publication and its preprint record; export or record the citing works; deduplicate by DOI, arXiv ID, and title. Check primary full text for the resource name, methods, tables, and bibliography. Record author overlap before calling use external. Count distinct works separately from distinct citing organizations. Keep coverage, citations, experiments, and extensions as separate measures. Record retrieval date and source whenever publishing a count.

## Final public-access and README review — November 10, 2026

The README review checked 112 distinct external content URLs before corrections (outside the generated contributor block), local links/anchors, the 15-entry paper table, dated metric arithmetic, and impact statements against their evidence. It replaced a missing GitHub Pages evaluation page with the repository documentation and a failing workshop-recording URL with IBM's publication record. The overview date and final HTML wrapper were corrected; the diagram and dated headline metrics were preserved.

An explicit repository-visibility check found that some authenticated discovery sources were not public. Private-only artifact entries, links, and the battery-scenario headline were removed from the current documentation and selected-source snapshots. Discovery totals describe search scope before this filter, not publicly verified adopters. An institution's publicly listed project connection may remain without a software-release claim. The earlier authenticated-access checks should not be read as public-release verification.

Some publishers returned 403 responses or transient server errors to automated requests. Those results were not classified as dead links or as newly verified full-text evidence; canonical references remain where independently identified. Downstream training/evaluation workloads were not rerun.

After corrections, 109 distinct external content URLs were rechecked: 101 returned HTTP 200, four returned HTTP 403, and four AAAI publisher URLs returned HTTP 520. No HTTP 404 remained in that pass. Access-limited URLs remain unverified for current reachability.

## Supplemental public-reference search — November 10

Six GitHub repository searches covered AssetOpsBench, FailureSensorIQ, PHMForge, DiagnosticIQ, IndustryAssetEQA, and the ReAct industrial-sensor benchmark title in names, descriptions, and READMEs. They reported 133 matches; the retrieved public result sets contained 132 hits representing 110 distinct repositories, including upstream projects, mirrors, related-work lists, and unrelated names—not 110 adopters. Thirteen selected public READMEs and sixteen targeted files were inspected, with immutable revisions recorded in the [reference snapshot](assets/reference-discovery-snapshot.json). Web searches also covered articles, directories, event archives, and company domains across the family.

The pass adds experimental/model-routing code, reported competition fine-tuning, a corrected PHMForge-derived provenance study, verification extensions, and reference-data integrations to [RESEARCH_USE.md](RESEARCH_USE.md#further-public-implementations--november-10-reference-pass). It also links the Damarla–Zhu implementation and ASTRIS training-data documentation without counting them as new studies or model releases.

Disposition matters: GOMAX’s related-work comparison is not dataset use; PHMForge-SME’s protocol adaptation does not establish reuse of the original raw datasets; similarly named medical DiagnosticIQ projects were excluded. The digital-twin repository’s corrected artifact supersedes its stale performance headline. Private repositories and reviewer-only weights are not public releases. Rate-limited or unavailable summary pages were not promoted as verified reuse. No additional company deployment was established by this pass. Search results are bounded and index-dependent, not a complete citation census.

## Platform follow-up — November 10

The live, signed-out Harbor registry verifies `phmforge/phmforge`, revision 3. Its 123 tasks adapt 99 scenarios and use a different execution contract from the paper. Kaggle’s public dataset API and version-4 archive resolve the DiagnosticIQ dataset-access lead. The public PHMForge Hugging Face mirror was checked separately; private GitHub repositories were not used as publication evidence. Additional checks distinguish Mercor occupational mapping and whatmodel’s reuse of published scores from fresh benchmark execution. See the [platform snapshot](assets/platform-access-snapshot.json) and [coverage details](COVERAGE_AND_EVENTS.md#harbor-and-kaggle-platform-check--november-10). These findings do not alter headline paper/download counts.

## Result-aggregation follow-up — November 10

A further web/public-code pass verified BenchmarkList’s DiagnosticIQ table and model JSON, AI Pulse’s paper-linked DiagnosticIQ coverage, and University of Zagreb Benchmark Explorer records. The Explorer’s PHMForge-named record actually points to PHMForge-SME; retain that lineage distinction. Previously cataloged ReasonLab/Simplicity Paradox and Mahal Systems were deduplicated. These additions are published-result reuse or discovery, not independent model runs. [Pinned records and dispositions](assets/result-discovery-snapshot.json).

## Deeper cross-family artifact search — November 10

A further pass queried public Hugging Face model and dataset metadata for all six family/framework names, then searched full dataset identifiers and broader names in GitHub code. Selected sources were checked for public visibility before inspection. Verified additions include a released smart-grid transformer scenario extension, IBM Evoflux’s connected AOB adapter, commercial model-catalog coverage, an AI Alliance guide, and explicit ReActXen directory profiles. See the [snapshot](assets/deeper-discovery-snapshot.json) and [research-use additions](RESEARCH_USE.md#additional-artifacts-from-the-deeper-cross-family-pass--november-10).

Limits and exclusions: the two additional AOB Hugging Face copies were not established as substantive extensions; the sparse PHMForge_Scenarios card does not establish independent reuse. DocuMind stores dataset metadata, not verified benchmark questions. IndustryAssetEQA mentions in graph-index entity records do not establish dataset execution. A ReActXen code copy does not establish a new experiment. MIT-LCP’s Croissant Baker inventory lists FailureSensorIQ among candidate NeurIPS datasets, but the inspected inventory alone does not establish a completed metadata-generation experiment. RuiSue719 app prompts/reference notes recommend FailureSensorIQ without a verified question loader. No new company deployment or public fine-tuned weights were verified. Discovery totals are not adopter counts.

## Provider and evaluation-artifact follow-up — November 10

Further web searches surfaced TokenRouter’s own AOB support announcement and upstream Stirrup/OpenCode integration documentation. Public HF APIs and pinned JSONL files verify A2B result distribution and I-ARIF-CANON’s held-out AOB records. This strengthens already known relationships rather than adding independent adopter counts. The held-out records are explicitly not evidence of AOB fine-tuning; no new trained model or company deployment was established. The exact-name ReActXen PyPI lead and the indexed Stirrup GitHub Pages URL returned 404. See [pinned artifacts, parsed row counts, and hashes](assets/provider-evaluation-snapshot.json).

## Further packaging and implementation search — November 10

A broader AOB code search and web queries across FailureSensorIQ, DiagnosticIQ, PHMForge, IndustryAssetEQA, and IoTBench/ReActXen found additional container packaging, a local smoke-test report, comparative discussion, and international coverage. Exgentic’s Dockerfile establishes pinned scenario import, but its grader returns an external-grading placeholder; the README alone would overstate completeness. The local Gemma report is one query, not a benchmark score, and the tesla_agent repository name does not establish Tesla affiliation.

A TORQ demo README claims AOB manual/history sourcing, but the inspected public tree contains only placeholders in those directories; dataset provenance and substantive reuse were not verified, so it is excluded from reuse highlights. IBM’s anomaly-detection code-pattern README links AOB as an open research benchmark; that cross-link is not an additional dataset experiment. Cymatix explicitly records a decision not to execute the proposed comparison. PHMForge-SME’s fine-tuning item remains a plan, not a released LoRA model. No new fine-tuned checkpoint or company deployment was established. [Snapshot and pinned sources](assets/container-discovery-snapshot.json).

## Official-fork follow-up — November 10

The first public-fork page (up to 100 entries) was checked for IBM/AssetOpsBench, IBM/FailureSensorIQ, and IBM/ReActXen. Six selected AOB forks were inspected at immutable revisions, including their README and tree. This is a bounded fork check, not a complete fork census; copied upstream content and upstream fine-tuning tests do not establish independent training.

The Edward-H26 hydraulic-rig extension supplies substantive additional code, a pinned 17-record AOB scenario subset, and IndustryAssetEQA facts-file loading. The pinned public source facts file was independently fetched and parsed (2,205 rows); wrappers were inspected. No model training or episode execution was performed. Results are not shipped, and pilot mapping assumptions remain explicitly unconfirmed. Other selected forks were not promoted merely for containing upstream files. No additional company deployment or fine-tuned checkpoint was verified by this pass. [Evidence snapshot](assets/hydraulic-fork-evidence.json).

## Direct multi-engine follow-up — November 10

Public search pages were inspected in **Google, Bing, Baidu, Sogou, and Perplexity**, in addition to the connected web-search tool and Chinese-language/domain queries. This records actual browser access, not an assumption that one search API independently represents those engines. Perplexity’s generated answer was used only to locate original sources: the inspected portion largely repeated predictive-maintenance, Industrial-Instruction, A2B, and the already listed Lion Latency project. The latter’s original fine-tuning check and profiling harness were inspected to strengthen its evidence; no new model-release count is added.

Sogou surfaced a WeChat article titled *不再只卷机器人：5个新开源项目，把AI铺进工厂的人机料法环*. Only the indexed snippet was visible: direct article reading was blocked by browser site-safety policy. It remains an unverified coverage lead, not dataset-use evidence. Baidu’s initial PHMForge results were readable; its FailureSensorIQ follow-up redirected to an authentication origin that automatic approval review rejected. That path was stopped, without sign-in or a workaround. Search-page advertisements, generated summaries, unrelated PHM/PhysForge results, and profile-side publication recommendations were not treated as company use.

Four previously unchecked industrial-agent repositories were also inspected. FabPilot cites AOB while using simulated MQTT telemetry; Argus uses UCI AI4I data; Maintenance Autonomy Agent positions itself relative to PHMForge; the multimodal inspection project explicitly says it does not copy AOB code. These establish discussion or design influence, not verified family-dataset reuse. [Source revisions, search queries, access limits, and dispositions](assets/multi-engine-search-audit.json). No additional company deployment or newly released fine-tuned weights were verified.

## Executable-reference and Brave follow-up — November 10

Six GitHub code searches restricted to Python covered AssetOpsBench, FailureSensorIQ, DiagnosticIQ, PHMForge, IndustryAssetEQA, and ReActXen. New candidates were checked for public repository visibility before inspection. A direct Brave Search query (`"FailureSensorIQ" training model`) supplied another engine’s results; its strongest sources largely repeated the known distillation, Industrial-Instruction, and Damarla–Zhu work, rather than establishing new fine-tuned releases.

The strongest artifact additions came from the original Industrial Mind & Code repository: two further FailureSensorIQ experiments with parsed public response files (702 and 500 records), provenance, and replay tooling. These strengthen a known researcher’s evidence rather than add company adopters. Other code hits were not promoted: FactReview’s mention is a search-result-disambiguation example; star-astrolab classifies benchmark repositories; justsearch explicitly marks its PHMForge comparison as not self-reproduced; P3-EVHM describes its own rule-based agents by analogy to PHMForge. A public multi-agent-discussion script comments on earlier FailureSensorIQ/MMLU-Pro runs, but the inspected release does not establish those pilot logs as available or as the current paper’s benchmark results. No new company deployment or released fine-tuned weights were verified. [Pinned sources, search scope, and response hashes](assets/response-artifact-audit.json).

## Family title and citation-edge follow-up — November 10

Discovery now explicitly follows **family names, earlier titles, stable paper IDs/DOIs, and citing-record edges**, rather than relying on “AOB” keyword matches. OpenAlex supplements the earlier Scholar audit; its counters are separate dated index observations and are not added to Scholar counts, headline papers, or adoption metrics.

| Family | Title/identity variants retained for discovery |
| :--- | :--- |
| **AssetOpsBench / AOB** | Current benchmark title; arXiv `2506.03828`; KDD DOI `10.1145/3770855.3817562`. OpenAlex also labels the preprint record “AssetOpsBench: A Real-World Evaluation Benchmark for AI-Driven Task Automation in Industrial Asset Management”; this is not treated as another publication. |
| **DiagnosticIQ** | Current symbolic-rule maintenance-action title; arXiv `2605.08614`; earlier *Augmenting Industrial Maintenance with LLMs: A Benchmark, Analysis, and Generalization Study* lead. A changed title or review submission is not a separate accepted paper. |
| **PHMForge** | Current *Evaluating LLM Agents on Industrial Prognostics through MCP-Native, Algorithm-Grounded Tools* and earlier *A Scenario-Driven Agentic Benchmark for Industrial Asset Lifecycle Maintenance*; arXiv `2604.01532`. Generic PHM research is not automatically this benchmark. |
| **FailureSensorIQ** | Full sensor-relationships/failure-modes title; arXiv `2506.03278`; related distillation and embedding titles remain distinct connected studies. |
| **IndustryAssetEQA** | Full neurosymbolic operational-intelligence title; arXiv `2604.23446`; ACL DOI `10.18653/v1/2026.acl-industry.49`. Similarly named IndustryEQA work is not assumed to be the same benchmark. |
| **IoTBench / ReActXen** | *ReAct Meets Industrial IoT: Language Agents for Data Access*; EMNLP DOI `10.18653/v1/2025.emnlp-industry.24`; separate tool/framework mentions are distinguished from dataset runs. |

New primary-text checks verify PHMForge’s discussion in Theiler/Telyatnikov/Von Krannichfeldt/Fink’s EPFL tabular-foundation-model paper (reference 18 and conclusion), AOB’s citation in Joo/Kim’s *Narrow and Deep* (introduction/reference 8), and AOB’s qualitative comparison in *CausalPulse* (related work/Table 4). These are **citation reach**, not new executions. CausalPulse’s Bosch deployment statement concerns its own system and does not establish Bosch deployment of AOB; author overlap with IndustryAssetEQA is retained.

The OpenAlex PHMForge and DiagnosticIQ citing lists and the remaining-family DOI lookups were inspected, including separate AOB preprint/proceedings identities. Repeated versions, connected papers, generic journal-container records, and metadata-only book/chapter leads were not promoted as new reuse. Some title-search requests returned 429/504; successful DOI/edge requests are recorded independently, without presenting failed lookups as zero citations. [Raw index metadata and checked-paper dispositions](assets/family-citation-edge-audit.json).

## Citation-context follow-through — November 10

Methods, datasets, and references were checked beyond the citation edge. *Capable but Unreliable* (`2602.19008v1`, Section 2) verifies another AOB methodology citation, while Section 3 identifies Toolathlon as its actual experimental data. Its tool-inventory characterization is not repeated as an AOB fact. This is citation reach, not dataset execution.

Two papers reached through the EPFL PHM reference trail—*From paper to benchmark* (`2605.28371v1`) and *Picid* (`2605.28345v1`)—were inspected; no `PHMForge` match was found in those HTML versions, so they are not counted as verified PHMForge citations or use. DiagnosticIQ search results continued to return its own records, summaries, or previously checked citing work. A new SSRN work-order scheduling lead (`10.2139/ssrn.7095162`) describes its own university-campus CMMS data in the public abstract; the indexed citation alone does not establish AOB data use, so experimental lineage remains unresolved.

The family-level research-use summary now consolidates already verified Harbor adaptation, DiagnosticIQ’s author release and published-result aggregation, and ReActXen discovery profiles. None of these is converted into a new company-use, model-weight, paper, or download count. [Checked primary contexts and dispositions](assets/citation-context-followthrough.json).
