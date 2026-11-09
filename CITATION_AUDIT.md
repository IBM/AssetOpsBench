# Citation discovery and verification

Reviewed November 9, 2026 using Google Scholar's “Cited by” lists and primary papers. This is a dated research-discovery audit, not a unique-citation total. Keyword search results are not citation counts. Scholar may split proceedings, preprints, and changed titles into different records, and may index blogs or patents alongside papers.

## Records inspected

| Resource / Scholar record | Displayed citing records | Discovery link |
| :--- | ---: | :--- |
| AssetOpsBench preprint | 31 | [Citing records](https://scholar.google.com/scholar?cites=3506473653926276354&hl=en) |
| AssetOpsBench KDD proceedings | 6 | [Citing records](https://scholar.google.com/scholar?cites=4319209530128476739&hl=en) |
| FailureSensorIQ | 17 | [Citing records](https://scholar.google.com/scholar?cites=16528683271765767035&hl=en) |
| PHMForge, current title | 1 | [Citing records](https://scholar.google.com/scholar?cites=14992453025155493477&hl=en) |
| PHMForge, earlier title | 3 | [Citing records](https://scholar.google.com/scholar?cites=9384586252328081340&hl=en) |

**Do not add these numbers.** Versions overlap, one citing work can reference several family members, and the PHMForge current-title record includes a project-connected mentorship blog. FailureSensorIQ's list also contains multiple versions of individual studies. No aggregate citation badge is proposed from these raw counts.

## Verified examples suitable for the README

The [research-use register](RESEARCH_USE.md) supplies people, author relationships, and evidence locations for:

- External experiments: **Can CaMeLs Talk?**, **Harnessing collective intelligence**, **Industrial-Instruction**, and **Retrieval-based and Fine-tuned LLM Approaches**.
- Connected experiments and extensions: **Agentics**, **DynAMO**, the knowledge-graph extension, the Columbia tool-knowledge project, and **Trajel**. Shared authorship is identified rather than counted as independent validation.
- External citations: **Confucius** and **CoffeeBench**. A citing author's employer is not evidence of company deployment.
- Company discussion: **LG AI Research**, **EQUA AI**, and **Lancartech**; external editorial review from **HysenLabs**. LG's stated evaluation interest remains prospective.

## Additional citing-work leads

The following appeared in Scholar's citation lists. Discovery does not by itself establish experimental use, independence, or endorsement. These remain follow-up leads unless the evidence register above explicitly verifies their relationship. Names below are bibliographic authors, not testimonials.

| Citing work | Family record where discovered | Primary record / follow-up |
| :--- | :--- | :--- |
| AI-Augmented SOC: A Survey of LLMs and Agents for Security Automation | AssetOpsBench | [Publisher](https://www.mdpi.com/2624-800X/5/4/95) |
| TraceBench — Tommaso Bendinelli, Artur Dox, Christian Holz | AssetOpsBench | [Preprint](https://arxiv.org/abs/2608.27182); abstract describes its own simulation benchmark |
| Capable but unreliable | AssetOpsBench | [Preprint](https://arxiv.org/abs/2602.19008) |
| IndustryBench | AssetOpsBench | [Preprint](https://arxiv.org/abs/2605.10267) |
| ADMITBench — Yash Misra, Javal Vyas, Siddharth Gutta, Mehmet Mercangöz | AssetOpsBench; PHMForge keyword discovery | [Preprint](https://arxiv.org/abs/2608.03866); abstract describes a separate advisory-evaluation framework |
| Evidence-Driven Reasoning for Industrial Maintenance Using Heterogeneous Data | AssetOpsBench | [ACL paper](https://aclanthology.org/2026.acl-industry.50/); check overlap with benchmark authors |
| Forecast Workflow Bench | AssetOpsBench | [Preprint](https://arxiv.org/abs/2609.27385) |
| Text-Informed Training Data Filtering for Enhanced Virtual Flow Meter Performance | AssetOpsBench | [Preprint](https://arxiv.org/abs/2610.19860) |
| FacProcessTwin | AssetOpsBench | [Preprint](https://arxiv.org/abs/2606.17666) |
| Imagine Before You Act | AssetOpsBench | [ACM record](https://doi.org/10.1145/3841560.3841567) |
| Hard-CAD | AssetOpsBench proceedings | [Publisher](https://www.sciencedirect.com/science/article/pii/S0097849326002979) |
| Large Language Models in Industry 5.0 — Multi-Case Experiences from the AIMS5.0 Project | AssetOpsBench proceedings | [Repository](https://real.mtak.hu/249438/); verify named companies and the actual AOB reference before claiming industry use |
| BTS-AgentBench — J. Y. Kim | AssetOpsBench proceedings | [Preprint](https://arxiv.org/abs/2608.27334) |
| Multi-Agent AI System for Preventive Maintenance Plan Optimization — R. K. Patel | AssetOpsBench | [TUM thesis](https://mediatum.ub.tum.de/1865598) |
| FORGE: Fine-grained multimodal evaluation for manufacturing quality assurance | FailureSensorIQ | Findings of ACL 2026; verify final proceedings record and citation context |
| Comprehensive benchmark of large language models for construction engineering and management | FailureSensorIQ | Automation in Construction, 2026; verify experiment scope |
| Simplicity Paradox | FailureSensorIQ | [Preprint](https://arxiv.org/abs/2607.14109); check author overlap |
| From Question Answering to Task Completion: A Survey on Agent System and Harness Design | PHMForge earlier title | [Preprint](https://arxiv.org/abs/2606.20683) |
| Prognostics and Health Management in the Era of Foundation Models — T. Han, X. Wang, H. Meng, E. Zio | PHMForge earlier title | [IEEE record](https://ieeexplore.ieee.org/document/11715714); citation discovery, not a verified PHMForge experiment |
| A Study on Locally Runnable Large Language Models for Bearing Fault Diagnosis — M. H. Shawon, P. Kumar | PHMForge earlier title | [Publisher](https://www.mdpi.com/2079-3197/14/9/220); inspect the reference and experiment separately |

This table omits obvious original-family papers and several low-confidence/non-paper results. It is not a complete bibliography. DiagnosticIQ, IoTBench/ReActXen, and IndustryAssetEQA are included in the six-family evidence register, but this audit does not report a complete Scholar citation count for those releases.

## Refresh procedure

Open each canonical publication and its preprint record; export or record the citing works; deduplicate by DOI, arXiv ID, and title. Check primary full text for the resource name, methods, tables, and bibliography. Record author overlap before calling use external. Count distinct works separately from distinct citing organizations. Keep coverage, citations, experiments, and extensions as separate measures. Record retrieval date and source whenever publishing a count.
