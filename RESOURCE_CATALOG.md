# Keeping the AssetOpsBench ecosystem complete

The README is the public front door. Maintain a resource catalog behind it so a new dataset, seminar, project, or pipeline can be added consistently.

## Give every resource a home

Use one primary type per resource:

| Type | Includes |
| :--- | :--- |
| `dataset` | Dataset releases, benchmark scenarios, evaluation sets |
| `tool` | MCP servers, libraries, connectors, reusable utilities |
| `pipeline` | Orchestration recipes, execution workflows, evaluation harnesses |
| `paper` | Conference and journal papers, preprints, workshop papers |
| `learning` | Tutorials, notebooks, courses, guides, recorded labs |
| `event` | Seminars, conference sessions, workshops, BoFs, meetups |
| `competition` | Challenge editions, individual tracks, results, retrospectives |
| `leaderboard` | Versioned rankings with dataset, scoring protocol, and update date |
| `evaluation` | Measurement protocols, audit frameworks, evaluation-science programs |
| `coverage` | News, newsletters, external reviews, social posts, and directory mentions |
| `project` | University projects, collaborations, community extensions |
| `application` | Demos, playgrounds, deployed systems |

Add tags for secondary roles. A benchmark dataset remains a `dataset`; tags can indicate `benchmark`, `time-series`, or `failure-reasoning`.

For coverage, record `publisher`, `author`, `published_on`, `relationship` (team, collaborator, external, or unknown), `format`, `original_source`, `verification_status`, and `correction_notes`. A syndicated copy links back to the original and does not count as a new original article. Store positive, neutral, and critical commentary with the same attribution standards.

The initial [coverage and events inventory](COVERAGE_AND_EVENTS.md) supplies researched entries for these types.

## Record the facts once

Suggested record format for a future machine-readable catalog:

```yaml
id: assetopsbench-kdd-tutorial-2026
type: learning
title: "Building Reliable Industrial Agents with MCP: A Hands-on AssetOpsBench Tutorial for AI-Driven Operations"
summary: "A guided industrial agent lab covering MCP tools, execution, and evaluation."
ownership: team # team | collaboration | community
status: released # draft | announced | released | archived
date: "2026-08-11"
version: null
venue: KDD 2026
location: "Jeju, South Korea"
url: https://shuxinlin.github.io/assetopsbench-kdd-tutorial-2026/
links:
  official_program: https://kdd2026.kdd.org/tutorials/
related:
  - assetopsbench-code
  - assetopsbench-paper
tags: [MCP, industrial-agents, hands-on]
verified_on: "2026-11-09"
```

For papers, add authors, publication status, track, and DOI. For datasets, add configuration, version, license, and access conditions. For tools and pipelines, add repository path, release, and setup instructions. For events, add start/end dates, timezone when relevant, organizer, registration, recording, and event status. For projects, add contributors, affiliation, ownership, and linked outputs.

Use `null` for unknown facts. Keep publication date, event date, and release date separate when they differ.

## Link related outputs

A research effort may produce a paper, dataset, tool, tutorial, and conference session. Give each independently accessible output its own record and connect their IDs through `related`. This preserves every contribution while allowing visitors to navigate the whole effort.

Track event status separately as `announced`, `scheduled`, `completed`, or `cancelled`; a published event page alone does not establish that an event took place.

## Publish different views from the same catalog

- **README:** selected highlights and entry points into every resource type.
- **Full directory:** all resources, grouped by type or filtered by topic.
- **Research bibliography:** papers with venue, track, status, and citation links.
- **Learning path:** tutorials and tools arranged for newcomers.
- **Timeline:** releases, seminars, workshops, and competitions by date.
- **Project directory:** community work with clear attribution.

Update the catalog when a resource is released, accepted, moved, or retired. Verify links and statuses before publishing a new overview. Counts should come from catalog records and distinguish papers, tutorials, datasets, projects, and events.

## Review notes from the initial public-source audit

- NUS's official listing gives **January 20, 2026**; the repository table listed May 10. The README follows the [university listing](https://events.comp.nus.edu.sg/view/25423).
- The benchmark paper describes **141 scenarios**; the current Hugging Face scenario viewer showed **152 rows**. These are version-specific counts, not interchangeable totals. See the [paper](https://arxiv.org/abs/2506.03828) and [dataset](https://huggingface.co/datasets/ibm-research/AssetOpsBench).
- The repository labels the IJCAI competition live, while [Track 1](https://www.kaggle.com/competitions/industrial-automation-challenge-track-1/) lists an August 18, 2026 close. Track statuses should be maintained individually.
- The repository's AAAI tutorial website link returned 404 during review. The README instead links to the [IBM Research tutorial record](https://research.ibm.com/publications/from-inception-to-productization-hands-on-lab-for-the-lifecycle-of-multimodal-agentic-ai-in-industry-40).
- Public sources cannot establish a complete inventory of unlisted team activities. Add internal or missing records before describing the directory as exhaustive.

## Companion benchmark versioning

DiagnosticIQ and PHMForge have dedicated entries in the overview. Link their papers, datasets, and comparisons through stable IDs. PHMForge’s current dataset card documents 99 scenarios, overlap with AssetOpsBench, and known issues; its older 75-scenario description is a different release. Do not count overlapping scenarios twice. DiagnosticIQ’s [public Kaggle dataset](https://www.kaggle.com/datasets/devindesilva1710/diagnosticiq-paper-rules) is now verified (version 4); its package contains question JSONL and asset descriptions, not a standalone model or full harness release. PHMForge also has a [Harbor package](https://hub.harborframework.com/datasets/phmforge/phmforge) with 123 tasks derived from its 99 scenarios; the adapted evaluation is separate from the MCP-paper protocol.

## Research-use and citation records

Track impact for each benchmark family, using the [research-use register](RESEARCH_USE.md). Record `resource_id`, `citing_work`, `authors`, `author_overlap`, `study_version`, `evidence_location`, and `relationship`: `evaluated_on`, `extended`, `cited`, `discussed`, or `prospective_use`. An external article about AOB does not establish experimental reuse of FailureSensorIQ or another family member. Include a bibliographic index and retrieval date before publishing any citation count.

For downstream reuse, record the exact training/evaluation split, whether weights or outputs are released, any fallback or synthetic data path, modified scoring, artifact revision, and whether an actual run is evidenced. A dataset loader, a model card, a saved experiment, and a citation support different claims.

### Additional released scenario extension

The [smart-grid transformer dataset](https://huggingface.co/datasets/rohithkanathur/assetopsbench-transformer-scenarios-dataset/blob/e3a0664ca455eb58be5fc5f3298bb06691cbb9bf/README.md) publishes 50 evaluation scenarios from a Columbia HPML project, with a Parquet data file and structured grading-rubric field. It extends the AOB task format; it does not increase the canonical scenario count. [Code-level integrations and limits](RESEARCH_USE.md#additional-artifacts-from-the-deeper-cross-family-pass--november-10).

### Further execution and evaluation artifacts

[Stirrup](https://github.com/IBM/AssetOpsBench/blob/e844f6468c6c68b328022e7ca8e31da140842cc7/docs/stirrup-agent.md) and [OpenCode](https://github.com/IBM/AssetOpsBench/blob/e844f6468c6c68b328022e7ca8e31da140842cc7/docs/opencode-agent.md) have public AOB runner documentation, including provider routing and code-versus-tools execution scope. A2B additionally releases [evaluation-result JSONL](https://huggingface.co/datasets/ariffazil/a2b-eval-results/blob/c0888fb89a986b007fdef00a43512f5b678273dc/README.md); I-ARIF-CANON keeps [AOB-derived records in a separate held-out file](https://huggingface.co/datasets/ariffazil/I-ARIF-CANON/blob/238a4cea2fda40a15f7cee9abddf0fe5e96c77cc/i-arif-a2b-heldout.jsonl). These artifacts do not increase the core benchmark or paper counts.

### Hydraulic-rig environment extension

The [Edward-H26 fork](https://github.com/Edward-H26/AssetOpsBench/blob/106ee90e3b3e13c69a280a50221b3b7a600ef7e3/docs/hydraulic_rig/README.md) implements a timed hydraulic-rig environment, action/diagnosis MCP tools, and Gymnasium/OpenEnv interfaces. It reuses the IndustryAssetEQA facts-file format and a pinned 17-scenario AOB subset. [Evidence and release limitations](RESEARCH_USE.md#hydraulic-rig-adaptation--november-10); no additional trained model or canonical benchmark score is claimed.
