<div align="center">

# AssetOpsBench

### AI Agents for Industrial Asset Operations & Maintenance

*A unified, open framework for building, orchestrating, and evaluating domain-specific AI agents in Industry 4.0.*

[![Stars](https://img.shields.io/github/stars/IBM/AssetOpsBench?style=for-the-badge&logo=github&color=yellow)](https://github.com/IBM/AssetOpsBench/stargazers)
[![Forks](https://img.shields.io/github/forks/IBM/AssetOpsBench?style=for-the-badge&logo=github)](https://github.com/IBM/AssetOpsBench/network/members)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](LICENSE)
[![KDD 2026](https://img.shields.io/badge/KDD%202026-Accepted-0f62fe?style=for-the-badge)](#publications)
[![IJCAI 2026](https://img.shields.io/badge/IJCAI%202026-Live%20Competition-d02670?style=for-the-badge)](#ai-competitions)
[![CODS 2025](https://img.shields.io/badge/CODS%202025-500%2B%20Submissions-007d79?style=for-the-badge)](#ai-competitions)

[![AssetOps](https://img.shields.io/badge/Domain-Asset_Operations-blue)](#)
[![MultiAgentBench](https://img.shields.io/badge/Domain-Multi--agent_Bench-blue)](#)
[![EMNLP 2025](https://img.shields.io/badge/EMNLP--2025-Accepted-blueviolet)](#publications)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS--2025-Accepted-blueviolet)](#publications)
[![AAAI 2026](https://img.shields.io/badge/AAAI--2026-Accepted-blueviolet)](#publications)
[![IAAI 2026](https://img.shields.io/badge/IAAI--2026-Accepted-blueviolet)](#publications)
[![ICLR 2026](https://img.shields.io/badge/ICLR--2026-Accepted-blueviolet)](#publications)
[![ACL 2026](https://img.shields.io/badge/ACL--2026-Accepted-blueviolet)](#publications)



📄 [**Paper**](https://arxiv.org/pdf/2506.03828) · 🤗 [**Dataset**](https://huggingface.co/datasets/ibm-research/AssetOpsBench) · 🎮 [**Playground**](https://huggingface.co/spaces/ibm-research/AssetOps-Bench) · 📢 [**IBM Blog**](https://research.ibm.com/blog/asset-ops-benchmark) · 🎥 [**Video**](https://www.youtube.com/watch?v=kXmBDMrKFjs) · 📊 [**Kaggle**](https://www.kaggle.com/benchmarks/ibm-research/asset-ops-bench) · 🚀 [**Colab**](https://colab.research.google.com/github/IBM/AssetOpsBench/blob/main-0.x/notebook/LLM_Agent.ipynb)

</div>

> [!IMPORTANT]
> 🎉 **AssetOpsBench is officially accepted at KDD 2026** (Datasets & Benchmarks Track), Jeju, South Korea, alongside our hands-on tutorial *Building Reliable Industrial Agents with MCP*. See [Publications](#publications) for the full list of 2025–2026 work.

---

## At a Glance

<table>
  <tr>
    <td align="center" width="16%"><b>9</b><br><sub>Asset classes</sub></td>
    <td align="center" width="16%"><b>141+</b><br><sub>Scenarios</sub></td>
    <td align="center" width="16%"><b>5</b><br><sub>Domain agents</sub></td>
    <td align="center" width="16%"><b>2</b><br><sub>Orchestration frameworks</sub></td>
    <td align="center" width="16%"><b>20+</b><br><sub>University extensions</sub></td>
    <td align="center" width="16%"><b>500+</b><br><sub>Competition submissions</sub></td>
  </tr>
</table>

**Built for:** maintenance engineers, reliability specialists, facility planners, and Industry 4.0 researchers.
**Powered by:** LLMs + Time Series Foundation Models, orchestrated over live sensor data and Industry 4.0 records (FMEA, work orders, alerts).
**Now with:** simplified interface and native **MCP (Model Context Protocol)** support.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/IBM/AssetOpsBench.git
cd AssetOpsBench
pip install -e .

# Try a scenario (to be enabled)
python -m assetopsbench.run --scenario "List all sensors of Chiller 6 in MAIN site"
```

Or jump in instantly:
- 🚀 **[Run on Colab](https://colab.research.google.com/github/IBM/AssetOpsBench/blob/main-0.x/notebook/LLM_Agent.ipynb)** — no install required (illustration of LLM Agent)
- 🎮 **[Try the HF Playground](https://huggingface.co/spaces/ibm-research/AssetOps-Bench)** — interactive demo
- 📖 **[Read INSTRUCTIONS.md](./INSTRUCTIONS.md)** — full setup, MCP servers, plan-execute runner

> [!NOTE]
> Active development is on `main`. The codebase used for various publication venues continues to be maintained on separate branches, for example, ACL 2026 [`IndustryAssetEQA`](https://github.com/IBM/AssetOpsBench/tree/IndustryAssetEQA) and prior experimental work is maintained on [`main-0.x`](https://github.com/IBM/AssetOpsBench/tree/main-0.x).

---

## What is AssetOpsBench?

AssetOpsBench is a **unified framework for developing, orchestrating, and evaluating domain-specific AI agents** in industrial asset operations and maintenance. It provides reproducible scenarios, agent tooling, and evaluation pipelines for multi-step workflows in simulated industrial environments.

### Domain-Specific MCP Servers

| MCP Servers | Important tools |
|---|---|
| **IoT** | `get_sites`, `get_history`, `get_assets`, `get_sensors` |
| **FMSR** | `get_sensors`, `get_failure_modes`, `get_failure_sensor_mapping` |
| **TSFM** | `forecasting`, `timeseries_anomaly_detection` |
| **WO** | `get_work_order_distribution`, `predict_next_work_order`, ... |
| **Vibration** | `compute_fft_spectrum`, `compute_envelope_spectrum`, ... |
| **...** | **...** |

### Agent Frameworks

- **[Plan Execute](./src/agent/plan_execute)** — plan-and-execute sequential workflow to work with any LLM
- **[Deep Agent](./src/agent/deep_agent)** — planning, sub-agents, and virtual filesystem for long-horizon tasks
- **[Claude Agent](./src/agent/claude_agent)** — ReAct-based orchestrator using Claude with agent-as-tool delegation
- **[OpenAI Agent](./src/agent/openai_agent)** — ReAct-based orchestrator using OpenAI models with agent-as-tool delegation


### MCP Environment

The `src/` directory contains MCP servers and a plan-execute runner built on the [Model Context Protocol](https://modelcontextprotocol.io/). See **[INSTRUCTIONS.md](./INSTRUCTIONS.md)** for setup.


---

## Example Scenarios

| Domain | Example Task |
|---|---|
| **IoT** | "List all sensors of Chiller 6 in MAIN site" |
| **FMSR** | "Identify failure modes detected by Chiller 6 Supply Temperature" |
| **TSFM** | "Forecast Chiller 9 Condenser Water Flow for the week of 2020-04-27" |
| **WO** | "Generate a work order for Chiller 6 anomaly detection" |

Some tasks focus on a single domain, others are multi-step end-to-end workflows. Explore all scenarios on [Hugging Face](https://huggingface.co/datasets/ibm-research/AssetOpsBench).

---

## Leaderboards

- To be revised (WIP with latest models)
- Evaluated with **7 Large Language Models**
- Trajectories scored using **LLM Judge (Llama-4-Maverick-17B)**
- **6-dimensional criteria** measuring reasoning, execution, and data handling

Example: MetaAgent leaderboard

![meta_agent_leaderboard](https://github.com/user-attachments/assets/615059be-e296-40d3-90ec-97ee6cb00412)

---

## Publications

**12+ contributions across 7 top venues** in 2025–2026 from the team behind AssetOpsBench.

<details>
<summary><b>⭐ KDD 2026 — Jeju, South Korea</b> (click to expand)</summary>

- **[D&B]** **AssetOpsBench: A Benchmark for Industrial Asset Operations Agents** · *D. Patel, S. Lin, et al.* · [📄 Paper](https://arxiv.org/pdf/2506.03828)
- **[Tutorial]** **Building Reliable Industrial Agents with MCP: A Hands-on AssetOpsBench Tutorial for AI-Driven Operations** · *D. Patel, C. Shyalika, et al.*

</details>

<details>
<summary><b>ACL 2026 - San Diego, USA</b></summary>

- **[Industry]** **IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance** · *C. Shyalika, D. Patel, A. Sheth*

</details>

<details>
<summary><b>ICLR 2026 - Brazil</b></summary>

- **[Main]** **Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring** · *N. Martinez, F. O'Donncha, W. M. Gifford, N. Zhou, D. C. Patel, R. Vaculin*

</details>

<details>
<summary><b>AAAI 2026 — Singapore</b></summary>

- **[Demo]** **AssetOpsBench-Live: Privacy-Aware Online Evaluation of Multi-Agent Performance in Industrial Operations** · *D. Patel, N. Zhou, S. Lin, J. T. Rayfield, C. Shyalika, S. R. Yarrabothula* · [🎥 Demo](https://www.youtube.com/watch?v=JcKlS5v5fGY)
- **[Main]** **SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search** · *Y. Zhang, G. Ganapavarapu, S. Jayaraman, B. Agrawal, D. Patel, A. Fokoue* · [💻 Code](https://github.com/IBM/SPIRAL)
- **[Bridge]** **Knowledge-Guided AI for Industrial Asset Health Monitoring** · *S. Lin, D. Patel*
- **[Tutorial]** **From Inception to Productization: Hands-on Lab for the Lifecycle of Multimodal Agentic AI in Industry 4.0** · *C. Shyalika, S. Ahuja, S. Lin, R. Wickramarachchi, D. Patel, A. Sheth* · [🌐 Website](https://ibm.github.io/AssetOpsBench/aaaiwebsite/) · [📊 Slides](https://drive.google.com/file/d/16GaYxBQ2FsVqKpkKOU0PI_ZCTCsowenF/view?usp=sharing)
- **[Workshop(AABA4ET)]** **Agentic Code Generation for Heuristic Rules in Equipment Monitoring** · *F. Lorenzi, A. Langbridge, F. O'Donncha, J. Rayfield, B. Eck, S. Rosato*

</details>

<details>
<summary><b>IAAI 2026 - Singapore</b></summary>

- **[Deployed]** **Deployed AI Agents for Industrial Asset Management: CodeReAct Framework for Event Analysis and Work Order Automation** · *N. Zhou, D. Patel, A. Bhattacharyya*
- **[Emmerging]** **Diversity Meets Relevancy: Multi-Agent Knowledge Probing for Industry 4.0 Applications** · *C. Constantinides, D. Patel, S. Kimbleton, N. Garg, M. Paracha*

</details>

<details>
<summary><b>NeurIPS 2025 — San Diego, USA</b></summary>

- **[D&B Track]** **FailureSensorIQ: A Multi-Choice QA Dataset for Understanding Sensor Relationships and Failure Modes** · *C. Constantinides, D. Patel, S. Lin, C. Guerrero, S. D. Patil, J. Kalagnanam* · [📄 arXiv](https://arxiv.org/abs/2506.03278) · [💻 Code](https://github.com/IBM/FailureSensorIQ)
- **[Social]** **Building Reliable Agentic Benchmarks: Insights from AssetOpsBench** *(invited talk, 2000+ registered)* · *D. Patel* · [📅 Luma](https://luma.com/mkyyvypm?tk=AkGVp5)

</details>

<details>
<summary><b>EMNLP 2025 — Suzhou, China</b></summary>

- **[Main]** **ReAct Meets Industrial IoT: Language Agents for Data Access** · *J. T. Rayfield, S. Lin, N. Zhou, D. C. Patel*
- **[Main]** **Generalized Embedding Models for Industry 4.0 Applications** · *C. Constantinides, S. Lin, D. C. Patel* · [📄 arXiv](https://arxiv.org/abs/2506.12607)
- **[Findings]** **Fine-Tuned Thoughts: Leveraging Chain-of-Thought Reasoning for Industrial Asset Health Monitoring** · *S. Lin, D. Patel, C. Constantinides* · [📄 ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1126/) · [💻 Code](https://github.com/IBM/FailureSensorIQ)

</details>

---

## Tutorials & Technical Material

📘 Hands-on guides from our team:

- [**ReActXen IoT Agent**](https://github.com/IBM/ReActXen/blob/main/docs/tutorial/ReActXen_IoT_Agent_EMNLP_2025.pdf) (EMNLP 2025)
- [**FailureSensorIQ**](https://github.com/IBM/FailureSensorIQ) (NeurIPS 2025)
- [**AssetOpsBench Lab**](https://ibm.github.io/AssetOpsBench/aaaiwebsite/) (AAAI 2026)
- [**SPIRAL**](https://github.com/IBM/SPIRAL) (AAAI 2026)
- [**AssetOpsBench Technical Material**](https://github.com/IBM/AssetOpsBench/blob/main/docs/tutorial/AssetOpsBench_Technical_Material.pdf)

---

## AI Competitions

AssetOpsBench powers public AI agent competitions that bring together researchers, students, and practitioners worldwide.

### 🔴 Live — IJCAI 2026

**Industrial Automation Challenge: Benchmarking Physics-Grounded LLMs for Task Reasoning**

A new challenge co-located with **IJCAI 2026** that pushes LLM agents on physics-grounded industrial reasoning.

- 🌐 **Challenge site:** [ai-industrial-challenge-ijcai](https://sites.google.com/view/ai-industrial-challenge-ijcai/home)
- 📋 **IJCAI 2026 competitions:** [2026.ijcai.org/competitions](https://2026.ijcai.org/competitions/)

### ✅ Completed — CODS 2025

**AssetOpsBench-Live: AI Agentic Challenge**

Launched in September 2025 at CODS 2025, the competition evaluated multi-agent systems on live industrial scenarios.

- 🏆 **Competition page:** [codabench.org/competitions/10206](https://www.codabench.org/competitions/10206/)
- 👥 **365 participants** · **500+ agent submissions**

---

## Talks & Events

| Date | Event |
|---|---|
| 2026-08 | **KDD 2026** — AssetOpsBench paper + MCP tutorial · *Jeju, South Korea* |
| 2026-05-10 | [NUS Seminar: AssetOpsBench Applications](https://events.comp.nus.edu.sg/view/25423) |
| 2025-12 | NeurIPS 2025 Social: *Building Reliable Agentic Benchmarks* (2000+ registered) |
| 2025-10-03 | 2-Hour Workshop: *AI Agents and Their Role in Industry 4.0 Applications* · NJIT ACM |
| 2025-09-01 | [CODS 2025 Competition Launch — AssetOpsBench-Live](https://www.codabench.org/competitions/10206/) |
| 2025-06-01 | AssetOpsBench v1.0 released — 141 industrial scenarios |

---

## University Projects & Extensions

AssetOpsBench is being extended by university research groups exploring new asset classes, evaluation paradigms, and agentic architectures. **To list your project, open a PR.**

- **Internalizing MCP Tool Knowledge in Small LLMs via QLoRA Fine-Tuning** — HPML project using AssetOpsBench to fine-tune ~4B models to internalize MCP tool knowledge and reduce prompt schema overhead. [Ayal Yakobe](https://github.com/yakobeayal), Columbia University · [repo](https://github.com/YuvalShemla/hpml-2026-project)
- **SPIN — Structural LLM Planning via Iterative Navigation for Industrial Tasks.** [Yusuke Ozaki](https://github.com/ozatamago), University at Albany · [paper](https://arxiv.org/abs/2605.14051) · [repo](https://github.com/ozatamago/AssetOpsBench/tree/UACap10)
- **Synthetic Scenario Generation for Evaluation of Industry 4.0 Agents** — Automated scenario generation, transformer asset integration, and scenario quality evaluation. [Rohith Kanathur](https://github.com/Rohith-Kanathur), [Sagar Chethan Kumar](https://github.com/Sagar-CK), Columbia University · [repo](https://github.com/Rohith-Kanathur/AssetOpsBench)
- **AgentOpsBench** — High-throughput battery analytics MCP server with DNN prognostics (RUL prediction) and 3.3× latency optimization. [Siddharth Gowda, Rushin Bhatt, Aryaman Agrawal, Winston Li](https://github.com/siddharthgowda), Columbia University · [repo](https://github.com/siddharthgowda/AssetOpsBench)
- **Skill-Knowledge-Augmented Agents on AssetOpsBench** — Confidence-gated skill execution with scoped knowledge plugins for industrial fault diagnosis. [Vera Mazeeva](https://github.com/verammaz), [Sanskruti Shejwal](https://github.com/Sans-Shej), [Shrey Arora](https://github.com/shreyarora2198), [Mana Abbaszadeh](https://github.com/Manazd), Columbia University · [repo](https://github.com/shreyarora2198/AssetOpsBench/tree/team14-final)
- **Evaluating Temporal Semantic Caching and Workflow Optimization in Agentic Plan-Execute Pipelines.** [Krish Veera](https://www.github.com/krishrveera/), [Alimurtaza Mustafa Merchant](https://github.com/alimurtaza0411/), [Sajal Kumar Goyla](https://github.com/SajalGoyla/), [Shambhawi Bhure](https://github.com/ShambhawiBhure/), Columbia University · [repo](https://github.com/alimurtaza0411/Latency-Optimized-AssetOpsBench/tree/feature/ablation-study)
- **Towards Multi-Turn Dialog Systems for Industrial Asset Operations and Maintenance** - Improved response quality and reduced redundant tool calls and multi-turn latency. [Chengrui Li](https://github.com/Coderlicr), [Rujing Li](https://github.com/rujing-li), [Yitong Bai](https://github.com/BrianBai093), [Rui Li](https://github.com/Rui2026), Columbia University · [repo](https://github.com/Coderlicr/Multi-Turn-AssetOps)
- **Skills and Knowledge Plugin MCP Servers for Optimized Industrial O&M Agents** - reducing planning overhead and improving retrieval grounding in industrial asset maintenance agents through an MCP Skills Server that exposes reusable multi-step operational workflows and a Knowledge Plugin Server that enables injection of context-specific documentation. [Andrew Li](https://github.com/andli28), [Kirthana Natarajan](https://github.com/kmn01), [Thai On](https://github.com/tq-o), [Trisha Maturi](https://github.com/trishamaturi), [Yeshitha Bhuvanesh](https://github.com/Yeshitha-co), Columbia University · [repo](https://github.com/kmn01/AssetOpsBench/)

---

## Call for Scenario Contribution

We are expanding **AssetOpsBench** to cover a broader range of industrial challenges. We invite researchers and practitioners to contribute new scenarios, particularly in:

- **Asset Classes:** Turbines, HVAC systems, Pumps, Transformers, CNC Machines, Robotics, Engines
- **Task Domains:** Prognostics and Health Management, Remaining Useful Life (RUL) estimation, Root Cause Analysis (RCA), Diagnostic Analysis, Predictive Maintenance

**How to contribute:**

1. **Define** your scenario following our [Utterance Guideline](docs/guideline/utterance_design_guideline.md) and [Ground Truth Guideline](docs/guideline/ground_truth_design_guideline.md)
2. **Explore** the [Hugging Face dataset](https://huggingface.co/datasets/ibm-research/AssetOpsBench) for examples
3. **Submit** a Pull Request or open an [Issue](https://github.com/IBM/AssetOpsBench/issues) with the tag `new-scenario`
4. **Contact us** with questions:
   - Dhaval Patel — [pateldha@us.ibm.com](mailto:pateldha@us.ibm.com)
   - Nianjun Zhou — [jzhou@us.ibm.com](mailto:jzhou@us.ibm.com)

---

## Contributors

Thanks to these wonderful people ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/ShuxinLin">
          <img src="https://github.com/ShuxinLin.png?s=50" width="50px;" alt="ShuxinLin"/><br />
          <sub><b>ShuxinLin</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=ShuxinLin" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/DhavalRepo18">
          <img src="https://github.com/DhavalRepo18.png?s=50" width="50px;" alt="DhavalRepo18"/><br />
          <sub><b>DhavalRepo18</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=DhavalRepo18" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/ChathurangiShyalika">
          <img src="https://github.com/ChathurangiShyalika.png?s=50" width="50px;" alt="ChathurangiShyalika"/><br />
          <sub><b>ChathurangiShyalika</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=ChathurangiShyalika" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/Dev-Scodes5">
          <img src="https://github.com/Dev-Scodes5.png?s=50" width="50px;" alt="Dev-Scodes5"/><br />
          <sub><b>Dev-Scodes5</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=Dev-Scodes5" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/DeveloperMindset123">
          <img src="https://github.com/DeveloperMindset123.png?s=50" width="50px;" alt="DeveloperMindset123"/><br />
          <sub><b>DeveloperMindset123</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=DeveloperMindset123" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/LGDiMaggio">
          <img src="https://github.com/LGDiMaggio.png?s=50" width="50px;" alt="LGDiMaggio"/><br />
          <sub><b>LGDiMaggio</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=LGDiMaggio" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/PUSHPAK-JAISWAL">
          <img src="https://github.com/PUSHPAK-JAISWAL.png?s=50" width="50px;" alt="PUSHPAK-JAISWAL"/><br />
          <sub><b>PUSHPAK-JAISWAL</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=PUSHPAK-JAISWAL" title="Code">💻</a>
      </td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/bradleyjeck">
          <img src="https://github.com/bradleyjeck.png?s=50" width="50px;" alt="bradleyjeck"/><br />
          <sub><b>bradleyjeck</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=bradleyjeck" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/florenzi002">
          <img src="https://github.com/florenzi002.png?s=50" width="50px;" alt="florenzi002"/><br />
          <sub><b>florenzi002</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=florenzi002" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/jack-pfeifer">
          <img src="https://github.com/jack-pfeifer.png?s=50" width="50px;" alt="jack-pfeifer"/><br />
          <sub><b>jack-pfeifer</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=jack-pfeifer" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/jdsheehan">
          <img src="https://github.com/jdsheehan.png?s=50" width="50px;" alt="jdsheehan"/><br />
          <sub><b>jdsheehan</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=jdsheehan" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/jtrayfield">
          <img src="https://github.com/jtrayfield.png?s=50" width="50px;" alt="jtrayfield"/><br />
          <sub><b>jtrayfield</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=jtrayfield" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/kushwaha001">
          <img src="https://github.com/kushwaha001.png?s=50" width="50px;" alt="kushwaha001"/><br />
          <sub><b>kushwaha001</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=kushwaha001" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/nianjunz">
          <img src="https://github.com/nianjunz.png?s=50" width="50px;" alt="nianjunz"/><br />
          <sub><b>nianjunz</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=nianjunz" title="Code">💻</a>
      </td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/sandeepkunkunuru">
          <img src="https://github.com/sandeepkunkunuru.png?s=50" width="50px;" alt="sandeepkunkunuru"/><br />
          <sub><b>sandeepkunkunuru</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=sandeepkunkunuru" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/srutanik">
          <img src="https://github.com/srutanik.png?s=50" width="50px;" alt="srutanik"/><br />
          <sub><b>srutanik</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=srutanik" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/thedgarg31">
          <img src="https://github.com/thedgarg31.png?s=50" width="50px;" alt="thedgarg31"/><br />
          <sub><b>thedgarg31</b></sub>
        </a><br />
        <a href="https://github.com/IBM/AssetOpsBench/commits?author=thedgarg31" title="Code">💻</a>
      </td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=IBM/AssetOpsBench&type=Date)](https://star-history.com/#IBM/AssetOpsBench&Date)

---

<div align="center">

**If AssetOpsBench is useful to your work, please ⭐ star the repo, 🍴 fork it, and tell us what you're building.**

</div>
