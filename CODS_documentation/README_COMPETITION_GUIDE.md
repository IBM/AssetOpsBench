# 📚 AssetOpsBench CODS Competition - Complete Documentation Index

## 🎯 Your Challenge

Build autonomous AI agents that solve industrial maintenance tasks through multi-agent reasoning. You have **4 specific editable sections** to optimize.

---

## 📖 Documentation Files Created

### 1. **QUICK_REFERENCE.md** ⚡ START HERE
**Best for:** Getting started quickly, understanding what to edit
- TL;DR of 4 editable sections
- Impact vs Difficulty matrix
- Testing instructions
- Common mistakes
- Pro tips

**Read time:** 10 minutes
**Path:** `/Users/srutanik/AssetOpsBench_CODS/QUICK_REFERENCE.md`

---

### 2. **COMPETITION_EDITABLE_SECTIONS.md** 📋 MAIN GUIDE
**Best for:** Deep understanding of each editable section
- Complete context and purpose for each edit
- Current code with inline explanations
- What you can/cannot change
- Example enhancements
- File connections
- Constraints and rules
- Scoring explanation
- Submission checklist

**Read time:** 30 minutes
**Path:** `/Users/srutanik/AssetOpsBench_CODS/COMPETITION_EDITABLE_SECTIONS.md`

---

### 3. **CONNECTION_MAP.md** 🔗 TECHNICAL REFERENCE
**Best for:** Understanding data flow and architecture
- Complete execution flow diagrams (both tracks)
- How edits propagate through system
- Key data structures
- Immutable points (what not to touch)
- Data flow visualization

**Read time:** 20 minutes
**Path:** `/Users/srutanik/AssetOpsBench_CODS/CONNECTION_MAP.md`

---

## 🎓 Recommended Reading Order

### For Quick Start (15 min):
1. This file (INDEX)
2. QUICK_REFERENCE.md
3. Open IDE, find editable sections

### For Deep Understanding (1-2 hours):
1. This file
2. QUICK_REFERENCE.md
3. COMPETITION_EDITABLE_SECTIONS.md
4. CONNECTION_MAP.md
5. Study the actual code files
6. Review agent few-shots examples

### For Implementation (Daily):
1. Open QUICK_REFERENCE.md (as you edit)
2. Reference COMPETITION_EDITABLE_SECTIONS.md for specific guidance
3. Use CONNECTION_MAP.md to understand impact
4. Test frequently

---

## 🎯 The 4 Editable Sections At A Glance

| # | Track | Section | File | Lines | What | Impact | Difficulty |
|---|-------|---------|------|-------|------|--------|------------|
| 1 | Track 1 | Agent Info Format | `track1_planning.py` | 65-89 | How agents are presented to LLM | Medium | Easy |
| 2 | Track 1 | Planning Prompt | `track1_planning.py` | 167-198 | System prompt for plan generation | HIGH ⭐ | Medium |
| 3 | Track 2 | Task Helper | `track2_execution.py` | 31-81 | Implement task validation/enrichment | Medium | Medium |
| 4 | Track 2 | Execution Logic | `track2_execution.py` | 141-180 | Customize execution with fallbacks | HIGH ⭐ | Hard |

---

## 📂 Key File Locations

### Editable Files:
```
src/agent_hive/workflows/
├── track1_planning.py        ← Edit 1 & 2 (TRACK 1)
└── track2_execution.py       ← Edit 3 & 4 (TRACK 2)
```

### Entry Points:
```
benchmark/
├── cods_track1/run_track_1.py    ← Run your Track 1 code
└── cods_track2/run_track_2.py    ← Run your Track 2 code
```

### Reference Files (Read-Only):
```
src/agent_hive/
├── agents/
│   ├── base_agent.py          ✗ Don't edit
│   └── react_reflect_agent.py ✗ Don't edit
├── tools/
│   ├── fmsr.py                ✗ Don't edit
│   ├── skyspark.py            ✗ Don't edit
│   ├── tsfm.py                ✗ Don't edit
│   └── wo.py                  ✗ Don't edit
└── workflows/
    ├── base_workflow.py       ✗ Don't edit
    └── sequential.py          ✗ Don't edit

src/meta_agent/agents/
├── IoT/IoTAgentFewShots.py          👀 Study these
├── FMSR/FMSRAgentFewShots.py        👀 Study these
├── TSFM/TSFMAgentFewShots.py        👀 Study these
└── WorkOrder/WorkOrderFewShots.py   👀 Study these
```

---

## 🚀 Quick Start Steps

### Step 1: Understand Your Challenge (5 min)
- Read QUICK_REFERENCE.md
- Identify the 4 editable sections

### Step 2: Set Up Environment (10 min)
```bash
# Navigate to project
cd /Users/srutanik/AssetOpsBench_CODS

# Run baseline test
cd benchmark/cods_track1
python run_track_1.py --utterance_ids "1"

# Check output
cat /home/track1_result/trajectory/Q_1_trajectory.json | jq .
```

### Step 3: Make Your First Edit (30 min)
- Open `src/agent_hive/workflows/track1_planning.py`
- Find Edit 2 (lines 167-198: `get_prompt()`)
- Enhance the prompt with better guidance
- Save file

### Step 4: Test Your Edit (10 min)
```bash
# Run again with your edit
python run_track_1.py --utterance_ids "1"

# Compare
diff /home/track1_result/trajectory/Q_1_trajectory.json <baseline>
```

### Step 5: Analyze Results (15 min)
```bash
# Run failure mode analysis
cd ../../src/TrajFM
python failure_mode_extractor.py \
    --traj_directory "/home/track1_result/trajectory"

# Review failures - use to improve prompt
cat summary/failure_modes_clustered.csv
```

### Step 6: Iterate (repeat 3-5 daily)
- Make improvements based on failure analysis
- Test with different utterance IDs
- Refine prompts iteratively

---

## 💡 Key Concepts

### Track 1: Planning Track
**Focus:** Decomposing complex questions into multi-agent plans

**How it works:**
1. User asks a complex question
2. LLM reads available agents and your planning prompt
3. LLM generates multi-step plan with format: `#Task`, `#Agent`, `#Dependency`, `#ExpectedOutput`
4. System parses plan and executes sequentially
5. Each agent gets context from previous steps

**What you control:**
- How agents are described (Edit 1)
- The planning prompt/strategy (Edit 2) ⭐ HIGH IMPACT

**Success means:**
- Good plan decomposition
- Efficient dependencies
- Correct agent assignments

---

### Track 2: Execution Track
**Focus:** Dynamic, resilient multi-agent execution

**How it works:**
1. System receives tasks (may skip planning, use pre-made tasks)
2. For each task, (optionally) validate/enrich input with helper
3. Execute with primary agent (or fallback to secondary)
4. Store response and pass as context to next task
5. Handle failures gracefully

**What you control:**
- Task validation/enrichment logic (Edit 1)
- Execution scheduling and fallback strategies (Edit 2) ⭐ HIGH IMPACT

**Success means:**
- Resilient to failures
- Good context reuse
- Smart fallback strategies

---

## 🎓 Understanding the Agents

### Available Agents You Can Use:

1. **IoT Data Download** (`iot_bms_tools`)
   - Queries sensor data from CouchDB
   - Lists sites, assets, sensors
   - Downloads time-series data
   - **Best for:** Data collection phase

2. **Failure Mode & Sensor Relevancy** (FMSR Agent)
   - Provides failure mode knowledge
   - Maps sensors to failure modes
   - Suggests monitoring strategies
   - **Best for:** Diagnosis phase

3. **Time Series Analytics & Forecasting** (TSFM Agent)
   - Detects anomalies
   - Forecasts trends
   - Analyzes patterns
   - **Best for:** Analysis phase

4. **Work Order Agent**
   - Generates maintenance work orders
   - Schedules maintenance
   - Tracks work progress
   - **Best for:** Action/remediation phase

---

## 📊 Expected Workflow

```
Week 1: Setup & Baseline
├─ Read documentation
├─ Set up environment
├─ Run baseline (no edits)
└─ Understand current performance

Week 2-3: Track 1 Optimization
├─ Edit 1: Agent info formatting
├─ Edit 2: Planning prompt (HIGH IMPACT)
├─ Test on 10-20 scenarios
└─ Analyze failures with TrajFM

Week 3-4: Track 2 Optimization
├─ Edit 3: Task helper agent
├─ Edit 4: Execution logic (HIGH IMPACT)
├─ Test on 10-20 scenarios
└─ Final failure analysis

Week 4: Polish & Submit
├─ Fine-tune both tracks
├─ Final comprehensive testing
├─ Document changes
└─ Submit results
```

---

## 🏆 Competition Scoring

Your agents will be scored on:

1. **Task Correctness** (40%)
   - Did agents solve the problem correctly?
   - Accuracy of identified issues
   - Quality of recommendations

2. **Efficiency** (30%)
   - Number of tool calls
   - Number of steps
   - Reasoning path complexity

3. **Reasoning Quality** (20%)
   - Logical soundness of plan
   - Correct dependency ordering
   - Appropriate agent selection

4. **Robustness** (10%)
   - Handled edge cases
   - Fallback strategies worked
   - Minimal failure modes detected

---

## 📝 Failure Modes You'll Be Evaluated On

The system analyzes your agent trajectories for these failure patterns:

| Category | Failure Mode |
|----------|--------------|
| **Task Understanding** | Disobey Task Specification |
| | Disobey Role Specification |
| | Step Repetition |
| **Context** | Loss of Conversation History |
| | Unaware of Termination Conditions |
| **Communication** | Conversation Reset |
| | Fail to Ask for Clarification |
| | Task Derailment |
| | Information Withholding |
| | Ignored Other Agent's Input |
| | Action-Reasoning Mismatch |
| **Verification** | Premature Termination |
| | No or Incorrect Verification |
| | Weak Verification |

**Your goal:** Minimize these failures through better planning and execution logic.

---

## 🔗 External Resources

### Official Links:
- **GitHub:** https://github.com/IBM/AssetOpsBench
- **Paper:** https://arxiv.org/pdf/2506.03828
- **Dataset:** https://huggingface.co/datasets/ibm-research/AssetOpsBench
- **Blog:** https://research.ibm.com/blog/asset-ops-benchmark

### Key Concepts:
- **React Agent:** Reasoning + Acting pattern
- **Few-Shot Learning:** Guide LLM with examples
- **Multi-Agent Systems:** Specialized agents coordinating
- **Tool Use:** LLM calling external functions

---

## ❓ FAQ

**Q: Which track should I focus on first?**
A: Start with Track 1 - it's foundational. Good planning makes Track 2 easier.

**Q: How much time should I spend on each edit?**
A: Track 1 Edit 2 (planning prompt) - 40% of time  
Track 2 Edit 2 (execution) - 40% of time  
Edits 1 & Polish - 20% of time

**Q: Can I modify the agents or tools?**
A: No, those are fixed. Only edit the marked TODO sections in planning/execution workflows.

**Q: How do I know if my edit is better?**
A: Compare trajectories before/after. Run TrajFM to see if failure modes decreased.

**Q: Can I add new agents?**
A: No, use only the 4 provided agents. Focus on optimizing their coordination.

**Q: What's the most impactful change I can make?**
A: Better planning prompt (Track 1 Edit 2). A well-crafted prompt dramatically improves plan quality.

---

## ✅ Pre-Submission Checklist

Before submitting your competition entry:

- [ ] All edits are in marked TODO sections only
- [ ] No changes to base classes or tools
- [ ] No new imports added
- [ ] Safety constraints maintained (15 loops, <5 steps)
- [ ] Regex patterns unchanged (#Task, #Agent, #Dependency, #ExpectedOutput)
- [ ] Tested on at least 10 different scenarios
- [ ] Compared baseline vs. improved (impact confirmed)
- [ ] Ran failure mode analysis
- [ ] Track 1 and Track 2 both working
- [ ] Code follows existing style
- [ ] No syntax errors
- [ ] Ready to share

---

## 🎓 Learning Resources (Inside Repo)

**Study these files to understand each agent:**
- `src/meta_agent/agents/IoT/IoTAgentFewShots.py` - What IoT agent can do
- `src/meta_agent/agents/FMSR/FMSRAgentFewShots.py` - Failure mode knowledge
- `src/meta_agent/agents/TSFM/TSFMAgentFewShots.py` - Time series analysis
- `src/meta_agent/agents/WorkOrder/WorkOrderFewShots.py` - Work order generation

**Understand the execution:**
- `src/agent_hive/workflows/planning_review.py` - Original planning (reference)
- `src/agent_hive/workflows/sequential.py` - How tasks execute (reference)
- `src/evaluation/analyze.py` - How results are evaluated

---

## 🚀 Next Action

1. **Right now (5 min):** Read QUICK_REFERENCE.md
2. **Next (10 min):** Open IDE, locate the 4 editable sections
3. **Then (30 min):** Study one agent's few-shots (e.g., IoTAgentFewShots.py)
4. **After (1 hour):** Make your first edit to Track 1 Edit 2 (planning prompt)
5. **Finally (30 min):** Test and compare results

---

**You've got this! 🎯 The competition is designed to help you build better agents. Start with the prompt engineering—that's where the magic happens! ✨**

---

## 📞 Support

If you have questions:
1. Check the QUICK_REFERENCE.md FAQ section
2. Review CONNECTION_MAP.md data flows
3. Look at actual code in `track1_planning.py` and `track2_execution.py`
4. Study agent few-shots to understand capabilities
5. Run failure mode analysis to debug issues

**Good luck! 🚀**
