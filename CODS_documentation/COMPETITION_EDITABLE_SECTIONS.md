# 🏆 AssetOpsBench CODS Competition - Editable Sections Guide

**Competition Focus:** Autonomous AI agents solving industrial maintenance tasks through multi-agent reasoning and coordination.

**Your Challenge:** Optimize agents' planning, reasoning, and execution strategies within clearly defined editable sections.

---

## 📊 Competition Structure

The competition has **2 tracks**, each with specific editable sections:

| Track | Focus | Editable Component | Entry Point |
|-------|-------|-------------------|------------|
| **Track 1** | Task Planning | Planning prompt + Agent info formatting | `run_track_1.py` |
| **Track 2** | Dynamic Execution | Task revision logic + Workflow execution | `run_track_2.py` |

---

## 🎯 TRACK 1: Task Planning Optimization

### **Purpose**
Improve how the LLM decomposes complex industrial questions into executable multi-agent plans.

### **File Structure**

```
src/agent_hive/workflows/
├── track1_planning.py          ← YOUR PRIMARY FILE
├── track1_fact_sheet.json      ← "Task Planning"
└── base_workflow.py            ← Do NOT edit
```

**Execution Flow:**
```
benchmark/cods_track1/run_track_1.py
    ↓
agent_hive.workflows.track1_planning.NewPlanningWorkflow.run()
    ↓
generate_steps()                 ← Calls LLM to create plan
    ↓
get_prompt()                     ← YOUR EDITABLE SECTION #1
    ↓
SequentialWorkflow.run()         ← Executes planned steps (DO NOT EDIT)
```

---

### **Editable Section 1: Agent Information Formatting**

**File:** `/src/agent_hive/workflows/track1_planning.py`  
**Lines:** 65-89  
**Purpose:** Format how agent descriptions are presented to the LLM

#### Current Code:
```python
# =========================================================
# TODO: Participants can edit this section ONLY
# 🎨 Purpose: Customize how agent information is collected and formatted
# ✅ Allowed: 
#     - Change numbering style or bullet points
#     - Include additional metadata (e.g., agent capabilities, tags)
#     - Provide examples in a different format
#     - Add emojis or formatting to make the prompt clearer 
#     - More thinking
# ❌ Not allowed: 
#     - Modify workflow execution
#     - Replace the base ReAct agent or Executor
#     - Change memory or retry logic
# =========================================================

for ii, aagent in enumerate(task.agents):
    agent_descriptions += f"\n({ii + 1}) Agent name: {aagent.name}"
    agent_descriptions += f"\nAgent description: {aagent.description}"
    if "task_examples" in aagent.__dict__ and aagent.task_examples:
        agent_descriptions += f"\nTasks that agent can solve:"
        for idx, task_example in enumerate(aagent.task_examples, start=1):
            agent_descriptions += f"\n{idx}. {task_example}"
    agent_descriptions += "\n"

# =========================================================
# END OF EDITABLE SECTION
# =========================================================
```

#### What You Can Change:
- **Numbering style:** Use bullets, emojis, hierarchical format
- **Agent info organization:** Add capabilities, difficulty ratings, tool counts
- **Example formatting:** Group examples by category, add descriptions
- **Metadata:** Add tags like `[Sensing]`, `[Decision-Making]`, `[Actuation]`
- **Prompt guidance:** Add hints about when to use each agent

#### Example Enhancement:
```python
for ii, aagent in enumerate(task.agents):
    agent_descriptions += f"\n🤖 [{ii + 1}] **{aagent.name}**"
    agent_descriptions += f"\n   📝 {aagent.description}"
    agent_descriptions += f"\n   🛠️ Tools: {len(aagent.tools)}"
    
    if "task_examples" in aagent.__dict__ and aagent.task_examples:
        agent_descriptions += f"\n   ✅ Can solve:"
        for idx, task_example in enumerate(aagent.task_examples, start=1):
            agent_descriptions += f"\n      {idx}. {task_example[:60]}..."
    agent_descriptions += "\n"
```

---

### **Editable Section 2: Planning Prompt Template**

**File:** `/src/agent_hive/workflows/track1_planning.py`  
**Lines:** 167-198  
**Purpose:** Craft the system prompt that guides LLM plan generation

#### Current Code:
```python
def get_prompt(self, task_description, agent_descriptions):
    # =========================================================
    # TODO: Participants can edit this section ONLY
    # 🎨 Purpose: Improve prompt clarity, formatting, emojis, guidance
    # ✅ Allowed: Wording, structure, examples, emojis
    # ❌ Not allowed: Changing workflow, ReAct agent, Executor, or memory logic
    # =========================================================

    prompt = f"""
🚀 You are an AI assistant tasked with creating a step-by-step plan to solve a complex problem using the external agents provided.  

⚠️ Constraints:
- Only use the agents listed below. No new agents may be added.
- The base ReAct agent and Executor component are fixed. Do not change them.
- Produce a plan with fewer than 5 steps.
- Include Task, Agent, Dependency, and ExpectedOutput for each step.
- Make instructions clear, unambiguous, and actionable.

Each step must follow this format:
#Task<N>: <Describe your task here>
#Agent<N>: <agent_name>
#Dependency<N>: <use #S1, #S2, ... or None>
#ExpectedOutput<N>: <Expected output>

## Here are the available agents: ##
{agent_descriptions}

## Problem to solve: ##
{task_description}

Output (your generated plan) ⬇️:
"""
    # =========================================================
    # End of participant editable section
    # =========================================================
    return prompt
```

#### What You Can Change:
- **Instructions clarity:** Add more detail, examples, or constraints
- **Format specification:** Change regex patterns (but keep `#Task`, `#Agent`, `#Dependency`, `#ExpectedOutput`)
- **Step limit:** Suggest strategies (e.g., "typically 3-4 steps")
- **Reasoning guidance:** Add heuristics ("Start with data collection, then analysis, then action")
- **Quality metrics:** Ask LLM to consider efficiency, clarity, dependencies
- **Industry context:** Add hints specific to maintenance/operations domain

#### Example Enhancement:
```python
prompt = f"""
🚀 You are an expert Industrial Operations Planner creating step-by-step action plans.

📋 CRITICAL GUIDELINES:
1. For maintenance tasks: Data Collection → Analysis → Action
2. Always verify data completeness before proceeding
3. Consider temporal dependencies (e.g., sensors must report before analysis)
4. Steps should be independent and parallelizable where possible

⚠️ Constraints:
- Only use agents listed below
- Maximum 4 steps (prefer 3)
- Task descriptions must be specific with parameters
- Agent names must EXACTLY match available agents

Format (MUST follow this exactly):
#Task<N>: <specific, actionable task description with parameters>
#Agent<N>: <exact agent name>
#Dependency<N>: <None or #S1, #S2, ...>
#ExpectedOutput<N>: <expected format and key fields>

## Available Agents (sorted by execution order): ##
{agent_descriptions}

## Your Problem: ##
{task_description}

Create an efficient, dependency-aware plan:
"""
```

---

### **Connections to Other Files**

#### ✅ Uses (Read-Only):
- `agent_hive/agents/react_reflect_agent.py` - Executes planned tasks
- `agent_hive/workflows/sequential.py` - Runs the plan sequentially
- `agent_hive/tools/fmsr.py`, `skyspark.py`, `tsfm.py`, `wo.py` - Tools available to agents
- `agent_hive/task.py` - Task data structure

#### ↪️ Called By:
- `benchmark/cods_track1/run_track_1.py` - Main entry point
  ```python
  from agent_hive.workflows.track1_planning import NewPlanningWorkflow
  
  wf = NewPlanningWorkflow(tasks=[task], llm=llm_model)
  result = wf.run()  # Calls your get_prompt() and generate_steps()
  ```

#### 🔄 Data Flow:
```
run_track_1.py (utterance question)
    ↓
NewPlanningWorkflow.__init__()
    ↓
generate_steps() calls get_prompt()          [YOUR SECTION #2]
    ↓
watsonx_llm(prompt) → LLM generates plan
    ↓
Plan regex parsing: #Task, #Agent, #Dependency, #ExpectedOutput
    ↓
Create Task objects with dependencies
    ↓
SequentialWorkflow.run() executes each task
    ↓
Each task calls assigned_agent.execute_task()
    ↓
agent.execute_task() calls ReactReflectAgent → LLM + Tools
    ↓
Result saved to: /home/track1_result/trajectory/Q_<id>_trajectory.json
```

---

## 🎯 TRACK 2: Dynamic Execution Optimization

### **Purpose**
Implement flexible multi-agent execution with task validation, fallback strategies, and dynamic scheduling.

### **File Structure**

```
src/agent_hive/workflows/
├── track2_execution.py         ← YOUR PRIMARY FILE
├── track2_fact_sheet.json      ← "Task Execution"
└── base_workflow.py            ← Do NOT edit
```

**Execution Flow:**
```
benchmark/cods_track2/run_track_2.py
    ↓
agent_hive.workflows.track2_execution.DynamicWorkflow.run()
    ↓
(For each task) _build_input()
    ↓
assigned_agent.execute_task()   [YOUR EDITABLE SECTION #1]
    ↓
Response processing            [YOUR EDITABLE SECTION #2]
```

---

### **Editable Section 1: Task Revision Helper Agent**

**File:** `/src/agent_hive/workflows/track2_execution.py`  
**Lines:** 31-81  
**Purpose:** Implement logic to validate, revise, or enhance task inputs before execution

#### Current Code:
```python
# =========================================================
# 🛠️ EDITABLE SECTION for Participants
# 🎯 Purpose: Implement revision / validation of task inputs
#
# ✅ Allowed:
#   - Define your own revision rules (clarity check, validation, enrichment)
#   - Add formatting styles (bullet points, numbered lists, emojis, etc.)
#   - Suggest metadata (tags, difficulty, clarity score)
#   - Return multiple variants of a revision
#
# ❌ Not Allowed:
#   - Modify workflow execution logic outside this agent
#   - Replace the base ReAct agent or Executor
#   - Change memory persistence or retry logic
# =========================================================

class TaskRevisionHelperAgent(BaseAgent):
    """
    Template for TaskRevisionHelperAgent.
    Participants should implement logic for revising/validating tasks.
    """

    name = "TaskRevisionHelperAgent"
    description = "Revises, validates, and provides suggestions for task inputs."
    memory = []
    tools = []

    def __init__(self, llm: str = None, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def execute_task(self, task_input: str) -> str:
        """
        Review and revise the given task input, suggesting improvements or validation.

        Args:
            task_input (str): The task description to revise.

        Returns:
            str: Revised/validated task description or feedback.
        """
        # =========================================================
        # 🚧 TODO: Implement your revision logic here
        # Example ideas:
        #   - Fix grammar / spelling
        #   - Improve clarity of task instructions
        #   - Suggest metadata (tags, difficulty, etc.)
        #   - Return multiple variants of the revision
        #
        # 👉 IMPORTANT: Do NOT modify anything outside this method
        # =========================================================
        raise NotImplementedError("Participants must implement this method.")
```

#### What You Should Implement:
- **Clarity validation:** Check if task is unambiguous
- **Enrichment:** Add context, parameters, or metadata
- **Error handling:** Detect malformed inputs
- **Multi-variants:** Return multiple interpretations for ambiguous tasks
- **Scoring:** Add quality/difficulty metrics

#### Example Implementation:
```python
def execute_task(self, task_input: str) -> str:
    """
    Validate and enrich task input for better agent execution.
    """
    # Check for completeness
    quality_issues = []
    if len(task_input.split()) < 5:
        quality_issues.append("Task too brief - add more context")
    if "?" not in task_input and "." not in task_input:
        quality_issues.append("Task lacks clear intent markers")
    
    # Enrich with metadata
    revised = f"""
ORIGINAL: {task_input}

CLARIFICATIONS:
- Domain: Industrial Maintenance / Operations
- Data types: Time-series, logs, sensor readings
- Expected tools: IoT, TSFM, FMSR agents

QUALITY CHECKS:
{chr(10).join(f"- ❌ {issue}" for issue in quality_issues) if quality_issues else "- ✅ Task is clear"}

ENHANCED TASK FOR AGENT:
{task_input}
When solving this, prioritize:
1. Data validation
2. Anomaly detection
3. Root cause analysis
"""
    return revised
```

---

### **Editable Section 2: Workflow Execution Logic**

**File:** `/src/agent_hive/workflows/track2_execution.py`  
**Lines:** 141-180  
**Purpose:** Customize how agents are scheduled, executed, and their responses processed

#### Current Code:
```python
def run(self):
    """
    Execute tasks dynamically.
    Participants can edit only the marked TODO section to introduce:
        - parallelism
        - conditional execution
        - helper agents
        - fallback strategies
    """
    self.memory = []

    # =========================================================
    # 🚧 TODO: Participants can edit this section ONLY
    # 🎨 Purpose: Customize how agents are scheduled and executed
    #
    # ✅ Allowed:
    #   - Replace for-loop with while-loop (max iterations fixed at 15)
    #   - Use TaskRevisionHelperAgent to refine or validate responses
    #   - Experiment with combining multiple agent responses
    #   - Add fallback strategies (e.g., retry with helper agent)
    #
    # ❌ Not Allowed:
    #   - Remove safety cap on iterations (max = 15 must remain)
    #   - Change memory persistence logic
    #   - Replace Executor orchestration logic
    # =========================================================
    self.context_type = ContextType.SELECTED
    max_loops = 15
    i = 0
    while i < len(self.tasks) and i < max_loops:
        task = self.tasks[i]
        task_no = i + 1
        logger.info(f"Task {task_no}: {task.description}")

        assigned_agents = task.agents

        # Build input with context
        user_input = self._build_input(task, i)

        # Baseline: Execute with first agent
        response = assigned_agents[0].execute_task(user_input)

        # 👉 OPTIONAL: Use TaskRevisionHelperAgent here
        # helper = TaskRevisionHelperAgent()
        # response = helper.execute_task(response)

        # 👉 OPTIONAL: Combine or compare multiple agent responses
        response = response.replace("Final Answer:","").strip()
        self.memory.append(response)

        i += 1

    history = self.generate_history()
    print(json.dumps(history, indent=4))
    return history
```

#### What You Can Change:
- **Execution pattern:** Sequential → Loop-based with conditions
- **Multi-agent coordination:** If task has multiple agents, combine their responses
- **Helper integration:** Use TaskRevisionHelperAgent for validation/refinement
- **Fallback strategies:** Retry with different agent if first fails
- **Context propagation:** Customize how context flows between tasks
- **Response processing:** Clean, aggregate, or enhance agent outputs

#### Example Enhancement:
```python
def run(self):
    self.memory = []
    self.context_type = ContextType.SELECTED
    max_loops = 15
    i = 0
    
    # Optional: Initialize helper
    helper = TaskRevisionHelperAgent(llm=self.llm)
    
    while i < len(self.tasks) and i < max_loops:
        task = self.tasks[i]
        task_no = i + 1
        logger.info(f"Task {task_no}: {task.description}")
        
        assigned_agents = task.agents
        user_input = self._build_input(task, i)
        
        # ENHANCEMENT 1: Pre-process with helper
        enriched_input = helper.execute_task(user_input)
        
        # ENHANCEMENT 2: Try primary agent
        response = None
        try:
            response = assigned_agents[0].execute_task(enriched_input)
        except Exception as e:
            logger.warning(f"Agent failed: {e}")
            
            # ENHANCEMENT 3: Fallback to second agent if available
            if len(assigned_agents) > 1:
                logger.info(f"Trying fallback agent: {assigned_agents[1].name}")
                response = assigned_agents[1].execute_task(user_input)
        
        if response is None:
            response = "FAILED TO EXECUTE TASK"
        
        # ENHANCEMENT 4: Post-process response
        response = response.replace("Final Answer:", "").strip()
        
        # ENHANCEMENT 5: Validate response quality
        if len(response) < 10:
            logger.warning(f"Response too short for task {task_no}")
        
        self.memory.append(response)
        i += 1
    
    history = self.generate_history()
    print(json.dumps(history, indent=4))
    return history
```

---

### **Connections to Other Files**

#### ✅ Uses (Read-Only):
- `agent_hive/agents/react_reflect_agent.py` - Task execution
- `agent_hive/enum.py` - ContextType (SELECTED, ALL, PREVIOUS, DISABLED)
- `agent_hive/task.py` - Task data structure
- `agent_hive/workflows/base_workflow.py` - Base class

#### ↪️ Called By:
- `benchmark/cods_track2/run_track_2.py` - Main entry point
  ```python
  from agent_hive.workflows.track2_execution import DynamicWorkflow
  
  wf = DynamicWorkflow(
      tasks=task_list,
      context_type=ContextType.SELECTED
  )
  result = wf.run()  # Calls your run() method
  ```

#### 🔄 Data Flow:
```
run_track_2.py (utterance question)
    ↓
DynamicWorkflow.__init__()
    ↓
run() loop for each task                    [YOUR SECTION #2]
    ↓
_build_input() creates context-aware input  (DO NOT EDIT)
    ↓
assigned_agent.execute_task(user_input)
    ↓
Agent calls ReactReflectAgent → LLM + Tools
    ↓
Response stored in self.memory
    ↓
Context flows to next task
    ↓
Result saved to: /home/track1_result/trajectory/Q_<id>_trajectory.json
```

---

## 🔑 Key Constraints & Rules

### **What You CANNOT Change**

❌ **Never modify:**
- Agent implementation (`base_agent.py`, `react_reflect_agent.py`)
- Tool definitions (`fmsr.py`, `skyspark.py`, `tsfm.py`, `wo.py`)
- Memory logic or retry mechanisms
- Context type enum
- Regex patterns for parsing plans (keep `#Task`, `#Agent`, `#Dependency`, `#ExpectedOutput`)
- Task class structure
- Workflow orchestration outside marked sections

❌ **Constraints that must be respected:**
- **Max iterations:** Track 2 limited to 15 loops
- **Max steps:** Planning prompts should suggest < 5 steps
- **Agent names:** Must match exactly (case-sensitive)
- **Output format:** Maintain JSON trajectory structure

### **What You CAN Change**

✅ **Track 1 (Planning):**
- Agent info formatting and structure
- Planning prompt wording, structure, examples, and guidance
- Constraints and requirements in the prompt
- Emojis, formatting, bullet points

✅ **Track 2 (Execution):**
- Implement TaskRevisionHelperAgent logic
- Add fallback strategies
- Customize response processing
- Add helper agent integration
- Enhance context building logic (within `_build_input` output usage)

---

## 📈 How Scoring Works

Your competition results will be evaluated on:

1. **Correctness** - Did agents solve the task correctly?
2. **Efficiency** - How many steps/tool calls were needed?
3. **Reasoning Quality** - Are the plans logical and dependencies correct?
4. **Failure Mode Detection** - TrajFM analyzes trajectory for agent errors:
   - Step Repetition
   - Action-Reasoning Mismatch
   - Loss of Conversation History
   - Premature Termination
   - And 10 other failure modes

---

## 🚀 How to Test Your Changes

### **Track 1 Test:**
```bash
cd /Users/srutanik/AssetOpsBench_CODS/benchmark/cods_track1

python run_track_1.py \
    --utterance_ids "1,106,42" \
    --generate_steps_only False

# Results in: /home/track1_result/trajectory/Q_<id>_trajectory.json
```

### **Track 2 Test:**
```bash
cd /Users/srutanik/AssetOpsBench_CODS/benchmark/cods_track2

python run_track_2.py \
    --utterance_ids "1,106,42" \
    --generate_steps_only False

# Results in: /home/track1_result/trajectory/Q_<id>_trajectory.json
```

### **Analyze Results:**
```bash
# View trajectory
cat /home/track1_result/trajectory/Q_1_trajectory.json | jq .

# Run failure mode detection
cd /Users/srutanik/AssetOpsBench_CODS/src/TrajFM
python failure_mode_extractor.py \
    --traj_directory "/home/track1_result/trajectory" \
    --summary_dir "./summary" \
    --model_id 16
```

---

## 📋 Competition Checklist

- [ ] Clone/fork the repository with `Competition_CODS` branch
- [ ] Set up environment (conda, Docker)
- [ ] Understand Track 1 planning prompt (read `track1_planning.py`)
- [ ] Modify Agent info formatting (Track 1, Section 1)
- [ ] Optimize planning prompt (Track 1, Section 2)
- [ ] Test Track 1: `python run_track_1.py --utterance_ids "1,2,3"`
- [ ] Implement TaskRevisionHelperAgent (Track 2, Section 1)
- [ ] Enhance execution logic (Track 2, Section 2)
- [ ] Test Track 2: `python run_track_2.py --utterance_ids "1,2,3"`
- [ ] Run failure mode analysis on trajectories
- [ ] Refine based on failure patterns
- [ ] Submit results

---

## 🔗 Reference Files

| File | Purpose | Editable? |
|------|---------|-----------|
| `track1_planning.py` | Planning workflow | ✅ YES (2 sections) |
| `track2_execution.py` | Execution workflow | ✅ YES (2 sections) |
| `base_workflow.py` | Base class | ❌ NO |
| `sequential.py` | Sequential executor | ❌ NO |
| `react_reflect_agent.py` | Agent implementation | ❌ NO |
| `task.py` | Task model | ❌ NO |
| `run_track_1.py` | Track 1 entry | ❌ NO |
| `run_track_2.py` | Track 2 entry | ❌ NO |
| `failure_mode_extractor.py` | Analysis tool | ❌ NO (reference only) |

---

## 💡 Pro Tips

1. **Track 1 Focus:** Better prompts = better plans. Invest time in clear, specific instructions.
2. **Track 2 Focus:** Resilience matters. Add fallback strategies and validation.
3. **Failure analysis:** Use TrajFM to identify patterns in failures, then optimize prompts.
4. **Few-shots matter:** Check `IoTAgentFewShots.py`, `FMSRAgentFewShots.py`, etc. Your prompt should align with agent capabilities.
5. **Tool availability:** Understand what each agent can do before writing prompts.
6. **Context flow:** In Track 2, context passes between tasks—design your execution logic to maximize reuse.

---

## 📞 Support Resources

- Repository: `https://github.com/IBM/AssetOpsBench` (Competition_CODS branch)
- Dataset: `ibm-research/AssetOpsBench` on HuggingFace
- Paper: https://arxiv.org/pdf/2506.03828
- Blog: https://research.ibm.com/blog/asset-ops-benchmark

---

**Good luck! 🚀 Remember: The goal is to create intelligent agents that can reason about complex industrial problems!**
