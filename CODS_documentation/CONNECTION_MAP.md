# 🔗 Editable Sections Connection Map

## TRACK 1: Planning Track Flow & Connections

```
┌─────────────────────────────────────────────────────────────────┐
│            benchmark/cods_track1/run_track_1.py                 │
│  (Entry Point: Loads scenarios, calls planning workflow)        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ Creates task with agents
                       ▼
        ┌──────────────────────────────────────┐
        │ NewPlanningWorkflow.__init__()       │
        │ (agent_hive/workflows/                │
        │  track1_planning.py:30-50)            │
        │                                       │
        │ ✅ Stores agents, LLM, task          │
        │ ✅ Validates task structure          │
        └──────────────────────┬────────────────┘
                               │
                               │ wf.run()
                               ▼
        ┌──────────────────────────────────────────────┐
        │ NewPlanningWorkflow.run()                    │
        │ (track1_planning.py:52-60)                   │
        │                                              │
        │ ✅ Calls generate_steps()                   │
        │ ✅ Creates SequentialWorkflow               │
        │ ✅ Returns trajectory                       │
        └──────────────────┬───────────────────────────┘
                           │
                           │ generate_steps()
                           ▼
        ┌──────────────────────────────────────────────────┐
        │ NewPlanningWorkflow.generate_steps()            │
        │ (track1_planning.py:62-145)                      │
        │                                                  │
        │ Step 1: Build agent descriptions ⭐ EDITABLE    │
        │ ┌────────────────────────────────────────────┐  │
        │ │ Editable Section 1: Lines 65-89            │  │
        │ │ - Format agent info (name, description)    │  │
        │ │ - Structure task examples                  │  │
        │ │ - Add metadata or emojis                   │  │
        │ │                                            │  │
        │ │ CURRENT: Simple numbered list             │  │
        │ │ "(1) Agent name: IoT Data Download"        │  │
        │ │ "Agent description: ..."                   │  │
        │ │ "Tasks that agent can solve:"              │  │
        │ │ "1. send me the values..."                 │  │
        │ │                                            │  │
        │ │ ENHANCE: Add emojis, better organization  │  │
        │ │ "🤖 [1] IoT Data Download"                │  │
        │ │ "📝 Can retrieve sensor data..."           │  │
        │ │ "✅ Can solve:"                           │  │
        │ │ "  • Query sensor values"                  │  │
        │ └────────────────────────────────────────────┘  │
        │                                                  │
        │ Step 2: Call get_prompt() ⭐ EDITABLE          │
        │ └──> See below                                  │
        │                                                  │
        │ Step 3: LLM generates plan                       │
        │ └──> watsonx_llm(prompt, model_id)             │
        │                                                  │
        │ Step 4: Parse plan with regex                   │
        │ └──> Extract #Task, #Agent, #Dependency        │
        │                                                  │
        │ Step 5: Create Task objects                     │
        │ └──> For each step: new Task(description, ...)│
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ NewPlanningWorkflow.get_prompt()                │
        │ (track1_planning.py:167-198) ⭐ EDITABLE         │
        │                                                  │
        │ Editable Section 2:                             │
        │ - System prompt for LLM                         │
        │ - Constraints and requirements                  │
        │ - Output format specification                   │
        │ - Guidance for decomposition                    │
        │ - Planning heuristics                           │
        │                                                  │
        │ CURRENT: Basic prompt with agent list           │
        │ "You are an AI assistant creating plans..."     │
        │ "Only use agents listed below"                  │
        │ "Produce plan with < 5 steps"                   │
        │ "Use format: #Task, #Agent, #Dependency, ..."   │
        │                                                  │
        │ ENHANCE: Add domain-specific guidance           │
        │ "For maintenance tasks: data → analysis → action│
        │ "Consider temporal dependencies"                │
        │ "Prioritize: 1) Data validation..."            │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ Returns prompt string
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ watsonx_llm(prompt) [LLM INFERENCE]             │
        │                                                  │
        │ Takes: prompt from get_prompt()                 │
        │ Returns: Multi-step plan with format:          │
        │                                                  │
        │ ## Step 1                                       │
        │ #Task1: Download sensor data for Chiller 6    │
        │ #Agent1: IoT Data Download                     │
        │ #Dependency1: None                             │
        │ #ExpectedOutput1: JSON with sensor readings    │
        │                                                  │
        │ ## Step 2                                       │
        │ #Task2: Detect anomalies in data               │
        │ #Agent2: Time Series Analytics...              │
        │ #Dependency2: #S1                              │
        │ #ExpectedOutput2: List of anomalies...         │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ Regex parsing
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ Task objects created from parsed plan           │
        │ Each with: description, expected_output,        │
        │            agent reference, dependency list     │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ Returned as planned_tasks[]
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ SequentialWorkflow(tasks=planned_tasks)         │
        │ (agent_hive/workflows/sequential.py)            │
        │                                                  │
        │ ✅ Initializes with planned tasks               │
        │ ✅ Sets context_type=ContextType.SELECTED       │
        │ ✅ Does NOT edit (read-only)                    │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ .run()
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ SequentialWorkflow.run() Loop                   │
        │ (sequential.py:65-110)                          │
        │                                                  │
        │ For each task in planned_tasks:                │
        │  1. Build user_input with context              │
        │  2. agent.execute_task(user_input)             │
        │  3. Store response in self.memory              │
        │  4. Next task gets previous context            │
        │                                                  │
        │ DO NOT EDIT (read-only)                        │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ For each task
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ ReactReflectAgent.execute_task(user_input)      │
        │ (agent_hive/agents/react_reflect_agent.py)      │
        │                                                  │
        │ Creates ReactReflectXenAgent:                   │
        │ - question = user_input                         │
        │ - cbm_tools = [all agent tools]                │
        │ - react_example = few_shots (from Tool files)  │
        │ - num_reflect_iteration = 5                    │
        │                                                  │
        │ Calls: .run()                                   │
        │ Returns: self.answer                            │
        │                                                  │
        │ DO NOT EDIT (read-only)                        │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ ReActXen React-Reflect Loop                     │
        │ (external: reactxen library)                    │
        │                                                  │
        │ For each iteration (max 6 steps):              │
        │  - Thought: LLM reasons about problem          │
        │  - Action: Choose tool to call                 │
        │  - Observation: Tool result                    │
        │                                                  │
        │ For each reflection iteration (max 5):          │
        │  - LLM reviews and refines answer              │
        │                                                  │
        │ Returns: Final Answer                           │
        │ DO NOT EDIT (read-only, external)              │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ Response
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ Response stored in self.memory[]                │
        │ (sequential.py:107)                             │
        │                                                  │
        │ memory[0] = Task 1 response (IoT data)         │
        │ memory[1] = Task 2 response (anomalies)        │
        │ memory[2] = Task 3 response (failures)         │
        │                                                  │
        │ Next task uses this as context                  │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ All tasks done
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ generate_history() creates trajectory structure │
        │ (sequential.py:113-127)                         │
        │                                                  │
        │ Returns:                                        │
        │ [                                               │
        │   {                                             │
        │     "task_number": 1,                          │
        │     "task_description": "...",                 │
        │     "agent_name": "IoT Data Download",         │
        │     "response": "..."                          │
        │   },                                            │
        │   ...                                           │
        │ ]                                               │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ Output saved to:                                │
        │ /home/track1_result/trajectory/                │
        │ Q_<utterance_id>_trajectory.json               │
        │                                                  │
        │ {                                               │
        │   "id": 1,                                      │
        │   "text": "Question",                           │
        │   "trajectory": [...]                          │
        │ }                                               │
        └──────────────────────────────────────────────────┘
```

---

## TRACK 2: Dynamic Execution Flow & Connections

```
┌─────────────────────────────────────────────────────────────────┐
│            benchmark/cods_track2/run_track_2.py                 │
│  (Entry Point: Loads scenarios, calls execution workflow)       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ Creates tasks (may skip planning)
                       ▼
        ┌──────────────────────────────────────┐
        │ DynamicWorkflow.__init__()           │
        │ (agent_hive/workflows/                │
        │  track2_execution.py:88-110)         │
        │                                       │
        │ ✅ Stores tasks, context_type        │
        │ ✅ Validates task structure          │
        │ ✅ Supports multiple agents/task     │
        └──────────────────────┬────────────────┘
                               │
                               │ wf.run()
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ DynamicWorkflow.run()                           │
        │ (track2_execution.py:128-195) ⭐ EDITABLE        │
        │                                                  │
        │ Editable Section 2 (Main Execution Loop):       │
        │                                                  │
        │ ┌──────────────────────────────────────────────┐│
        │ │ Lines 141-180: While loop customization      ││
        │ │                                              ││
        │ │ CURRENT:                                     ││
        │ │ while i < len(tasks) and i < 15:            ││
        │ │   task = tasks[i]                           ││
        │ │   user_input = _build_input(task, i)         ││
        │ │   response = agents[0].execute_task(...)     ││
        │ │   response = clean_response(response)        ││
        │ │   memory.append(response)                    ││
        │ │   i += 1                                     ││
        │ │                                              ││
        │ │ YOU CAN ENHANCE:                             ││
        │ │ - Add TaskRevisionHelperAgent()              ││
        │ │ - Try multiple agents if first fails         ││
        │ │ - Combine responses from multiple agents     ││
        │ │ - Add validation logic                       ││
        │ │ - Custom fallback strategies                 ││
        │ │ - Dynamic task scheduling                    ││
        │ │                                              ││
        │ │ MUST RESPECT:                                ││
        │ │ - Keep max_loops = 15 safety cap             ││
        │ │ - Keep memory persistence                    ││
        │ │ - Keep history generation                    ││
        │ └──────────────────────────────────────────────┘│
        │                                                  │
        │ For each task i (max 15 iterations):            │
        │                                                  │
        │ ┌ Step 1: Build input with context            │
        │ │ user_input = _build_input(task, i)          │
        │ │ DO NOT EDIT (_build_input)                  │
        │ └ Returns string with context                  │
        │                                                  │
        │ ┌ Step 2: Execute task ⭐ EDITABLE            │
        │ │ response = assigned_agent.execute_task(...) │
        │ │ OPTIONAL: Use TaskRevisionHelperAgent       │
        │ │ OPTIONAL: Try fallback agents               │
        │ │ OPTIONAL: Combine multiple responses        │
        │ └ Returns text                                 │
        │                                                  │
        │ ┌ Step 3: Clean response                       │
        │ │ response.replace("Final Answer:","")         │
        │ │ response.strip()                             │
        │ └ Store in memory                              │
        │                                                  │
        │ ┌ Step 4: Move to next task                    │
        │ │ i += 1                                       │
        │ │ (or conditional advance if enhanced)        │
        │ └                                               │
        │                                                  │
        └──────────────────────┬──────────────────────────┘
                               │
        ┌──────────────────────┴─────────────────────────┐
        │ (Optional) TaskRevisionHelperAgent ⭐ EDITABLE  │
        │ (track2_execution.py:31-81)                    │
        │                                                 │
        │ Editable Section 1:                            │
        │ ┌───────────────────────────────────────────┐  │
        │ │ execute_task(task_input) method           │  │
        │ │                                            │  │
        │ │ CURRENT: Raises NotImplementedError       │  │
        │ │                                            │  │
        │ │ YOU MUST IMPLEMENT:                       │  │
        │ │ - Validate task clarity                   │  │
        │ │ - Check for required parameters           │  │
        │ │ - Enrich with domain context              │  │
        │ │ - Score task quality                      │  │
        │ │ - Return revised/validated task           │  │
        │ │                                            │  │
        │ │ USE CASES:                                │  │
        │ │ - Pre-process input before agents        │  │
        │ │ - Validate agent responses                │  │
        │ │ - Enrich context                          │  │
        │ │ - Add metadata                            │  │
        │ │                                            │  │
        │ │ EXAMPLE IMPLEMENTATION:                   │  │
        │ │ def execute_task(self, task_input):       │  │
        │ │   # Check clarity                         │  │
        │ │   if len(task_input.split()) < 5:         │  │
        │ │     issues.append("Too brief")           │  │
        │ │   # Enrich                                │  │
        │ │   enriched = task_input + context_hints  │  │
        │ │   return enriched                         │  │
        │ └───────────────────────────────────────────┘  │
        │                                                 │
        └────────────────┬────────────────────────────────┘
                         │
                         │ (if used)
                         ▼
        ┌──────────────────────────────────────────────────┐
        │ Agent.execute_task(processed_input)             │
        │ (agent_hive/agents/react_reflect_agent.py)      │
        │                                                  │
        │ Same as Track 1 execution                       │
        │ DO NOT EDIT (read-only)                        │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ Generate execution history                      │
        │ (track2_execution.py:197-211)                   │
        │                                                  │
        │ Returns:                                        │
        │ [                                               │
        │   {                                             │
        │     "task_number": 1,                          │
        │     "task_description": "...",                 │
        │     "agent_names": ["Agent1", "Agent2"],       │
        │     "response": "..."                          │
        │   },                                            │
        │   ...                                           │
        │ ]                                               │
        │                                                  │
        │ DO NOT EDIT (read-only)                        │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ Output saved to:                                │
        │ /home/track1_result/trajectory/                │
        │ Q_<utterance_id>_trajectory.json               │
        │                                                  │
        │ Same format as Track 1                          │
        └──────────────────────────────────────────────────┘
```

---

## 🔄 How Edits Propagate Through System

### Track 1 Edit Impact:

```
Your edit to get_prompt()
    ↓
LLM receives different instruction
    ↓
LLM generates different decomposition
    ↓
Different #Task, #Agent order
    ↓
Regex extracts different planned_tasks
    ↓
SequentialWorkflow.run() executes different plan
    ↓
Different agent calls (different order/combination)
    ↓
Different trajectory output
```

### Track 2 Edit Impact:

```
Your edit to execute_task() in TaskRevisionHelperAgent
    ↓
Input validation/enrichment before agent
    ↓
Agent receives better-formatted input
    ↓
Better reasoning by LLM
    ↓
Better response quality
    ↓
Better stored in memory
    ↓
Better context for next task

Your edit to run() loop
    ↓
Different task scheduling/fallback logic
    ↓
Different response processing
    ↓
Different memory accumulation
    ↓
Different trajectory flow
```

---

## 📌 Key Data Structures

### Task Object:
```python
from agent_hive.task import Task

task = Task(
    description: str,              # What to do
    agents: List[BaseAgent],       # Who does it
    expected_output: str,          # What we expect
    context: List[Task] = None     # Depends on these tasks
)
```

### Agent Object:
```python
from agent_hive.agents.react_reflect_agent import ReactReflectAgent

agent = ReactReflectAgent(
    name: str,                     # "IoT Data Download"
    description: str,              # "Can retrieve sensor data..."
    tools: List[BaseTool],         # Available tools
    llm: str,                       # Model ID (6-19)
    few_shots: str,                # Examples to guide reasoning
    task_examples: List[str] = []  # What it can solve
)
```

### ContextType Enum:
```python
from agent_hive.enum import ContextType

ContextType.DISABLED    # Each task independent
ContextType.ALL         # Task sees all previous outputs
ContextType.PREVIOUS    # Task sees only previous task output
ContextType.SELECTED    # Task sees specified task outputs (via task.context)
```

---

## 🎯 Summary: What Flows Where

```
TRACK 1 Data Flow:

User Question
    ↓ (in agents list, task examples)
get_prompt() ⭐ EDITABLE SECTION 2
    ↓ (formatted prompt)
watsonx_llm()
    ↓ (multi-step plan)
regex parsing (extractors)
    ↓ (Task objects)
SequentialWorkflow.run()
    ↓ (task by task)
ReactReflectAgent.execute_task() (×N agents)
    ↓ (responses)
generate_history()
    ↓ (trajectory JSON)
Saved to file

TRACK 2 Data Flow:

User Question
    ↓ (as task input)
DynamicWorkflow.run() ⭐ EDITABLE SECTION 2
    ↓ (optional)
TaskRevisionHelperAgent.execute_task() ⭐ EDITABLE SECTION 1
    ↓ (enriched input)
ReactReflectAgent.execute_task() (×N agents)
    ↓ (optional fallback)
ReactReflectAgent.execute_task() (alternative agent)
    ↓ (responses)
generate_history()
    ↓ (trajectory JSON)
Saved to file
```

---

## 🔐 Immutable Points (DO NOT TOUCH)

These are read-only and protected:

1. **base_agent.py** - Abstract base
2. **base_workflow.py** - Workflow base class
3. **sequential.py** - Sequential executor logic
4. **react_reflect_agent.py** - Agent implementation
5. **task.py** - Task data model
6. **enum.py** - ContextType enum
7. **All tool files** - fmsr.py, skyspark.py, tsfm.py, wo.py
8. **run_track_1.py**, **run_track_2.py** - Entry points

---

**Ready to edit? Start with TRACK 1 SECTION 2 (get_prompt) - it has the biggest impact! 🚀**
