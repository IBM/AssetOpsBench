# Ground Truth Creation Best Practice 

## Overview

This document provides best practices for creating ground truth for an utterance. It can be viewed as a trajectory of the execution process of the utterance under an ideal execution process without hallucination or execution failure.   The ground truth is the cornerstone to evaluate AI agent systems irresoective it domain or application. Ground truth represents the **ideal trajectory** - the expected sequence of planning decisions and execution steps that a system should follow to correctly answer a user's natural language query, i.e. utterance.

### What is Ground Truth?

Ground truth is a structured specification that defines:
1. **What** the AI agent system should accomplish (expected output)
2. **How** the AI agent system should accomplish it (execution trajectory)
3. **Why** each step is necessary (planning rationale)

It serves as the **soft ground truth** for automated evaluation, enabling systematic comparison of actual agent behavior against expert-defined expected behavior.  "Soft" means the framework accommodates both types, providing exact answers for deterministic tasks and acceptance criteria for non-deterministic tasks. For example, in a deterministic task like "How many work orders in 2017?", the ground truth would specify the exact number of work orders. In a non-deterministic task like "Recommend maintenance actions", the ground truth would specify the criteria for a valid recommendation, such as "The recommended actions should address the top 3 failure modes with the highest risk scores."

### Purpose

Enable rigorous, reproducible benchmarking of AI agent systems by providing:
- Objective evaluation criteria
- Automated scoring mechanisms
- Failure mode discovery
- Architectural comparison
- Model selection guidance

### Use Cases

While this document uses industrial asset management examples (from AssetOpsBench), the principles apply to any domain requiring agent-based automation, including energy systems, healthcare, finance, and customer service.

### Two System Paradigms

Ground truth supports evaluation of two complementary paradigms:

**1. Multi-Agent Systems**
- Multiple specialized AI agents collaborating to solve tasks
  - Each agent has domain expertise (e.g., IoTAgent for sensor data, WOAgent for work orders)
  - Agents communicate and coordinate through orchestration
- Example: AssetOpsBench with 4 specialized agents (IoT, FMSR, TSFM, WO)

**2. Multi-MCP Systems**
- Multiple Model Context Protocol (MCP) servers providing standardized tools
- MCP servers expose capabilities (e.g., filesystem, database, web-search)
- LLM orchestrates tool usage to accomplish tasks
- Example: MCP-Bench with filesystem, database, API servers

**Key Insight**: Both paradigms share the **same ground truth structure**. The difference lies in semantic interpretation:
- **Multi-Agent**: `type` = agent type (e.g., "IoTAgent"), `agent` = agent name
- **Multi-MCP**: `type` = MCP server type (e.g., "filesystem"), `agent` = server name

### Benchmarking Context

Ground truth enables three critical benchmarking capabilities:

**1. Architectural Comparison**
- Compare different orchestration strategies (Agent-As-Tool vs Plan-Execute)
- Evaluate trade-offs between reasoning approaches
- Example: AssetOpsBench found Agent-As-Tool achieves 60-70% completion vs 46% for Plan-Execute

**2. Model Selection**
- Objectively compare LLM performance across scenarios
- Identify model-specific strengths and weaknesses
- Example: AssetOpsBench evaluates models across multi-agent coordination, tool selection, and domain-specific reasoning tasks

**3. Failure Mode Discovery**
- Automatically identify common failure patterns
- Diagnose reasoning and recovery bottlenecks
- Enable systematic debugging and improvement

### Community and Reproducibility

Ground truth serves as shared infrastructure for open source and research communities:
- **Standardized Evaluation**: All systems measured against same baseline
- **Fair Comparison**: Eliminates subjective judgment
- **Progress Tracking**: Quantify improvements over time
- **Collaboration**: Shared benchmarks accelerate development and research

### Document Scope

This document provides systematic workflow, structure definitions, best practices, and validation guidance for creating ground truth trajectories. Based on AssetOpsBench and industry best practices for agent evaluation.

---

## 1. High-Level Ground Truth Creation Workflow

This section provides a systematic, step-by-step process for creating ground truth trajectories. Follow this workflow to ensure completeness and consistency.

### 1.1 Overview: The Five-Phase Process

```
Phase 1: Understand the Utterance
    ↓
Phase 2: Define Expected Behavior
    ↓
Phase 3: Design the Ideal Trajectory
    ↓
Phase 4: Validate and Refine
    ↓
Phase 5: Document and Review
```

### 1.2 Phase 1: Understand the Utterance

**Objective**: Fully comprehend what the user is asking and what constitutes success.

**Key Questions**:
1. What is the user's intent?
2. What information or action does the user need?
3. Which agents/servers are required?
4. What data sources must be accessed?
5. What constraints or conditions apply?

**Actions**:
- [ ] Read the utterance carefully
- [ ] Identify key entities (equipment, time ranges, metrics)
- [ ] Determine the problem category (RETROSPECTIVE, PREDICTIVE, PRESCRIPTIVE)
- [ ] Classify the utterance type (Knowledge Query, Data Query, Inference Query, etc.)
- [ ] Identify the primary agent/server type

**Example**:
```
Utterance: "Get the work order of equipment CWC04013 for year 2017."

Analysis:
- Intent: Retrieve historical work order records
- Information needed: Work orders for specific equipment and time period
- Required agent: WOAgent (Work Order Agent)
- Data source: Work order database
- Constraints: Equipment ID = CWC04013, Year = 2017
- Group: retrospective (historical data retrieval)
- Category: Knowledge Query
- Type: Workorder
- Entity: Equipment
- Deterministic: true (33 records expected)
```

### 1.3 Phase 2: Define Expected Behavior

**Objective**: Specify what the correct output should be and how to validate it.

**Key Decisions**:

**Decision 1: Is this deterministic or non-deterministic?**

Ask: "Is there exactly one correct answer?"

- **Deterministic (true)**:
  - Data retrieval with specific parameters
  - Factual queries with verifiable answers
  - Calculations with exact results
  - Example: "How many work orders in 2017?" → Exact count

- **Non-deterministic (false)**:
  - Recommendations with multiple valid options
  - Predictions with probabilistic outputs
  - Analysis with subjective interpretations
  - Example: "What maintenance actions are recommended?" → Multiple valid suggestions

**Decision 2: What is the characteristic form?**

For **deterministic** utterances:
```
"There will be 33 records. The expected response should retrieve
all work orders for equipment CWC04013 in the year 2017, ensuring
correct equipment ID and time filtering."
```

For **non-deterministic** utterances:
```
"The answer should contain one or more valid work order
recommendations with rationale. The recommendations should be
relevant to addressing anomalies in equipment CWC04009."
```

**Decision 3: What is the expected output structure?**

Define `final_out`:
- For deterministic: Exact expected values
- For non-deterministic: Representative example

**Actions**:
- [ ] Set `deterministic` flag (true/false)
- [ ] Write clear `characteristic_form` description
- [ ] Define `final_out` structure
- [ ] Write `final_out_description` explaining the output

### 1.4 Phase 3: Design the Ideal Trajectory

**Objective**: Map out the step-by-step process the agent should follow.

#### Step 3A: High-Level Planning

**Create `planning_steps`**: Strategic decomposition of the problem

**Template**:
```json
"planning_steps": [
  {
    "agent": "AgentName",
    "instruction": "What this agent should accomplish (high-level)"
  }
]
```

**Guidelines**:
- Use semantic, intent-focused descriptions
- Each step should be a meaningful sub-goal
- Order matters - represent logical sequence
- Use predefined agent/server names

**Example**:
```json
"planning_steps": [
  {
    "agent": "WOAgent",
    "instruction": "Create Equipment instance for CWC04013"
  },
  {
    "agent": "WOAgent",
    "instruction": "Create DateRange for year 2017"
  },
  {
    "agent": "WOAgent",
    "instruction": "Retrieve work orders for equipment and date range"
  }
]
```

#### Step 3B: Detailed Execution Design

**Create `execution_steps`**: Specific tool calls and operations

**For each step, define**:
1. **name**: Unique identifier (e.g., "create_equipment", "get_work_orders")
2. **action**: Tool/function name (e.g., "Equipment", "get_work_orders")
3. **agent**: Which agent/server executes this
4. **arguments**: Input parameters (can reference previous outputs)
5. **outputs**: Variable names produced
6. **deterministic**: Flags for name, action, arguments, outputs

**Template**:
```json
{
  "name": "descriptive_step_name",
  "action": "tool_or_function_name",
  "agent": "AgentName",
  "arguments": {
    "param1": "value1",
    "param2": "reference_to_previous_output"
  },
  "outputs": ["output_variable_name"],
  "deterministic": {
    "name": false,
    "action": true,
    "arguments": true,
    "outputs": true
  }
}
```

**Key Considerations**:
- **Data Flow**: Outputs from one step become inputs to later steps
- **Dependencies**: Identify which steps depend on others
- **Intermediate Processing**: Include load, parse, filter, transform steps
- **Special Steps**: Always end with "Finish" step for final output

#### Step 3C: Define Execution Links

**Create `execution_links`**: Connect steps to form DAG

**Template**:
```json
"execution_links": [
  {"source": "step1_name", "target": "step2_name"},
  {"source": "step2_name", "target": "step3_name"}
]
```

**Rules**:
- Every step (except first) must have incoming link
- Every step (except last) must have outgoing link
- No cycles allowed (must be a DAG)
- Multiple sources can target same step (parallel → sequential)
- One source can target multiple steps (branching)

**Validation**:
- [ ] All step names in links exist in execution_steps
- [ ] No orphaned steps
- [ ] No cycles (use topological sort to verify)
- [ ] Data dependencies correctly represented

### 1.5 Phase 4: Validate and Refine

**Objective**: Ensure ground truth is correct, complete, and consistent.

#### Validation Checklist

**Structural Validation**:
- [ ] All required fields present (id, text, type, category, deterministic, characteristic_form)
- [ ] planning_steps is list of dicts with "agent" and "instruction"
- [ ] execution_steps has all required fields for each step
- [ ] execution_links properly connects all steps
- [ ] final_out and final_out_description present

**Semantic Validation**:
- [ ] Utterance text is clear and unambiguous
- [ ] Category correctly classifies the utterance
- [ ] Deterministic flag matches answer uniqueness
- [ ] characteristic_form clearly describes expected output
- [ ] Planning steps use predefined agent names
- [ ] Execution steps use valid tool/function names
- [ ] All argument references point to existing outputs

**Logical Validation**:
- [ ] Planning steps form logical problem-solving sequence
- [ ] Execution steps accomplish planning goals
- [ ] Data flows correctly through execution steps
- [ ] DAG structure is valid (no cycles, no orphans)
- [ ] Final output matches characteristic_form

**Domain Validation**:
- [ ] Ground truth reflects actual operational workflow
- [ ] Tool usage is realistic and efficient
- [ ] Parameters use correct domain terminology
- [ ] Expected output is achievable with available tools

#### Refinement Process

1. **Test with Domain Expert**: Walk through trajectory with SME
2. **Simulate Execution**: Mentally execute each step to verify feasibility
3. **Check Edge Cases**: Consider what could go wrong
4. **Optimize Path**: Remove unnecessary steps, combine where possible
5. **Update Documentation**: Ensure notes explain any non-obvious decisions

### 1.6 Phase 5: Document and Review

**Objective**: Finalize documentation and prepare for use.

#### Documentation Requirements

**Required Documentation**:
1. **characteristic_form**: Clear, detailed description of expected output
2. **final_out_description**: Explanation of what the output represents
3. **note** field: Context, rationale, and metadata

**Recommended Note Content**:
```json
"note": "Source: [where this scenario came from];
         Owner: [SME who created/validated it];
         Context: [important background information];
         Design rationale: [why this approach was chosen];
         Implementation notes: [technical details];
         Omitted: [what was intentionally left out]"
```

#### Final Review

**Review with Stakeholders**:
- [ ] Domain expert validates correctness
- [ ] System architect validates technical feasibility
- [ ] End user validates practical utility
- [ ] Peer reviewer validates consistency with other scenarios

**Quality Gates**:
- [ ] Passes all validation checks
- [ ] Achieves >90% confidence from domain expert
- [ ] Aligns with existing ground truth patterns
- [ ] Documentation is complete and clear

### 1.7 Quick Reference: Creation Process Flow

```
START: New Utterance
    ↓
Q1: What is user's intent?
    → Identify: RETROSPECTIVE, PREDICTIVE, or PRESCRIPTIVE
    ↓
Q2: Is there exactly one correct answer?
    → YES: deterministic = true
    → NO: deterministic = false
    ↓
Q3: Which agents/servers are needed?
    → Single: type = "AgentName"
    → Multiple: type = "multiagent"
    ↓
Q4: What are the high-level steps?
    → Create planning_steps (strategic intent)
    ↓
Q5: What are the detailed operations?
    → Create execution_steps (tool calls)
    ↓
Q6: How do steps connect?
    → Create execution_links (DAG)
    ↓
Q7: What is the expected output?
    → Define final_out and characteristic_form
    ↓
Q8: Does it pass validation?
    → YES: Document and finalize
    → NO: Return to appropriate phase and refine
    ↓
DONE: Ground Truth Complete
```

### 1.8 Common Pitfalls and How to Avoid Them

**Pitfall 1: Vague characteristic_form**
- Bad: "The response should return the data."
- Good: "There will be 33 records. The response should retrieve all work orders for equipment CWC04013 in year 2017."

**Pitfall 2: Missing intermediate steps**
- Bad: get_data → finish
- Good: get_data → load_data → parse_data → process_data → finish

**Pitfall 3: Incorrect deterministic flag**
- Bad: Recommendation query marked as deterministic
- Good: Recommendation query marked as non-deterministic

**Pitfall 4: Circular dependencies**
- Bad: A → B → C → A (cycle)
- Good: A → B → C → D (DAG)

**Pitfall 5: Ambiguous variable references**
- Bad: arguments: {"data": "result"}
- Good: arguments: {"data": "sensor_readings"} (clear variable name)

### 1.9 Effort Considerations

Ground truth creation effort varies significantly based on scenario complexity:

**Factors Affecting Creation Time**:
- **Scenario Complexity**: Number of execution steps and dependencies
- **Domain Familiarity**: Understanding of tools and workflows
- **Determinism**: Non-deterministic scenarios require more careful specification
- **Validation Requirements**: Domain expert review adds time
- **Documentation Quality**: Clear characteristic_form and notes take effort

**Observed Patterns from AssetOpsBench**:
- **Simple scenarios** (e.g., Scenario 3: 2 steps, linear): Faster to create
- **Complex scenarios** (e.g., Scenario 421: 9 steps, parallel branches): More time-intensive
- **Dataset scale**: 141 scenarios created for AssetOpsBench benchmark

**Efficiency Strategies**:
- Use templates for common patterns (e.g., Equipment + DateRange initialization)
- Reuse execution step structures from similar scenarios
- Batch similar scenarios together (e.g., all work order queries)
- Involve domain experts early to avoid rework
- Start with deterministic scenarios before tackling non-deterministic ones

**Note**: Specific time estimates depend on team expertise, tooling, and domain complexity. Track your own metrics to establish baselines for your context.

---

## 2. Ground Truth Structure

### 2.1 Core Components

Each ground truth scenario consists of the following key components:

```json
{
  "id": <integer>,
  "uuid": "<optional-unique-identifier>",
  "text": "<natural language utterance>",
  "type": "<agent or MCP server type>",
  "category": "<utterance category>",
  "deterministic": <boolean>,
  "characteristic_form": "<expected response description>",
  "planning_steps": [...],
  "execution_steps": [...],
  "execution_links": [...],
  "final_out": {...},
  "final_out_description": [...]
}
```

### 2.2 Field Definitions

#### Required Fields

1. **id** (integer): Unique identifier for the scenario
   - Must be unique across the entire ground truth dataset
   - Used for tracking and referencing scenarios

2. **text** (string): The natural language utterance/query
   - Should be clear, unambiguous, and representative of real user queries
   - Must align with the utterance design guidelines

3. **type** (string): Primary agent or MCP server responsible
   - Examples: `IoT`, `FMSA`, `TSFM`, `Workorder`, `multiagent`
   - In MCP context: `filesystem`, `database`, `web-search`, etc.
   - Use `multiagent` when multiple agents must collaborate

4. **category** (string): Classification of the utterance
   - Examples: `Knowledge Query`, `Data Query`, `Inference Query`, `Decision Support`, `Prediction`, `Complex Query`
   - Should align with the 8 universal categories from utterance design guideline

5. **deterministic** (boolean): Whether the utterance has a single correct answer
   - `true`: Single verifiable correct answer (e.g., data retrieval, factual queries)
   - `false`: Multiple valid responses possible (e.g., recommendations, predictions)

6. **characteristic_form** (string): Description of expected response format and validation criteria
   - For deterministic: Specify exact expected output or validation rules
   - For non-deterministic: Describe acceptable response characteristics
   - Include data format, file references, or structural requirements

#### Planning Steps

7. **planning_steps** (array of objects): High-level agent-level planning
   - Each step is a dictionary with keys: `agent` and `instruction`
   - Represents the strategic decomposition of the problem
   - Must use predefined agent names from the available agent list
   - Order matters - represents logical sequence of problem-solving

**Structure**:
```json
"planning_steps": [
  {
    "agent": "AgentName",
    "instruction": "High-level description of what this agent should do"
  }
]
```

**Best Practices**:
- Use semantic descriptions that capture intent, not implementation details
- Each step should represent a meaningful sub-goal
- Comparing plans requires semantic matching (same goal, different wording is acceptable)

#### Execution Steps

8. **execution_steps** (array of objects): Detailed tool-level operations
   - Each step represents a specific tool/function call
   - Forms nodes in the execution DAG

**Structure**:
```json
"execution_steps": [
  {
    "name": "unique_step_identifier",
    "action": "tool_or_function_name",
    "agent": "AgentName",
    "arguments": {
      "param1": "value1",
      "param2": "reference_to_previous_output"
    },
    "outputs": ["output_variable_name"],
    "deterministic": {
      "name": false,
      "action": true/false,
      "arguments": true/false,
      "outputs": true/false
    }
  }
]
```

**Field Details**:
- **name**: Unique identifier for this step (used in execution_links)
  - Should be human-readable and descriptive
  - Examples: `step1`, `create_equipment`, `get_sensor_data`, `filter_anomalies`
  
- **action**: Name of the tool, function, or operation
  - Examples: `sensors`, `history`, `get_work_orders`, `pickle.load`, `Finish`
  - Special actions: `Finish`, `Self-Ask`, `Agent-Ask`
  
- **agent**: Which agent executes this step
  
- **arguments**: Dictionary of parameters passed to the action
  - Can reference outputs from previous steps using variable names
  - Use `@variable_name` or just `variable_name` for references
  
- **outputs**: List of variable names produced by this step
  - These can be referenced by subsequent steps
  - Empty list `[]` if no output
  
- **deterministic**: Dictionary indicating determinism at different levels
  - `name`: Is the step name deterministic? (usually false - can vary)
  - `action`: Is the action/tool deterministic? (true for data retrieval, false for LLM calls)
  - `arguments`: Are the arguments deterministic?
  - `outputs`: Are the outputs deterministic?

#### Execution Links

9. **execution_links** (array of objects): Defines the DAG structure
   - Connects execution steps to form a directed acyclic graph
   - Represents dependencies and execution order

**Structure**:
```json
"execution_links": [
  {
    "source": "step_name_1",
    "target": "step_name_2"
  }
]
```

**Best Practices**:
- Every step (except the first) should have at least one incoming link
- Every step (except the last) should have at least one outgoing link
- Must form a valid DAG (no cycles)
- Multiple sources can target the same step (parallel execution converging)
- One source can target multiple steps (branching execution)

#### Final Output

10. **final_out** (object): Structured representation of the expected final result
    - JSON object containing the actual expected output
    - Can be null if not applicable
    - Should match the format described in `characteristic_form`

11. **final_out_description** (array of strings): Textual description of the final output
    - Human-readable explanation of what the output represents
    - Can include validation criteria, interpretation notes, or context

#### Optional Fields

12. **uuid** (string): Globally unique identifier (optional)
    - Use when scenarios need to be tracked across systems
    - Format: UUID v4 (e.g., "efc94d35-5236-410c-9e4f-5dcdfee818cc")

13. **expected_result** (any): Legacy field for expected results (optional)
    - Can be null or omitted
    - Prefer using `final_out` instead

14. **data** (object): Additional metadata or context (optional)
    - Can store scenario-specific data
    - Usually empty `{}`

15. **possible_alternatives** (object): Alternative valid execution paths (optional)
    - Documents other acceptable ways to solve the problem
    - Useful for non-deterministic scenarios

---

## 3. Special Execution Steps

### 3.1 Finish Step

The `Finish` step is a special terminal step that marks task completion and provides the final answer.

**Characteristics**:
- **Action**: Always `"Finish"`
- **Purpose**: Signals task completion and returns final result
- **Placement**: Typically the last step in the execution DAG
- **Output**: The final answer that will be presented to the user

**Usage Patterns**:

**Pattern 1: Direct Finish (IoT Agent)**
```json
{
  "name": "finish",
  "action": "Finish",
  "agent": "IoTAgent",
  "argument": "The assets at the MAIN site are: CQPA AHU 1, CQPA AHU 2B, Chiller 4, Chiller 6, Chiller 9, Chiller 3.",
  "deterministic": {
    "name": false,
    "action": true,
    "arguments": true,
    "outputs": true
  }
}
```

**Pattern 2: Finish with Structured Arguments (TSFM Agent)**
```json
{
  "name": "return_final_answer",
  "action": "Finish",
  "agent": "TSFMAgent",
  "arguments": {
    "answer": true,
    "evidence_models": ["ttm_96_28", "ttm_512_96"]
  },
  "outputs": ["final_answer"],
  "deterministic": {
    "name": false,
    "action": true,
    "arguments": true,
    "outputs": true
  }
}
```

**Pattern 3: Finish without Arguments (FMSA Agent)**
```json
{
  "name": "finish",
  "action": "Finish",
  "arguments": "",
  "outputs": ""
}
```

**Best Practices**:
- FMSA and TSFM agents typically use `Finish` as an anchor/terminal step
- IoT agent may or may not use explicit `Finish` step (inconsistency noted in original data)
- The `Finish` step's output is used as the "final answer" for evaluation
- For deterministic queries, the finish argument should contain the exact expected answer
- For non-deterministic queries, the finish argument should contain representative valid output

### 3.2 Other Special Steps

**Self-Ask**: Agent asks itself a clarifying question
- Used when agent needs to decompose complex queries
- Creates internal reasoning loops

**Agent-Ask**: Agent requests information from another agent
- Used in multi-agent scenarios
- Represents inter-agent communication

---

## 4. Best Practices

### 4.1 Consistency and Standards

1. **Agent Naming**
   - Use predefined agent names consistently
   - Common agents: `IoTAgent`, `FMSRAgent`, `TSFMAgent`, `WOAgent`
   - In MCP context: Use standard MCP server names

2. **Action Naming**
   - Use consistent tool/function names across scenarios
   - Document available tools and their signatures
   - Examples: `sensors`, `history`, `get_work_orders`, `Equipment`, `DateRange`

3. **Variable Naming**
   - Use descriptive variable names for outputs
   - Examples: `equipment`, `date_range`, `events`, `sensor_data`, `forecast_results`
   - Avoid generic names like `result`, `output`, `data` unless context is clear

4. **Step Naming**
   - Use descriptive step names that indicate purpose
   - Good: `create_equipment`, `filter_anomalies`, `calculate_statistics`
   - Avoid: `step1`, `step2`, `step3` (unless no better name exists)

### 4.2 Determinism Marking

**When to mark as deterministic**:

- **Action deterministic**: Tool always produces same output for same input
  - Data retrieval: `true`
  - Mathematical calculations: `true`
  - LLM calls: `false`
  - Model inference: `false`

- **Arguments deterministic**: Input parameters are fixed
  - Hardcoded values: `true`
  - References to deterministic outputs: `true`
  - User input or dynamic values: `false`

- **Outputs deterministic**: Output is predictable
  - Fixed data retrieval: `true`
  - Calculations: `true`
  - LLM-generated content: `false`
  - Model predictions: `false`

**Example**:
```json
{
  "name": "load_work_orders",
  "action": "pickle.load",
  "deterministic": {
    "name": false,        // Step name can vary
    "action": false,      // pickle.load is not a standard action
    "arguments": false,   // File path may vary
    "outputs": false      // Content depends on file
  }
}
```

### 4.3 Multi-Agent Scenarios

For complex queries requiring multiple agents:

1. **Set type to "multiagent"**
   ```json
   "type": "multiagent"
   ```

2. **Include all agents in planning_steps**
   - Show agent handoffs explicitly
   - Document inter-agent communication

3. **Use Agent-Ask for coordination**
   - When one agent needs information from another
   - Document the request-response pattern

4. **Consider execution order**
   - Some agents may need to complete before others start
   - Use execution_links to enforce ordering

**Example**:
```json
"planning_steps": [
  {
    "agent": "IoTAgent",
    "instruction": "Retrieve sensor data for the equipment"
  },
  {
    "agent": "TSFMAgent",
    "instruction": "Analyze sensor data for anomalies"
  },
  {
    "agent": "WOAgent",
    "instruction": "Recommend work orders based on detected anomalies"
  }
]
```

### 4.4 Handling Non-Deterministic Scenarios

For utterances with multiple valid answers:

1. **Set deterministic to false**
   ```json
   "deterministic": false
   ```

2. **Provide representative output**
   - Show one valid example in `final_out`
   - Note that other valid outputs exist

3. **Define acceptance criteria**
   - In `characteristic_form`, describe what makes an answer valid
   - List required elements
   - Specify constraints

4. **Document alternatives**
   - Use `possible_alternatives` to show other valid approaches
   - Helps evaluators understand acceptable variations

**Example**:
```json
"characteristic_form": "The answer should contain one or more valid work order recommendations with rationale. The recommendations should be relevant to addressing anomalies in equipment CWC04009.",
"deterministic": false,
"final_out": {
  "recommendations": ["M005", "M006", "OP004"]
},
"final_out_description": [
  "Recommended work orders based on anomaly analysis",
  "Other valid recommendations may exist depending on prioritization criteria"
]
```

### 4.5 Cross-Validation Checks

Before finalizing ground truth, validate:

1. **Planning Steps Validation**
   - All agents are from predefined list
   - Instructions are clear and semantic
   - Steps form logical sequence

2. **Execution Steps Validation**
   - All step names are unique
   - All actions are valid tool/function names
   - All argument references point to existing outputs
   - Deterministic flags are correctly set

3. **Execution Links Validation**
   - All source and target names exist in execution_steps
   - No cycles in the DAG
   - All steps (except first) have incoming links
   - All steps (except last) have outgoing links

4. **Output Validation**
   - `final_out` matches `characteristic_form` description
   - `final_out_description` is clear and complete
   - For deterministic queries, exact expected output is specified

5. **Consistency Validation**
   - `type` matches primary agent in planning/execution
   - `category` aligns with utterance characteristics
   - `deterministic` flag matches output predictability

---


### 4.6 Quality Assurance Checklist

Before submitting ground truth, verify:

#### Structural Completeness
- [ ] All required fields present (id, text, type, category, deterministic, characteristic_form)
- [ ] planning_steps is a list of dicts with "agent" and "instruction" keys
- [ ] execution_steps is a list with all required fields
- [ ] execution_links properly connects all steps
- [ ] final_out and final_out_description are present (if applicable)

#### Semantic Correctness
- [ ] Utterance text is clear and unambiguous
- [ ] Category correctly classifies the utterance type
- [ ] Deterministic flag accurately reflects answer uniqueness
- [ ] characteristic_form clearly describes expected output

#### Planning Quality
- [ ] Planning steps use predefined agent names
- [ ] Instructions are semantic and intent-focused
- [ ] Steps form logical problem-solving sequence
- [ ] All necessary agents are included

#### Execution Quality
- [ ] All step names are unique
- [ ] All actions are valid tool/function names
- [ ] All argument references point to existing outputs
- [ ] Deterministic flags correctly set at all levels
- [ ] Output variables are properly named and used

#### DAG Validity
- [ ] No cycles in execution_links
- [ ] All steps (except first) have incoming links
- [ ] All steps (except last) have outgoing links
- [ ] No orphaned steps
- [ ] Dependencies correctly represent data flow

#### Output Quality
- [ ] final_out matches characteristic_form description
- [ ] For deterministic queries, exact expected output specified
- [ ] For non-deterministic queries, representative example provided
- [ ] final_out_description is clear and complete

#### Consistency
- [ ] Type matches primary agent in execution
- [ ] Category aligns with utterance characteristics
- [ ] Planning and execution steps are aligned
- [ ] Terminology is consistent across fields

---


### 4.7 Common Pitfalls and How to Avoid Them

#### 8.1 Inconsistent Finish Steps

**Problem**: Different agents use different Finish patterns
- IoT agent sometimes omits Finish
- FMSA/TSFM always include Finish

**Solution**: Standardize Finish usage
- Always include explicit Finish step
- Use consistent argument structure
- Document agent-specific conventions

#### 8.2 Ambiguous Variable References

**Problem**: Unclear how outputs are referenced in arguments

**Solution**: Use consistent reference syntax
- Document whether to use `@variable` or just `variable`
- Be explicit about variable scope
- Validate all references point to existing outputs

#### 8.3 Incomplete Determinism Marking

**Problem**: Deterministic flags not set at all levels

**Solution**: Always include complete deterministic dict
```json
"deterministic": {
  "name": false,
  "action": true/false,
  "arguments": true/false,
  "outputs": true/false
}
```

#### 8.4 Circular Dependencies

**Problem**: Execution links create cycles

**Solution**: Validate DAG structure
- Use topological sort to detect cycles
- Ensure data flows in one direction
- Break cycles by reordering steps

#### 8.5 Missing Intermediate Steps

**Problem**: Jumping from data retrieval to final output without processing

**Solution**: Include all necessary transformations
- Add load/parse steps for file-based data
- Include filter/transform steps for data processing
- Show LLM calls explicitly when used

#### 8.6 Vague Characteristic Forms

**Problem**: Expected output not clearly described

**Solution**: Be specific and detailed
- For deterministic: Specify exact expected values or validation rules
- For non-deterministic: List required elements and constraints
- Include format specifications (JSON, CSV, text, etc.)

---


### 4.8 Advanced Scenarios

For complex ground truth scenarios:

#### Conditional Execution

For scenarios with conditional logic:

```json
{
  "name": "conditional_step",
  "action": "if_then_else",
  "agent": "Agent",
  "arguments": {
    "condition": "anomaly_detected == true",
    "then_action": "recommend_maintenance",
    "else_action": "continue_monitoring"
  },
  "outputs": ["action_taken"],
  "deterministic": {
    "name": false,
    "action": false,
    "arguments": false,
    "outputs": false
  }
}
```


#### Parallel Execution Paths

For scenarios with parallel processing:

```json
"execution_links": [
  {"source": "get_data", "target": "analyze_path1"},
  {"source": "get_data", "target": "analyze_path2"},
  {"source": "analyze_path1", "target": "merge_results"},
  {"source": "analyze_path2", "target": "merge_results"}
]
```


---

## 5. Common Patterns and Templates

This section presents real-world patterns from AssetOpsBench ground truth scenarios, demonstrating practical implementations used in industrial asset management.

### 5.1 Simple Knowledge Query Pattern (ID 103)

**Use Case**: Single-agent query for domain knowledge

**Scenario**: "List all failure modes of asset Wind Turbine."

**Pattern Characteristics**:
- Single agent (FMSRAgent)
- Direct knowledge retrieval
- Non-deterministic outputs (knowledge-based)
- Simple linear execution

```json
{
  "planning_steps": [
    {
      "agent": "FMSRAgent",
      "instruction": "List all failure modes of asset Wind Turbine."
    }
  ],
  "execution_steps": [
    {
      "name": "get_failure_modes",
      "action": "Get Failure Modes",
      "agent": "FMSRAgent",
      "arguments": "Wind Turbine",
      "outputs": "The failure modes of asset Wind Turbine: ['Blade Failure', 'Gearbox Failure', 'Generator Failure', 'Bearing Failure', 'Electrical System Failure', 'Control System Failure', 'Foundation Failure', 'Tower Failure', 'Hub Failure', 'Pitch System Failure ', 'Yaw System Failure ', 'Hydraulic System Failure']",
      "deterministic": {
        "name": false,
        "action": true,
        "arguments": true,
        "outputs": false
      }
    },
    {
      "name": "finish",
      "action": "Finish",
      "arguments": "",
      "outputs": ""
    }
  ],
  "execution_links": [
    {"source": "get_failure_modes", "target": "finish"}
  ]
}
```

### 5.2 Multi-Step Data Processing Pattern (ID 400)

**Use Case**: Create objects, retrieve data, process, and generate output

**Scenario**: "Get the work order of equipment CWC04013 for year 2017."

**Pattern Characteristics**:
- Single agent with multiple steps
- Object creation → data retrieval → processing → output generation
- Mix of deterministic and non-deterministic steps
- Parallel execution branches (print and generate_json)

```json
{
  "planning_steps": [
    {
      "agent": "WOAgent",
      "instruction": "Create an Equipment instance using the provided equipment_id CWC04013."
    },
    {
      "agent": "WOAgent",
      "instruction": "Create a DateRange object spanning 2017-01-01 to 2017-12-31."
    },
    {
      "agent": "WOAgent",
      "instruction": "Retrieve the work orders file for the specified equipment and date range."
    },
    {
      "agent": "WOAgent",
      "instruction": "Load the work orders data from the retrieved file using pickle."
    },
    {
      "agent": "WOAgent",
      "instruction": "Generate a structured JSON file from the work orders using an LLM."
    }
  ],
  "execution_steps": [
    {
      "name": "create_equipment",
      "action": "Equipment",
      "agent": "WOAgent",
      "arguments": {"equipment_id": "CWC04013"},
      "outputs": ["equipment"],
      "deterministic": {"name": false, "action": true, "arguments": true, "outputs": true}
    },
    {
      "name": "create_date_range",
      "action": "DateRange",
      "agent": "WOAgent",
      "arguments": {"start_date": "2017-01-01", "end_date": "2017-12-31"},
      "outputs": ["date_range"],
      "deterministic": {"name": false, "action": true, "arguments": true, "outputs": true}
    },
    {
      "name": "get_work_orders_file",
      "action": "get_work_orders",
      "agent": "WOAgent",
      "arguments": {"equipment": "equipment", "date_range": "date_range"},
      "outputs": ["file"],
      "deterministic": {"name": false, "action": true, "arguments": true, "outputs": true}
    },
    {
      "name": "load_work_orders",
      "action": "pickle.load",
      "agent": "WOAgent",
      "arguments": {"file": "file"},
      "outputs": ["work_orders"],
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "print_work_orders",
      "action": "print",
      "agent": "WOAgent",
      "arguments": {"obj": "work_orders"},
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "generate_json",
      "action": "LLM.generate_json",
      "agent": "WOAgent",
      "arguments": {"data": "work_orders"},
      "outputs": ["json_file"],
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    }
  ],
  "execution_links": [
    {"source": "create_equipment", "target": "get_work_orders_file"},
    {"source": "create_date_range", "target": "get_work_orders_file"},
    {"source": "get_work_orders_file", "target": "load_work_orders"},
    {"source": "load_work_orders", "target": "print_work_orders"},
    {"source": "load_work_orders", "target": "generate_json"}
  ]
}
```

### 5.3 Multi-Agent Coordination Pattern (ID 601)

**Use Case**: Simple multi-agent coordination for knowledge retrieval

**Scenario**: "List all failure modes of asset Chiller 6 at MAIN site."

**Pattern Characteristics**:
- Two agents (IoTAgent for verification, FMSRAgent for knowledge)
- Agent handoff pattern
- Validation and finalization steps
- Structured final output

```json
{
  "planning_steps": [
    {
      "agent": "IoTAgent",
      "instruction": "Get basic description/identifiers for Chiller 6 at MAIN site (asset verification)."
    },
    {
      "agent": "FMSRAgent",
      "instruction": "List all failure modes for the verified asset (Chiller 6 at MAIN)."
    }
  ],
  "execution_steps": [
    {
      "name": "fmsr_list",
      "agent": "FMSRAgent",
      "action": "get_failure_modes",
      "arguments": {"asset_name": "Chiller 6", "site_name": "MAIN"},
      "outputs": {
        "status": "success",
        "failure_modes": [
          "Compressor Overheating: Failed due to Normal wear, overheating",
          "Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use",
          "Evaporator Water side fouling",
          "Condenser Water side fouling",
          "Condenser Improper water side flow rate",
          "Purge Unit Excessive purge",
          "Refrigerant Operated Control Valve Failed spring"
        ]
      },
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "finalize",
      "agent": "FMSRAgent",
      "action": "Finish",
      "arguments": {"validation_checks": ["len(failure_modes)==7"]},
      "outputs": {"final_state": "completed"},
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    }
  ],
  "execution_links": [
    {"source": "fmsr_list", "target": "finalize"}
  ],
  "final_output": {
    "asset": "Chiller 6",
    "site": "MAIN",
    "parameter": "Failure Modes",
    "failure_modes": [
      "Compressor Overheating: Failed due to Normal wear, overheating",
      "Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use",
      "Evaporator Water side fouling",
      "Condenser Water side fouling",
      "Condenser Improper water side flow rate",
      "Purge Unit Excessive purge",
      "Refrigerant Operated Control Valve Failed spring"
    ]
  }
}
```

### 5.4 Complex Multi-Agent with Error Recovery Pattern (ID 502)

**Use Case**: Multi-agent workflow with retry logic and error handling

**Scenario**: "What is the forecast for 'Chiller 9 Condenser Water Flow' in the week of 2020-04-27 based on data from the MAIN site?"

**Pattern Characteristics**:
- Three agents (IoTAgent, TSFMAgent coordination)
- Error detection and recovery (failed attempt → successful retry)
- Model selection and forecasting
- Comprehensive validation and artifact management

**Key Pattern Elements**:
1. **Sensor Location**: IoTAgent locates the specific sensor
2. **Data Retrieval with Retry**: Initial failure due to incorrect asset identifier, followed by successful retry
3. **Model Selection**: TSFMAgent enumerates available models and selects appropriate checkpoint
4. **Forecasting**: Run time-series forecasting with validated arguments
5. **Validation**: Confirm artifacts and provide comprehensive summary

```json
{
  "planning_steps": [
    {
      "agent": "IoTAgent",
      "instruction": "Locate 'Condenser Water Flow' sensor for 'Chiller 9' at 'MAIN'."
    },
    {
      "agent": "IoTAgent",
      "instruction": "Retrieve windowed history (2020-04-27 → 2020-05-03) and save CSV."
    },
    {
      "agent": "TSFMAgent",
      "instruction": "List available pretrained checkpoints and select 'ttm_96_28'."
    },
    {
      "agent": "TSFMAgent",
      "instruction": "Run forecast on the CSV and save JSON to declared path."
    },
    {
      "agent": "TSFMAgent",
      "instruction": "Validate artifacts and finalize summary."
    }
  ],
  "execution_steps": [
    {
      "name": "sensors",
      "agent": "IoTAgent",
      "action": "LocateSensor",
      "arguments": {
        "site_name": "MAIN",
        "assetnum": "Chiller 9",
        "sensor_name_filter": "Condenser Water Flow"
      },
      "outputs": {
        "status": "success",
        "site_name": "MAIN",
        "assetnum": "Chiller 9",
        "sensor_id": "chiller9_condenser_water_flow",
        "sensor_name": "Condenser Water Flow",
        "file_path": "/tmp/cbmdir/chiller9_MAIN_sensors.json",
        "message": "Located target sensor."
      },
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "history",
      "agent": "IoTAgent",
      "action": "DownloadSeries_Attempt1",
      "arguments": {
        "site_name": "MAIN",
        "assetnum": "9",
        "metric": "Condenser Water Flow",
        "start": "2020-04-27T00:00:00Z",
        "final": "2020-05-03T23:59:59Z",
        "save_series_csv": true,
        "save_path": "data/raw/iot/main/chiller_9_condenser_water_flow_2020-04-27_to_2020-05-03.csv"
      },
      "outputs": {
        "status": "failure",
        "error": "assetnum '9' not found (expected 'Chiller 9').",
        "csv_path": null,
        "message": "History retrieval failed due to asset identifier mismatch."
      },
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "history",
      "agent": "IoTAgent",
      "action": "DownloadSeries_ReattemptWithFixedAssetnum",
      "arguments": {
        "site_name": "MAIN",
        "assetnum": "Chiller 9",
        "metric": "Condenser Water Flow",
        "start": "2020-04-27T00:00:00Z",
        "final": "2020-05-03T23:59:59Z",
        "save_series_csv": true,
        "save_path": "data/raw/iot/main/chiller_9_condenser_water_flow_2020-04-27_to_2020-05-03.csv"
      },
      "outputs": {
        "status": "success",
        "rows": 581,
        "sampling_hint": "irregular/partial (expected ~672 @ 15min)",
        "csv_path": "data/raw/iot/main/chiller_9_condenser_water_flow_2020-04-27_to_2020-05-03.csv",
        "message": "Series retrieved and saved to CSV."
      },
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "jsonreader",
      "agent": "IoTAgent",
      "action": "DirectFileRead_Attempt",
      "arguments": {
        "file_name": "chiller9_condenser_water_flow_MAIN_2020-04-27.json"
      },
      "outputs": {
        "status": "failure",
        "error": "[Errno 2] No such file or directory: 'chiller9_condenser_water_flow_MAIN_2020-04-27.json'",
        "message": "Direct JSON read failed; proceed with CSV path from history step."
      },
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "tsfm_list_models",
      "agent": "TSFMAgent",
      "action": "EnumeratePretrainedCheckpoints",
      "arguments": {},
      "outputs": {
        "status": "success",
        "models": [
          {"model_id": "ttm_96_28", "model_checkpoint": "ttm_96_28", "model_description": "Pretrained forecasting model with context length 96"},
          {"model_id": "ttm_512_96", "model_checkpoint": "ttm_512_96", "model_description": "Pretrained forecasting model with context length 512"},
          {"model_id": "ttm_energy_96_28", "model_checkpoint": "ttm_96_28", "model_description": "Energy-tuned model, context length 96"},
          {"model_id": "ttm_energy_512_96", "model_checkpoint": "ttm_512_96", "model_description": "Energy-tuned model, context length 512"}
        ],
        "selected": "ttm_96_28",
        "message": "Selected checkpoint 'ttm_96_28' for forecasting."
      },
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "tsfm_forecast",
      "agent": "TSFMAgent",
      "action": "RunForecast_ttm_96_28_WithCorrectArgs",
      "arguments": {
        "input_csv": "data/raw/iot/main/chiller_9_condenser_water_flow_2020-04-27_to_2020-05-03.csv",
        "timestamp_col": "timestamp",
        "value_col": "value",
        "window_start": "2020-04-27T00:00:00Z",
        "window_end": "2020-05-03T23:59:59Z",
        "checkpoint": "ttm_96_28",
        "save_path": "data/derived/forecast/chiller_9/condenser_water_flow/forecast_2020-04-27_to_2020-05-03_ttm_96_28.json"
      },
      "outputs": {
        "status": "success",
        "model_checkpoint": "ttm_96_28",
        "input_read": true,
        "input_rows": 581,
        "forecast_span": {"start": "2020-04-27T00:00:00Z", "end": "2020-05-03T23:59:59Z"},
        "forecast_file": "data/derived/forecast/chiller_9/condenser_water_flow/forecast_2020-04-27_to_2020-05-03_ttm_96_28.json",
        "message": "Forecast generated and saved (arguments validated; prior unpacking/validation errors resolved)."
      },
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    },
    {
      "name": "finalize",
      "agent": "TSFMAgent",
      "action": "Finish",
      "arguments": {
        "validation_checks": [
          "exists:data/raw/iot/main/chiller_9_condenser_water_flow_2020-04-27_to_2020-05-03.csv",
          "exists:data/derived/forecast/chiller_9/condenser_water_flow/forecast_2020-04-27_to_2020-05-03_ttm_96_28.json"
        ],
        "summary_request": "Confirm variable binding, input read, forecast span, row counts, and artifact paths. Note prior failure remediation (assetnum correction; correct TSFM input schema)."
      },
      "outputs": {
        "final_state": "completed",
        "summary": "Forecast for Chiller 9 'Condenser Water Flow' (2020-04-27 → 2020-05-03) generated with checkpoint ttm_96_28. Input CSV (581 rows) read; forecast saved to declared path. Earlier issues (assetnum mismatch, missing JSON file, TSFM arg validation) were identified and resolved.",
        "forecast_file": "data/derived/forecast/chiller_9/condenser_water_flow/forecast_2020-04-27_to_2020-05-03_ttm_96_28.json"
      },
      "deterministic": {"name": false, "action": false, "arguments": false, "outputs": false}
    }
  ],
  "execution_links": [
    {"source": "sensors", "target": "history"},
    {"source": "history", "target": "history"},
    {"source": "history", "target": "jsonreader"},
    {"source": "jsonreader", "target": "tsfm_list_models"},
    {"source": "tsfm_list_models", "target": "tsfm_forecast"},
    {"source": "tsfm_forecast", "target": "finalize"}
  ]
}
```

**Key Lessons from ID 502**:
- **Error Recovery**: Document both failed and successful attempts
- **Self-Correction**: Show how agents detect and fix errors (assetnum correction)
- **Validation**: Include comprehensive validation checks in finalization
- **Artifact Management**: Track all intermediate and final file paths
- **Non-Deterministic Throughout**: Complex workflows are typically non-deterministic at every step

---

## 6. Evaluation Using Ground Truth

### 6.1 Evaluation Metrics

Ground truth enables multiple evaluation dimensions:

1. **Planning Accuracy**
   - Compare agent's planning steps to ground truth planning_steps
   - Use semantic similarity (not exact match)
   - Measure: Planning step coverage, agent selection accuracy

2. **Execution Accuracy**
   - Compare agent's execution DAG to ground truth execution DAG
   - Check: Tool selection, argument correctness, execution order
   - Measure: DAG similarity, tool usage accuracy

3. **Output Correctness**
   - Compare agent's final output to ground truth final_out
   - For deterministic: Exact match or validation rules
   - For non-deterministic: Semantic similarity, constraint satisfaction
   - Measure: Output accuracy, format compliance

4. **Efficiency**
   - Compare number of steps taken vs. ground truth
   - Identify unnecessary steps or missing optimizations
   - Measure: Step count ratio, execution time

### 6.2 Scoring Functions

**DAG Comparison**:
- Node matching: Compare execution steps
- Edge matching: Compare execution links
- Path similarity: Compare execution sequences
- Tool usage: Verify correct tools are called

**Output Comparison**:
- Exact match (deterministic)
- Semantic similarity (non-deterministic)
- Constraint validation
- Format compliance

**Planning Comparison**:
- Agent selection accuracy
- Step sequence similarity
- Semantic intent matching

---

## 7. Paradigm-Specific Best Practices

### 7.1 Multi-Agent Ground Truth Best Practices

When creating ground truth for multi-agent systems:

#### Agent Selection
- Use predefined agent names consistently
- Document agent capabilities and responsibilities
- Common agents: `IoTAgent`, `FMSRAgent`, `TSFMAgent`, `WOAgent`
- Use `multiagent` type for scenarios requiring multiple agents

#### Agent Coordination
- Show explicit agent handoffs in planning_steps
- Document inter-agent communication patterns
- Use `Agent-Ask` for agent-to-agent queries
- Ensure proper sequencing of agent activities

#### Domain-Specific Actions
- Use agent-specific function names (e.g., `get_work_orders`, `get_sensor_data`)
- Document available actions for each agent type
- Include domain objects (e.g., `Equipment`, `DateRange`, `Sensor`)

**Example Multi-Agent Pattern**:
```json
{
  "type": "multiagent",
  "planning_steps": [
    {
      "agent": "IoTAgent",
      "instruction": "Retrieve sensor data for equipment"
    },
    {
      "agent": "TSFMAgent",
      "instruction": "Analyze sensor data for anomalies"
    },
    {
      "agent": "WOAgent",
      "instruction": "Recommend work orders based on anomalies"
    }
  ],
  "execution_steps": [
    {
      "name": "get_sensor_data",
      "action": "sensors",
      "agent": "IoTAgent",
      "arguments": {"site_name": "MAIN", "assetnum": "Chiller 6"}
    },
    {
      "name": "detect_anomalies",
      "action": "detect_anomalies",
      "agent": "TSFMAgent",
      "arguments": {"data": "sensor_data"}
    },
    {
      "name": "recommend_wo",
      "action": "recommend_work_order",
      "agent": "WOAgent",
      "arguments": {"anomalies": "detected_anomalies"}
    }
  ]
}
```

### 7.2 Multi-MCP Ground Truth Best Practices

When creating ground truth for MCP (Model Context Protocol) systems:

#### MCP Server Types

Use standard MCP server types:
- `filesystem`: File operations (read, write, list, search)
- `database`: Database queries (SQL, NoSQL)
- `web-search`: Web search and scraping
- `api-client`: REST/GraphQL API calls
- `git`: Version control operations
- `slack`: Team communication
- `email`: Email operations
- `calendar`: Calendar management
- `computation`: Mathematical operations
- `memory`: Persistent storage for LLM context

#### MCP Tool Naming

Use MCP protocol tool naming conventions:
```json
{
  "action": "filesystem/read_file",      // Server/tool format
  "action": "database/query",
  "action": "web-search/search",
  "action": "api-client/post"
}
```

Or simplified format:
```json
{
  "action": "read_file",                 // Tool name only
  "agent": "filesystem"                  // Server specified separately
}
```

#### MCP Resource References

Handle MCP resources with proper URIs:
```json
{
  "arguments": {
    "resource_uri": "file:///path/to/data.csv",
    "resource_type": "text/csv"
  }
}
```

```json
{
  "arguments": {
    "resource_uri": "db://localhost/sensors",
    "query": "SELECT * FROM readings WHERE date > '2024-01-01'"
  }
}
```

#### MCP Tool Schemas

Reference MCP tool schemas in execution steps:
```json
{
  "name": "read_config",
  "action": "read_file",
  "agent": "filesystem",
  "arguments": {
    "path": "/config/app.json"
  },
  "outputs": ["file_content"],
  "deterministic": {
    "name": false,
    "action": true,
    "arguments": true,
    "outputs": true
  }
}
```

#### MCP Server Coordination

For multi-MCP scenarios:
```json
{
  "type": "multiagent",  // Or "multi-mcp"
  "planning_steps": [
    {
      "agent": "filesystem",
      "instruction": "Read data file"
    },
    {
      "agent": "database",
      "instruction": "Query reference data"
    },
    {
      "agent": "api-client",
      "instruction": "Send results to external API"
    }
  ]
}
```

**Example Multi-MCP Pattern**:
```json
{
  "id": 600,
  "text": "Read customer data from CSV, enrich with database info, and send to CRM API.",
  "type": "multiagent",
  "category": "Complex Query",
  "deterministic": true,
  "planning_steps": [
    {
      "agent": "filesystem",
      "instruction": "Read customer CSV file"
    },
    {
      "agent": "database",
      "instruction": "Query customer details from database"
    },
    {
      "agent": "api-client",
      "instruction": "POST enriched data to CRM API"
    }
  ],
  "execution_steps": [
    {
      "name": "read_csv",
      "action": "read_file",
      "agent": "filesystem",
      "arguments": {
        "path": "/data/customers.csv"
      },
      "outputs": ["csv_content"],
      "deterministic": {
        "name": false,
        "action": true,
        "arguments": true,
        "outputs": true
      }
    },
    {
      "name": "parse_csv",
      "action": "parse",
      "agent": "filesystem",
      "arguments": {
        "content": "csv_content",
        "format": "csv"
      },
      "outputs": ["customer_ids"],
      "deterministic": {
        "name": false,
        "action": true,
        "arguments": true,
        "outputs": true
      }
    },
    {
      "name": "query_details",
      "action": "query",
      "agent": "database",
      "arguments": {
        "sql": "SELECT * FROM customers WHERE id IN (@customer_ids)",
        "params": {"customer_ids": "customer_ids"}
      },
      "outputs": ["customer_details"],
      "deterministic": {
        "name": false,
        "action": true,
        "arguments": true,
        "outputs": true
      }
    },
    {
      "name": "send_to_crm",
      "action": "post",
      "agent": "api-client",
      "arguments": {
        "url": "https://crm.example.com/api/customers",
        "body": "customer_details",
        "headers": {"Content-Type": "application/json"}
      },
      "outputs": ["api_response"],
      "deterministic": {
        "name": false,
        "action": true,
        "arguments": true,
        "outputs": false
      }
    }
  ],
  "execution_links": [
    {"source": "read_csv", "target": "parse_csv"},
    {"source": "parse_csv", "target": "query_details"},
    {"source": "query_details", "target": "send_to_crm"}
  ]
}
```

### 7.3 Hybrid Approach Best Practices

When combining multi-agent and multi-MCP:

#### Clear Separation of Concerns
- Agents handle reasoning and decision-making
- MCP servers provide tools and data access
- Document which layer handles what

#### Consistent Naming
- Use `agent` field for both agents and MCP servers
- Use `type` to indicate primary system type
- Add `paradigm` field if needed for clarity:
  ```json
  {
    "type": "multiagent",
    "paradigm": "hybrid",
    "note": "IoTAgent uses filesystem MCP server"
  }
  ```

#### Layered Execution
```json
{
  "execution_steps": [
    {
      "name": "agent_decides",
      "action": "analyze_requirements",
      "agent": "PlannerAgent",
      "outputs": ["required_files"]
    },
    {
      "name": "mcp_retrieves",
      "action": "read_file",
      "agent": "filesystem",
      "arguments": {"paths": "required_files"},
      "outputs": ["file_contents"]
    },
    {
      "name": "agent_processes",
      "action": "process_data",
      "agent": "ProcessorAgent",
      "arguments": {"data": "file_contents"},
      "outputs": ["results"]
    }
  ]
}
```

### 7.4 Choosing Between Paradigms

**Use Multi-Agent When**:
- System has specialized agents with domain expertise
- Focus on agent reasoning and coordination
- Agents make complex decisions
- Example: AssetOpsBench, healthcare diagnosis, financial analysis

**Use Multi-MCP When**:
- System uses standardized tool interfaces
- Focus on tool selection and usage
- LLM orchestrates tool calls
- Example: File operations, database queries, API integrations

**Use Hybrid When**:
- Agents need external tools
- Combining reasoning with data access
- Best of both worlds
- Example: AI agent using filesystem and database MCP servers

---

## 8. Using Ground Truth for Agent Evaluation

### 8.1 Three-Level Evaluation Framework

Ground truth enables comprehensive evaluation at three levels:

| Level | Focus | Metrics | Weight |
|-------|-------|---------|--------|
| **Outcome** | Final result correctness | Task completeness (y₁), Data retrieval (y₂), Result verification (y₃) | 50% |
| **Process** | Execution trajectory | Action sequence, DAG structure, Step completeness | 30% |
| **Planning** | Strategic decomposition | ROUGE scores (planning alignment) | 20% |

### 8.2 Evaluation Workflow

**Step 1: Collect Agent Trace** → **Step 2: Load Ground Truth** → **Step 3: Multi-Level Comparison**

### 8.3 Rubric-Based Scoring (Outcome Level)

**Three Core Metrics**:

1. **Task Completeness (y₁)**: Compare agent output against `characteristic_form` using LLM-as-Judge (0-1 score)
2. **Data Retrieval Accuracy (y₂)**: Verify correct parameters (equipment IDs, date ranges, sensors)
3. **Result Verification (y₃)**:
   - Deterministic: Exact match with `final_out`
   - Non-deterministic: Semantic similarity

**Combined Score**: `Rubric_Score = (y₁ + y₂ + y₃) / 3`

**Example** (Scenario 421):
- y₁ = 1.0 (task completed), y₂ = 1.0 (correct params), y₃ = 0.6 (3/5 codes)
- Rubric Score = 0.87

### 8.4 Execution Alignment (Process Level)

**Comparison Methods**:
- **Action Sequence**: Use sequence alignment (e.g., Levenshtein distance) to compare action lists
- **DAG Structure**: Calculate node/edge overlap between ground truth and agent execution graphs
- **Step Completeness**: Identify missing or extra steps

**Example**:
- GT: 8 steps, Agent: 6 steps (missing `load_events`, `dedupe`)
- Alignment Score: 6/8 = 0.75

### 8.5 Planning Alignment (Strategic Level)

Use ROUGE metrics to compare planning strategies:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

**Example**: GT planning vs Agent planning → ROUGE-L: 0.65

### 8.6 Hybrid Scoring Formula

```
Final_Score = 0.5 × Rubric_Score + 0.3 × Execution_Score + 0.2 × Planning_Score
```

**Complete Example** (Scenario 421):
```json
{
  "final_score": 0.78,
  "rubric_score": 0.87,
  "execution_score": 0.68,
  "planning_score": 0.65,
  "failure_analysis": {
    "missing_steps": ["load_events", "dedupe"],
    "output_issues": "Missing 2 of 5 expected codes"
  }
}
```

### 8.7 Batch Evaluation

**Aggregate Metrics Across Benchmark**:
- Average final score
- Task completion rate
- Per-scenario breakdown
- Failure mode distribution

**AssetOpsBench Example**: GPT-4.1 achieved 0.72 avg score across 141 scenarios (70% completion rate)

### 8.8 Automated Failure Detection

Ground truth enables automatic identification of:
- **Tool Selection Errors**: Wrong action at specific step
- **Parameter Errors**: Incorrect arguments passed
- **Missing Steps**: Skipped required operations
- **Incorrect Sequencing**: Steps executed out of order
- **Premature Termination**: Task incomplete

### 8.9 Evaluation Best Practices

| Scenario Type | Approach |
|---------------|----------|
| **Deterministic** | Exact matching, strict validation, focus on correctness |
| **Non-Deterministic** | Semantic similarity, allow trajectory variation, focus on outcome quality |
| **Multi-Agent** | Track coordination, evaluate inter-agent communication, verify specialization |
| **Multi-MCP** | Verify server selection, check tool parameters, validate resource URIs |

---

## 9. Illustrated Examples from AssetOpsBench

### Example 1: Simple Query (Scenario 3)
**Utterance**: "What assets can be found at the MAIN site?"

```json
{
  "id": 3,
  "type": "IoT",
  "text": "What assets can be found at the MAIN site?",
  "category": "Knowledge Query",
  "deterministic": true,
  "characteristic_form": "The expected response should be the return value from querying the assets at the MAIN site. The response should be a reference to a file containing the list of assets",
  "group": "retrospective",
  "entity": "Site",
  "note": "Source: IoT data operations; Deterministic query with single correct answer; Category: Knowledge Query",
  "planning_steps": [
    {"agent": "IoTAgent", "instruction": "list assets for site MAIN"}
  ],
  "execution_steps": [
    {
      "name": "step1",
      "action": "assets",
      "agent": "IoTAgent",
      "arguments": {"site_name": "MAIN"},
      "outputs": []
    },
    {
      "name": "finish",
      "action": "Finish",
      "agent": "IoTAgent",
      "argument": "The assets at the MAIN site are: CQPA AHU 1, CQPA AHU 2B, Chiller 4, Chiller 6, Chiller 9, Chiller 3."
    }
  ],
  "execution_links": [{"source": "step1", "target": "finish"}]
}
```

**Key Features**: Linear DAG (2 steps), fully deterministic, single agent. Includes complete meta information: `text`, `characteristic_form`, `group` (retrospective), `entity` (Site), and `note` fields.

### Example 2: Complex Diagnostic (Scenario 421)
**Utterance**: "Review anomalies/alerts for Chiller 9 (CWC04009) in May-June 2020 and suggest work orders."

```json
{
  "id": 421,
  "type": "Workorder",
  "deterministic": false,
  "planning_steps": [
    {"agent": "WOAgent", "instruction": "Create Equipment(CWC04009) and DateRange(2020-05-01 to 2020-06-30)"},
    {"agent": "WOAgent", "instruction": "Retrieve and filter events to ALERT and ANOMALY"},
    {"agent": "WOAgent", "instruction": "Get KPI-based and alert-based recommendations in parallel"},
    {"agent": "WOAgent", "instruction": "Combine and deduplicate recommendations"}
  ],
  "execution_steps": [
    {"name": "create_equipment", "action": "Equipment", "arguments": {"equipment_id": "CWC04009"}, "outputs": ["equipment"]},
    {"name": "create_date_range", "action": "DateRange", "arguments": {"start_date": "2020-05-01", "end_date": "2020-06-30"}, "outputs": ["date_range"]},
    {"name": "get_events", "action": "get_events", "arguments": {"equipment": "equipment", "date_range": "date_range"}, "outputs": ["events_pickle_path"]},
    {"name": "load_events", "action": "pickle_load", "arguments": {"file_path": "events_pickle_path"}, "outputs": ["events"]},
    {"name": "filter_anomaly_alert", "action": "filter", "arguments": {"obj": "events", "condition": "event.event_category in {'ALERT','ANOMALY'}"}, "outputs": ["anomaly_alert_events"]},
    {"name": "get_top_kpi_recommendations", "action": "recommend_from_top_kpi_anomalies", "arguments": {"events": "anomaly_alert_events", "top_k": 3}, "outputs": ["kpi_recommendations"]},
    {"name": "alert_recommendation", "action": "recommend_work_order_from_alert", "arguments": {"rule_id": "RUL0018"}, "outputs": ["alert_rec"]},
    {"name": "combine_and_dedupe", "action": "dedupe_by_key", "arguments": {"items": "[alert_rec] + kpi_recommendations", "key": "rec.primary_code"}, "outputs": ["final_result"]},
    {"name": "print_final", "action": "print", "arguments": {"obj": "final_result"}}
  ],
  "execution_links": [
    {"source": "create_equipment", "target": "get_events"},
    {"source": "create_date_range", "target": "get_events"},
    {"source": "get_events", "target": "load_events"},
    {"source": "load_events", "target": "filter_anomaly_alert"},
    {"source": "filter_anomaly_alert", "target": "get_top_kpi_recommendations"},
    {"source": "filter_anomaly_alert", "target": "alert_recommendation"},
    {"source": "get_top_kpi_recommendations", "target": "combine_and_dedupe"},
    {"source": "alert_recommendation", "target": "combine_and_dedupe"},
    {"source": "combine_and_dedupe", "target": "print_final"}
  ],
  "final_out": {"CWC04009": ["M005", "M006", "OP004", "M013", "OP002"]}
}
```

**Key Features**: Parallel branches (KPI + alert paths converge at dedupe), non-deterministic filtering/reasoning, 9 steps with 8 dependencies, ~3-4 hours to create.

**DAG Pattern**:
```
create_equipment ──┐
                   ├─> get_events ─> load_events ─> filter ─┬─> kpi_rec ──┐
create_date_range ─┘                                         │              ├─> dedupe ─> print
                                                              └─> alert_rec ─┘
```

**Comparison**:

| Aspect | Example 1 | Example 2 |
|--------|-----------|-----------|
| Steps | 2 | 9 |
| Dependencies | 1 | 8 |
| Deterministic | Fully | Partially |
| DAG Pattern | Linear | Parallel + Convergence |
| Creation Time | 30 min | 3-4 hours |

---

## 10. Two Paradigms: Multi-Agent vs Multi-MCP

### 10.1 Unified Ground Truth Structure

**Key Insight**: Both paradigms use the **same ground truth structure**. The only difference is semantic interpretation:

| Field | Multi-Agent Interpretation | Multi-MCP Interpretation |
|-------|---------------------------|-------------------------|
| `type` | Agent type (IoTAgent, WOAgent) | MCP server type (filesystem, database) |
| `agent` | Agent name | MCP server name |
| `action` | Agent-specific function | MCP tool name |
| `planning_steps` | Agent orchestration plan | MCP server coordination plan |
| `execution_steps` | Agent function calls | MCP tool invocations |

### 10.2 Understanding the Two Approaches

Ground truth creation supports two distinct but structurally similar paradigms:

#### Multi-Agent Systems

**Concept**: Multiple specialized AI agents, each with domain expertise, collaborate to solve complex tasks.

**Characteristics**:
- Each agent has specific capabilities (e.g., IoTAgent handles sensor data, WOAgent manages work orders)
- Agents communicate and coordinate through orchestration
- Agent selection is based on task requirements and domain knowledge
- Example: AssetOpsBench with IoTAgent, FMSRAgent, TSFMAgent, WOAgent

**Ground Truth Representation**:
```json
{
  "type": "Workorder",           // Primary agent type
  "planning_steps": [
    {
      "agent": "WOAgent",         // Specific agent name
      "instruction": "Create Equipment instance"
    }
  ],
  "execution_steps": [
    {
      "name": "create_equipment",
      "action": "Equipment",      // Agent-specific action
      "agent": "WOAgent",         // Agent executing this step
      "arguments": {...}
    }
  ]
}
```

#### Multi-MCP Systems

**Concept**: Multiple Model Context Protocol (MCP) servers provide standardized tools and resources that LLMs can use to accomplish tasks.

**Characteristics**:
- Each MCP server exposes specific capabilities (e.g., filesystem operations, database queries, web search)
- MCP servers are service locators providing tools to LLMs
- Server selection is based on required operations and data sources
- Example: MCP-Bench with filesystem, database, web-search, api-client servers

**Ground Truth Representation**:
```json
{
  "type": "filesystem",          // Primary MCP server type
  "planning_steps": [
    {
      "agent": "filesystem",      // MCP server name
      "instruction": "Read configuration file"
    }
  ],
  "execution_steps": [
    {
      "name": "read_config",
      "action": "read_file",      // MCP tool name
      "agent": "filesystem",      // MCP server executing this step
      "arguments": {
        "path": "/config/app.json"
      }
    }
  ]
}
```

### 10.3 When to Use Each Paradigm

**Use Multi-Agent Ground Truth When**:
- Evaluating specialized AI agent systems
- Agents have distinct domain expertise and reasoning capabilities
- Focus is on agent coordination and decision-making
- Example domains: Industrial operations, healthcare diagnostics, financial analysis

**Use Multi-MCP Ground Truth When**:
- Evaluating LLM tool-using capabilities
- Focus is on tool selection and parameter passing
- Servers provide standardized interfaces (MCP protocol)
- Example domains: File operations, database queries, API integrations, web research

**Use Hybrid Approach When**:
- System combines both paradigms
- AI agents use MCP servers as tools
- Example: An IoTAgent that uses filesystem MCP server to read sensor data files

### 10.4 Conversion Between Paradigms

Ground truth can be adapted between paradigms by reinterpreting fields:

**Multi-Agent → Multi-MCP**:
```json
// Multi-Agent
{
  "type": "IoTAgent",
  "action": "get_sensor_data"
}

// Equivalent Multi-MCP
{
  "type": "database",
  "action": "query"
}
```

**Multi-MCP → Multi-Agent**:
```json
// Multi-MCP
{
  "type": "filesystem",
  "action": "read_file"
}

// Equivalent Multi-Agent
{
  "type": "FileAgent",
  "action": "read"
}
```

### 10.5 Examples of Both Paradigms

#### Example 1: Multi-Agent Ground Truth

**Scenario**: Retrieve work orders for equipment

```json
{
  "id": 400,
  "text": "Get the work order of equipment CWC04013 for year 2017.",
  "type": "Workorder",
  "category": "Data Query",
  "deterministic": true,
  "planning_steps": [
    {
      "agent": "WOAgent",
      "instruction": "Create Equipment instance for CWC04013"
    },
    {
      "agent": "WOAgent",
      "instruction": "Retrieve work orders for 2017"
    }
  ],
  "execution_steps": [
    {
      "name": "create_equipment",
      "action": "Equipment",
      "agent": "WOAgent",
      "arguments": {"equipment_id": "CWC04013"}
    },
    {
      "name": "get_work_orders",
      "action": "get_work_orders",
      "agent": "WOAgent",
      "arguments": {
        "equipment": "equipment",
        "date_range": "date_range"
      }
    }
  ]
}
```

#### Example 2: Multi-MCP Ground Truth

**Scenario**: Read and analyze configuration file

```json
{
  "id": 501,
  "text": "Read the application configuration and validate required fields.",
  "type": "filesystem",
  "category": "Data Query",
  "deterministic": true,
  "planning_steps": [
    {
      "agent": "filesystem",
      "instruction": "Read configuration file from disk"
    },
    {
      "agent": "json-parser",
      "instruction": "Parse JSON configuration"
    },
    {
      "agent": "validator",
      "instruction": "Validate required fields"
    }
  ],
  "execution_steps": [
    {
      "name": "read_config",
      "action": "read_file",
      "agent": "filesystem",
      "arguments": {
        "path": "/config/app.json"
      },
      "outputs": ["config_content"]
    },
    {
      "name": "parse_json",
      "action": "parse",
      "agent": "json-parser",
      "arguments": {
        "content": "config_content"
      },
      "outputs": ["config_object"]
    },
    {
      "name": "validate",
      "action": "validate_schema",
      "agent": "validator",
      "arguments": {
        "data": "config_object",
        "required_fields": ["api_key", "endpoint", "timeout"]
      },
      "outputs": ["validation_result"]
    }
  ]
}
```

#### Example 3: Hybrid Ground Truth

**Scenario**: IoT agent uses filesystem MCP server

```json
{
  "id": 502,
  "text": "Load sensor data from CSV file and detect anomalies.",
  "type": "multiagent",
  "category": "Complex Query",
  "deterministic": false,
  "planning_steps": [
    {
      "agent": "filesystem",
      "instruction": "Read sensor data CSV file"
    },
    {
      "agent": "IoTAgent",
      "instruction": "Parse and validate sensor data"
    },
    {
      "agent": "AnomalyDetectionAgent",
      "instruction": "Detect anomalies in sensor readings"
    }
  ],
  "execution_steps": [
    {
      "name": "read_csv",
      "action": "read_file",
      "agent": "filesystem",
      "arguments": {"path": "/data/sensors.csv"},
      "outputs": ["csv_content"]
    },
    {
      "name": "parse_data",
      "action": "parse_sensor_data",
      "agent": "IoTAgent",
      "arguments": {"csv_data": "csv_content"},
      "outputs": ["sensor_readings"]
    },
    {
      "name": "detect_anomalies",
      "action": "detect",
      "agent": "AnomalyDetectionAgent",
      "arguments": {"data": "sensor_readings"},
      "outputs": ["anomalies"]
    }
  ]
}
```

### 10.6 Choosing the Right Paradigm for Your Ground Truth

**Decision Framework**:

1. **What are you evaluating?**
   - AI agent reasoning and coordination → Multi-Agent
   - LLM tool-using capabilities → Multi-MCP
   - Both → Hybrid

2. **What is your system architecture?**
   - Specialized agents with domain logic → Multi-Agent
   - MCP servers with standardized tools → Multi-MCP
   - Agents using MCP tools → Hybrid

3. **What is your evaluation focus?**
   - Agent selection and orchestration → Multi-Agent
   - Tool selection and parameter passing → Multi-MCP
   - End-to-end task completion → Either/Both

**Recommendation**: Start with the paradigm that matches your system architecture, then adapt as needed. The ground truth structure is flexible enough to support both.

---

## 11. Summary

### Core Requirements for High-Quality Ground Truth

1. **Clear Understanding**: Thoroughly understand utterance intent and domain context
2. **Structured Planning**: Break down problems into logical `planning_steps`
3. **Detailed Execution**: Specify exact tools, arguments, and data flow in `execution_steps`
4. **Valid DAG**: Ensure `execution_links` form proper directed acyclic graph
5. **Accurate Determinism**: Mark determinism correctly at all levels
6. **Complete Output**: Provide `final_out` or `final_out_description` with validation criteria
7. **Thorough Validation**: Verify completeness, correctness, and testability

### Key Principles

- **Consistency**: Standard patterns and naming conventions
- **Completeness**: All necessary steps and information
- **Clarity**: Explicit and unambiguous expectations
- **Validity**: Structural and semantic correctness
- **Testability**: Automated evaluation and comparison

### Document Organization

**Main Sections** (1-12): Core concepts, workflow, structure, best practices, examples, value proposition
**References** (Section 13): Citations and related work
**Appendices** (A-E): Detailed technical content
  - **Appendix A**: JSON Schema
  - **Appendix B**: Evaluation Metrics Details
  - **Appendix C**: Common Failure Patterns
  - **Appendix D**: Domain-Specific Examples
  - **Appendix E**: Quick Reference Checklist

### Getting Started

1. Review **Section 2** (High-Level Workflow) for the 5-phase process
2. Study **Section 11** (Illustrated Examples) for concrete patterns
3. Use **Appendix E** (Quick Reference Checklist) during creation
4. Consult **Section 4** (Ground Truth Structure) for field definitions
5. Reference **Section 10** (Paradigm-Specific Practices) for multi-agent vs. multi-MCP guidance

### Value Proposition

Ground truth trajectories enable:
- Systematic benchmarking of agent architectures
- Objective comparison of models and approaches
- Automated discovery of failure modes
- Reproducible research across the community
- Confident deployment to production systems

**ROI**: 564 hours to create 141 scenarios → 1000+ hours saved in evaluation (AssetOpsBench experience)

By following this best practice, you can create ground truth data that enables systematic, reproducible evaluation of AI agent systems across diverse domains and use cases.

---

## 12. References

### Primary References

1. **Patel, D., Lin, S., Rayfield, J., Zhou, N., Vaculin, R., Martinez, N., O'donncha, F., & Kalagnanam, J. (2025)**. "AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance." *arXiv preprint arXiv:2506.03828*.
   - URL: https://arxiv.org/abs/2506.03828
   - Key Contribution: 141 ground truth scenarios, evaluation metrics (rubric-based + ROUGE), architectural comparison
2. **AssetOpsBench Utterance Dataset**. IBM Research (2025).
   - URL: https://huggingface.co/datasets/ibm-research/AssetOpsBench
   - Contains: 152 task utterances for industrial asset operations and maintenance scenarios
   - Platform: HuggingFace Datasets
3. **AssetOpsBench Ground Truth Dataset (2026 - IBM Internal)**.
   - Repository: `AssetOpsBenchGroundTruth/`
   - Contains: 152 validated scenarios across IoT, Workorder, TSFM, and FMSR domains
4. **Lin, C. Y. (2004)**. "ROUGE: A Package for Automatic Evaluation of Summaries." *Proceedings of the ACL Workshop on Text Summarization Branches Out*.
   - Used for planning and execution alignment scoring in ground truth evaluation

### Technical Specifications

5. **Model Context Protocol (MCP) Specification**. Anthropic.
   - URL: https://modelcontextprotocol.io/
   - Framework for standardized tool interfaces in multi-MCP systems

---

## Appendix A: Ground Truth JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "text", "type", "category", "deterministic", "characteristic_form", "planning_steps", "execution_steps", "execution_links"],
  "properties": {
    "id": {"type": "integer"},
    "uuid": {"type": "string", "format": "uuid"},
    "text": {"type": "string"},
    "type": {"type": "string"},
    "category": {"type": "string"},
    "deterministic": {"type": "boolean"},
    "characteristic_form": {"type": "string"},
    "planning_steps": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["agent", "instruction"],
        "properties": {
          "agent": {"type": "string"},
          "instruction": {"type": "string"}
        }
      }
    },
    "execution_steps": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "action", "agent"],
        "properties": {
          "name": {"type": "string"},
          "action": {"type": "string"},
          "agent": {"type": "string"},
          "arguments": {"type": ["object", "string"]},
          "outputs": {"type": "array"},
          "deterministic": {
            "type": "object",
            "properties": {
              "name": {"type": "boolean"},
              "action": {"type": "boolean"},
              "arguments": {"type": "boolean"},
              "outputs": {"type": "boolean"}
            }
          }
        }
      }
    },
    "execution_links": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["source", "target"],
        "properties": {
          "source": {"type": "string"},
          "target": {"type": "string"}
        }
      }
    },
    "final_out": {"type": ["object", "null"]},
    "final_out_description": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
```

---

## Appendix B: Common Failure Patterns

### C.1 Tool Selection Errors

**Pattern**: Agent uses wrong tool for the task
```json
// Ground Truth
{"action": "get_work_orders"}

// Agent Execution
{"action": "get_events"}  // Wrong tool
```

**Detection**: Compare `execution_steps[].action` fields
**Fix**: Improve tool descriptions, add examples

### C.2 Parameter Errors

**Pattern**: Agent uses incorrect arguments
```json
// Ground Truth
{"equipment_id": "CWC04013"}

// Agent Execution
{"equipment_id": "CWC04009"}  // Wrong equipment
```

**Detection**: Compare `execution_steps[].arguments` fields
**Fix**: Validate parameter extraction, add constraints

### C.3 Missing Steps

**Pattern**: Agent skips required operations
```json
// Ground Truth: 5 steps
create_equipment → create_date_range → get_events → load_events → process

// Agent: 3 steps (missing load_events, process)
create_equipment → create_date_range → get_events
```

**Detection**: Compare DAG node counts and structure
**Fix**: Improve planning, add step-by-step guidance

### C.4 Incorrect Sequencing

**Pattern**: Agent executes steps in wrong order
```json
// Ground Truth
A → B → C

// Agent Execution
A → C → B  // Wrong order, may cause dependency errors
```

**Detection**: Compare `execution_links` topology
**Fix**: Enforce dependency constraints, validate prerequisites

### C.5 Premature Termination

**Pattern**: Agent stops before completing workflow
```json
// Ground Truth
... → final_step → Finish

// Agent Execution
... → partial_step  // Stops early, no Finish
```

**Detection**: Check for `Finish` action and `final_out` presence
**Fix**: Add completion criteria, validate output requirements

---

## Appendix D: Domain-Specific Examples

### D.1 Healthcare Diagnosis

```json
{
  "id": 701,
  "text": "Review patient symptoms and lab results to recommend diagnostic tests.",
  "type": "Healthcare",
  "category": "Diagnostic Recommendation",
  "planning_steps": [
    {"agent": "DiagnosticAgent", "instruction": "Retrieve patient symptoms and history"},
    {"agent": "DiagnosticAgent", "instruction": "Analyze lab results"},
    {"agent": "DiagnosticAgent", "instruction": "Recommend diagnostic tests based on findings"}
  ]
}
```

### D.2 Financial Analysis

```json
{
  "id": 801,
  "text": "Analyze Q4 revenue trends and forecast Q1 performance.",
  "type": "Finance",
  "category": "Forecasting",
  "planning_steps": [
    {"agent": "FinanceAgent", "instruction": "Retrieve Q4 revenue data"},
    {"agent": "FinanceAgent", "instruction": "Calculate trends and growth rates"},
    {"agent": "FinanceAgent", "instruction": "Generate Q1 forecast using time series model"}
  ]
}
```

### D.3 Customer Support

```json
{
  "id": 901,
  "text": "Resolve customer complaint about delayed shipment.",
  "type": "CustomerService",
  "category": "Issue Resolution",
  "planning_steps": [
    {"agent": "SupportAgent", "instruction": "Retrieve order and shipment details"},
    {"agent": "SupportAgent", "instruction": "Identify delay cause"},
    {"agent": "SupportAgent", "instruction": "Propose resolution and compensation"}
  ]
}
```

---

