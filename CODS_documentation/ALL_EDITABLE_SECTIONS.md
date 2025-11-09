# 🎯 All 4 Editable Sections: Side-By-Side Reference

## TRACK 1: Planning

### ✏️ EDIT 1: Agent Information Formatting

**File:** `src/agent_hive/workflows/track1_planning.py`  
**Lines:** 65-89

#### Current Code:
```python
for ii, aagent in enumerate(task.agents):
    agent_descriptions += f"\n({ii + 1}) Agent name: {aagent.name}"
    agent_descriptions += f"\nAgent description: {aagent.description}"
    if "task_examples" in aagent.__dict__ and aagent.task_examples:
        agent_descriptions += f"\nTasks that agent can solve:"
        for idx, task_example in enumerate(aagent.task_examples, start=1):
            agent_descriptions += f"\n{idx}. {task_example}"
    agent_descriptions += "\n"
```

#### Constraints:
- ✅ CAN: Change formatting, add emojis, restructure layout
- ✅ CAN: Add metadata (tool counts, tags, categories)
- ✅ CAN: Enhance readability
- ❌ CANNOT: Remove agent description or examples
- ❌ CANNOT: Add new agents to list
- ❌ CANNOT: Modify workflow logic

#### Example Enhancement:
```python
for ii, aagent in enumerate(task.agents):
    # Determine agent type/domain
    domain = "Sensing" if "IoT" in aagent.name else \
             "Analysis" if "TSFM" in aagent.name else \
             "Reasoning" if "FMSR" in aagent.name else "Action"
    
    agent_descriptions += f"\n🤖 [{ii + 1}] {aagent.name} [{domain}]"
    agent_descriptions += f"\n   📝 {aagent.description}"
    agent_descriptions += f"\n   🛠️ Tools available: {len(aagent.tools)}"
    
    if "task_examples" in aagent.__dict__ and aagent.task_examples:
        agent_descriptions += f"\n   ✅ {domain} Agent can solve:"
        # Group examples if possible
        for idx, task_example in enumerate(aagent.task_examples[:3], start=1):
            short_ex = task_example[:50] + "..." if len(task_example) > 50 else task_example
            agent_descriptions += f"\n      • {short_ex}"
        if len(aagent.task_examples) > 3:
            agent_descriptions += f"\n      + {len(aagent.task_examples)-3} more..."
    agent_descriptions += "\n"
```

---

### ✏️ EDIT 2: Planning Prompt Template ⭐ HIGHEST IMPACT

**File:** `src/agent_hive/workflows/track1_planning.py`  
**Lines:** 167-198

#### Current Code:
```python
def get_prompt(self, task_description, agent_descriptions):
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
    return prompt
```

#### Constraints:
- ✅ CAN: Change wording, add detail, improve structure
- ✅ CAN: Add domain-specific heuristics
- ✅ CAN: Provide examples of good plans
- ✅ CAN: Add quality criteria
- ✅ CAN: Emphasize particular reasoning strategies
- ❌ CANNOT: Change the format strings (#Task, #Agent, #Dependency, #ExpectedOutput)
- ❌ CANNOT: Change the prompt structure fundamentally
- ❌ CANNOT: Add new agents or constraints

#### Example Enhancement:
```python
def get_prompt(self, task_description, agent_descriptions):
    prompt = f"""
🏭 INDUSTRIAL OPERATIONS PLANNER - Create Maintenance Action Plans

Your task: Create a step-by-step action plan to solve complex industrial problems.

📋 PLANNING STRATEGY (proven to work):
1. **Data Collection** - Always start by gathering complete sensor/system data
2. **Analysis** - Examine data for patterns, anomalies, correlations
3. **Reasoning** - Match findings to failure modes and root causes
4. **Action** - Plan maintenance, work orders, or system adjustments

⚙️ CRITICAL REQUIREMENTS:
- Use ONLY the agents listed below (no new agents allowed)
- Maximum 4 steps (aim for 3)
- Each step must be independent and achievable
- Dependencies must form valid DAG (no cycles)
- Task descriptions must be specific with parameters
- Agent names must match EXACTLY (case-sensitive)

📝 FORMAT (MUST follow exactly for parsing):
#Task<N>: <specific, actionable task with parameters and context>
#Agent<N>: <exact agent name from list>
#Dependency<N>: <None or #S1, #S2, ... (space-separated)>
#ExpectedOutput<N>: <describe expected format, key fields, or result structure>

## Available Agents (by execution phase): ##
{agent_descriptions}

## Problem to Solve: ##
{task_description}

## Planning Guidelines:
- For maintenance questions: Prioritize 1) data validation, 2) anomaly detection, 3) root cause
- Use agent specialization: IoT for data, TSFM for analysis, FMSR for diagnosis, WO for action
- Consider temporal dependencies: later steps may need outputs from earlier steps
- Ensure efficiency: reuse data when possible, avoid redundant collection
- Each task should produce concrete outputs useful for next steps

## Example Good Plan:
(Use if task involves equipment diagnostics)
## Step 1
#Task1: Download historical sensor data for the target equipment from past 7 days
#Agent1: IoT Data Download
#Dependency1: None
#ExpectedOutput1: JSON file with sensor readings (timestamps, values for all parameters)

## Step 2
#Task2: Analyze time-series sensor data for anomalies and trends
#Agent2: Time Series Analytics and Forecasting
#Dependency2: #S1
#ExpectedOutput2: List of detected anomalies with timestamps, normal ranges, deviation severity

## Step 3
#Task3: Map detected anomalies to failure modes and sensors
#Agent3: Failure Mode & Sensor Relevancy Expert
#Dependency3: #S2
#ExpectedOutput3: Most likely failure modes with confidence scores and affected components

## Your Plan:
(Create efficient plan following above strategy)
"""
    return prompt
```

---

## TRACK 2: Execution

### ✏️ EDIT 3: Task Revision Helper Agent

**File:** `src/agent_hive/workflows/track2_execution.py`  
**Lines:** 31-81

#### Current Code:
```python
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
        raise NotImplementedError("Participants must implement this method.")
```

#### Constraints:
- ✅ CAN: Implement validation logic
- ✅ CAN: Add clarity checking
- ✅ CAN: Enrich with domain context
- ✅ CAN: Score quality
- ✅ CAN: Return enriched/revised input
- ❌ CANNOT: Call other agents
- ❌ CANNOT: Modify the method signature
- ❌ CANNOT: Break the interface

#### Example Implementation:
```python
def execute_task(self, task_input: str) -> str:
    """
    Validate and enrich task input for better agent execution.
    
    Checks:
    - Clarity and completeness
    - Presence of key information
    - Domain context
    
    Enriches with:
    - Domain hints
    - Expected output format
    - Quality assessment
    """
    
    enriched_lines = []
    enriched_lines.append("=" * 60)
    enriched_lines.append("TASK VALIDATION & ENRICHMENT REPORT")
    enriched_lines.append("=" * 60)
    enriched_lines.append(f"\nORIGINAL TASK:\n{task_input}\n")
    
    # Quality assessment
    quality_issues = []
    
    # Check length
    words = len(task_input.split())
    if words < 5:
        quality_issues.append(f"❌ Too brief ({words} words) - add more context")
    elif words < 15:
        quality_issues.append(f"⚠️  Could be more specific ({words} words)")
    else:
        quality_issues.append(f"✅ Good length ({words} words)")
    
    # Check for action words
    action_words = ["identify", "find", "analyze", "diagnose", "detect", "check", "verify", "create"]
    has_action = any(word in task_input.lower() for word in action_words)
    if not has_action:
        quality_issues.append("⚠️  No clear action verb - what should agent DO?")
    else:
        quality_issues.append("✅ Clear action verb present")
    
    # Check for time context
    has_time = any(word in task_input.lower() for word in ["today", "week", "month", "year", "recent", "last"])
    if not has_time:
        quality_issues.append("⚠️  No temporal context - what time period?")
    else:
        quality_issues.append("✅ Temporal context present")
    
    # Check for asset/system context
    has_asset = any(word in task_input.lower() for word in ["chiller", "pump", "equipment", "system", "sensor", "motor"])
    if not has_asset:
        quality_issues.append("⚠️  No specific asset/system - which one?")
    else:
        quality_issues.append("✅ Specific asset identified")
    
    # Report quality
    enriched_lines.append("QUALITY ASSESSMENT:")
    for issue in quality_issues:
        enriched_lines.append(f"  {issue}")
    
    # Domain context
    enriched_lines.append("\n" + "=" * 60)
    enriched_lines.append("DOMAIN CONTEXT FOR AGENT:")
    enriched_lines.append("=" * 60)
    enriched_lines.append("- Area: Industrial Asset Operations & Maintenance")
    enriched_lines.append("- Data types: Time-series sensor readings, system logs, event data")
    enriched_lines.append("- Available tools: IoT queries, time-series analysis, failure mode maps, work order generation")
    enriched_lines.append("- Expected process: Data → Analyze → Diagnose → Recommend Action")
    
    # Recommendations
    enriched_lines.append("\n" + "=" * 60)
    enriched_lines.append("RECOMMENDED APPROACH FOR AGENT:")
    enriched_lines.append("=" * 60)
    enriched_lines.append("1. **Validate input** - Ensure you have all needed information")
    enriched_lines.append("2. **Gather data** - If task mentions specific assets/sensors, query them")
    enriched_lines.append("3. **Analyze data** - Look for anomalies, trends, patterns")
    enriched_lines.append("4. **Correlate** - Match findings to known failure modes")
    enriched_lines.append("5. **Recommend** - Suggest maintenance actions or work orders")
    
    # Original task (important for agent)
    enriched_lines.append("\n" + "=" * 60)
    enriched_lines.append("TASK FOR AGENT (solve this):")
    enriched_lines.append("=" * 60)
    enriched_lines.append(task_input)
    
    enriched_lines.append("\n" + "=" * 60)
    
    return "\n".join(enriched_lines)
```

---

### ✏️ EDIT 4: Execution Logic ⭐ HIGHEST IMPACT (Track 2)

**File:** `src/agent_hive/workflows/track2_execution.py`  
**Lines:** 141-180

#### Current Code:
```python
def run(self):
    self.memory = []
    self.context_type = ContextType.SELECTED
    max_loops = 15
    i = 0
    while i < len(self.tasks) and i < max_loops:
        task = self.tasks[i]
        task_no = i + 1
        logger.info(f"Task {task_no}: {task.description}")
        assigned_agents = task.agents
        user_input = self._build_input(task, i)
        response = assigned_agents[0].execute_task(user_input)
        response = response.replace("Final Answer:","").strip()
        self.memory.append(response)
        i += 1

    history = self.generate_history()
    print(json.dumps(history, indent=4))
    return history
```

#### Constraints:
- ✅ CAN: Add fallback agents
- ✅ CAN: Use TaskRevisionHelperAgent
- ✅ CAN: Combine multiple responses
- ✅ CAN: Add validation logic
- ✅ CAN: Implement dynamic scheduling
- ✅ CAN: While loop (but keep max_loops = 15)
- ❌ CANNOT: Remove max_loops safety cap
- ❌ CANNOT: Change memory persistence
- ❌ CANNOT: Modify generate_history()
- ❌ CANNOT: Change history structure

#### Example Enhancement - Option A (Fallback Strategy):
```python
def run(self):
    self.memory = []
    self.context_type = ContextType.SELECTED
    max_loops = 15
    i = 0
    
    while i < len(self.tasks) and i < max_loops:
        task = self.tasks[i]
        task_no = i + 1
        logger.info(f"Task {task_no}: {task.description}")
        
        assigned_agents = task.agents
        user_input = self._build_input(task, i)
        
        response = None
        agent_used = None
        
        # Try each agent until one succeeds
        for agent_idx, agent in enumerate(assigned_agents):
            try:
                logger.info(f"  Trying agent {agent_idx + 1}: {agent.name}")
                potential_response = agent.execute_task(user_input)
                
                # Validate response
                if potential_response and len(potential_response.strip()) > 10:
                    response = potential_response
                    agent_used = agent.name
                    logger.info(f"  ✅ Success with {agent.name}")
                    break
            except Exception as e:
                logger.warning(f"  ❌ Agent {agent.name} failed: {e}")
                continue
        
        # Fallback if no agent succeeded
        if response is None:
            logger.warning(f"  ⚠️  No agent succeeded for task {task_no}")
            response = "EXECUTION FAILED - Please review task or data"
            agent_used = "NONE"
        
        # Clean response
        response = response.replace("Final Answer:","").strip()
        self.memory.append(response)
        
        logger.info(f"  Agent used: {agent_used}")
        i += 1
    
    history = self.generate_history()
    print(json.dumps(history, indent=4))
    return history
```

#### Example Enhancement - Option B (Helper + Validation):
```python
def run(self):
    self.memory = []
    self.context_type = ContextType.SELECTED
    max_loops = 15
    i = 0
    
    # Initialize helper
    helper = TaskRevisionHelperAgent(llm=self.llm)
    
    while i < len(self.tasks) and i < max_loops:
        task = self.tasks[i]
        task_no = i + 1
        logger.info(f"Task {task_no}: {task.description}")
        
        assigned_agents = task.agents
        user_input = self._build_input(task, i)
        
        # OPTIONAL: Enhance input with helper
        enriched_input = helper.execute_task(user_input)
        
        # Execute with primary agent
        response = assigned_agents[0].execute_task(enriched_input)
        
        # Clean response
        response = response.replace("Final Answer:","").strip()
        
        # Optional: Validate response quality
        if len(response) < 20:
            logger.warning(f"Response too short for task {task_no} - consider retry")
        
        self.memory.append(response)
        i += 1
    
    history = self.generate_history()
    print(json.dumps(history, indent=4))
    return history
```

#### Example Enhancement - Option C (Multi-Agent Voting):
```python
def run(self):
    self.memory = []
    self.context_type = ContextType.SELECTED
    max_loops = 15
    i = 0
    
    while i < len(self.tasks) and i < max_loops:
        task = self.tasks[i]
        task_no = i + 1
        logger.info(f"Task {task_no}: {task.description}")
        
        assigned_agents = task.agents
        user_input = self._build_input(task, i)
        
        if len(assigned_agents) > 1:
            # Get multiple perspectives
            responses = []
            for agent_idx, agent in enumerate(assigned_agents):
                try:
                    resp = agent.execute_task(user_input)
                    responses.append(resp)
                    logger.info(f"  Agent {agent_idx+1} ({agent.name}): provided response")
                except Exception as e:
                    logger.warning(f"  Agent {agent_idx+1} ({agent.name}): failed - {e}")
                    continue
            
            # Combine responses
            if responses:
                response = f"CONSENSUS FROM {len(responses)} AGENTS:\n\n" + \
                          "\n---\n".join(responses)
            else:
                response = "ALL AGENTS FAILED"
        else:
            # Single agent
            response = assigned_agents[0].execute_task(user_input)
        
        # Clean response
        response = response.replace("Final Answer:","").strip()
        self.memory.append(response)
        i += 1
    
    history = self.generate_history()
    print(json.dumps(history, indent=4))
    return history
```

---

## 🎯 Summary Table

| Edit | Track | Component | Lines | Current Focus | Enhancement Ideas | Impact |
|------|-------|-----------|-------|----------------|-------------------|--------|
| 1 | 1 | Agent Formatting | 65-89 | Simple numbered list | Emojis, tags, grouping | Medium |
| 2 | 1 | Planning Prompt | 167-198 | Basic constraints | Domain strategy, examples, heuristics | HIGH ⭐⭐⭐ |
| 3 | 2 | Task Helper | 31-81 | NotImplementedError | Validation, enrichment, scoring | Medium |
| 4 | 2 | Execution Loop | 141-180 | Simple sequential | Fallbacks, voting, validation | HIGH ⭐⭐⭐ |

---

## 🚀 Implementation Priority

1. **Start with Edit 2** (Planning Prompt) - Highest impact, foundation for everything else
2. **Then Edit 4** (Execution Logic) - Robustness and resilience
3. **Then Edit 1** (Agent Formatting) - Polish and clarity
4. **Finally Edit 3** (Task Helper) - Refinement and validation

---

**Ready to edit? Copy one of the example enhancements and customize it for your approach! 🎯**
