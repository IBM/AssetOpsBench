# 🎯 Quick Reference: What To Edit

## TL;DR - The 4 Editable Sections

### **TRACK 1: Planning** 📋

#### Edit 1️⃣: Agent Information Format
- **File:** `src/agent_hive/workflows/track1_planning.py`
- **Lines:** 65-89
- **What:** How agent names, descriptions, and task examples are formatted
- **Impact:** Medium - Helps LLM understand agent capabilities better
- **Difficulty:** Easy - Just formatting/structure
- **Time:** 30 minutes

```python
# CURRENT (SIMPLE):
agent_descriptions += f"\n({ii + 1}) Agent name: {aagent.name}"
agent_descriptions += f"\nAgent description: {aagent.description}"
# ... list task examples

# ENHANCED (BETTER):
agent_descriptions += f"\n🤖 [{ii + 1}] **{aagent.name}**"
agent_descriptions += f"\n   📝 {aagent.description}"
agent_descriptions += f"\n   ✅ Capabilities:"
# ... better formatted examples
```

**Ideas to try:**
- Add emojis for visual distinction
- Group capabilities by category
- Add tool counts
- Better example formatting
- Add domain tags

---

#### Edit 2️⃣: Planning Prompt
- **File:** `src/agent_hive/workflows/track1_planning.py`
- **Lines:** 167-198
- **What:** The system prompt that tells LLM how to decompose tasks
- **Impact:** HIGH ⭐⭐⭐ - Biggest impact on plan quality
- **Difficulty:** Medium - Requires prompt engineering skills
- **Time:** 1-2 hours

```python
# CURRENT:
prompt = f"""
🚀 You are an AI assistant creating plans...
- Only use agents listed below
- Produce a plan with fewer than 5 steps
- Include Task, Agent, Dependency, ExpectedOutput

{agent_descriptions}

## Problem: {task_description}
"""

# ENHANCED:
prompt = f"""
🏭 You are an Industrial Ops AI creating maintenance action plans.

🎯 STRATEGY:
1. For maintenance: Data Collection → Analysis → Action
2. Verify data completeness before analysis
3. Match sensors to failure modes

📋 CRITICAL FORMAT (MUST follow exactly):
#Task<N>: specific, actionable task
#Agent<N>: exact agent name
#Dependency<N>: None or #S1, #S2
#ExpectedOutput<N>: expected result format

{agent_descriptions}

PROBLEM: {task_description}

Create an EFFICIENT plan considering:
- Temporal dependencies
- Data reusability
- Agent specialization
"""
```

**Ideas to try:**
- Add domain-specific heuristics
- Provide step-by-step reasoning strategy
- Add quality criteria
- Specify format more strictly
- Add examples of good plans
- Emphasize data validation first

---

### **TRACK 2: Execution** ⚙️

#### Edit 1️⃣: Task Revision Helper
- **File:** `src/agent_hive/workflows/track2_execution.py`
- **Lines:** 31-81 (implement `execute_task()` method)
- **What:** Validate and enrich task inputs before sending to agents
- **Impact:** Medium - Improves input quality to agents
- **Difficulty:** Medium - Need LLM knowledge
- **Time:** 1 hour

```python
# CURRENT:
def execute_task(self, task_input: str) -> str:
    raise NotImplementedError("Participants must implement this method.")

# IMPLEMENT:
def execute_task(self, task_input: str) -> str:
    """Validate and enrich task input."""
    
    # Check clarity
    issues = []
    if len(task_input.split()) < 5:
        issues.append("Task too brief")
    if "?" not in task_input:
        issues.append("No clear question")
    
    # Enrich with context
    enriched = f"""
ORIGINAL TASK: {task_input}

DOMAIN CONTEXT:
- Area: Industrial Maintenance
- Data types: Sensors, logs, time-series
- Expected tools: IoT, TSFM, FMSR

QUALITY ASSESSMENT:
{chr(10).join(f"❌ {i}" for i in issues) if issues else "✅ Clear and complete"}

REFINED TASK FOR AGENT:
{task_input}

When solving, prioritize:
1. Data validation
2. Anomaly detection
3. Root cause analysis
"""
    return enriched
```

**Ideas to try:**
- Grammar/clarity checking
- Parameter extraction
- Domain context addition
- Metadata tagging
- Quality scoring
- Multiple interpretations

---

#### Edit 2️⃣: Execution Logic
- **File:** `src/agent_hive/workflows/track2_execution.py`
- **Lines:** 141-180 (inside `run()` method)
- **What:** How tasks are executed and responses processed
- **Impact:** HIGH ⭐⭐⭐ - Affects execution quality and robustness
- **Difficulty:** Hard - Complex logic required
- **Time:** 2-3 hours

```python
# CURRENT:
self.context_type = ContextType.SELECTED
max_loops = 15
i = 0
while i < len(self.tasks) and i < max_loops:
    task = self.tasks[i]
    user_input = self._build_input(task, i)
    response = assigned_agents[0].execute_task(user_input)
    response = response.replace("Final Answer:","").strip()
    self.memory.append(response)
    i += 1

# ENHANCED:
# Option 1: Add helper validation
self.context_type = ContextType.SELECTED
max_loops = 15
i = 0
helper = TaskRevisionHelperAgent(llm=self.llm)

while i < len(self.tasks) and i < max_loops:
    task = self.tasks[i]
    user_input = self._build_input(task, i)
    
    # Enhance input with helper
    enriched_input = helper.execute_task(user_input)
    
    # Execute with primary agent
    response = assigned_agents[0].execute_task(enriched_input)
    
    # Clean and store
    response = response.replace("Final Answer:","").strip()
    self.memory.append(response)
    i += 1

# Option 2: Add fallback strategy
while i < len(self.tasks) and i < max_loops:
    task = self.tasks[i]
    user_input = self._build_input(task, i)
    
    response = None
    for attempt, agent in enumerate(assigned_agents):
        try:
            response = agent.execute_task(user_input)
            if response and len(response) > 20:
                break
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            continue
    
    if response is None:
        response = "EXECUTION FAILED"
    
    response = response.replace("Final Answer:","").strip()
    self.memory.append(response)
    i += 1

# Option 3: Combine multiple agent responses
while i < len(self.tasks) and i < max_loops:
    task = self.tasks[i]
    user_input = self._build_input(task, i)
    
    if len(assigned_agents) > 1:
        # Get multiple perspectives
        responses = [agent.execute_task(user_input) for agent in assigned_agents]
        # Combine (e.g., vote, merge, pick best)
        response = " | ".join(responses)
    else:
        response = assigned_agents[0].execute_task(user_input)
    
    response = response.replace("Final Answer:","").strip()
    self.memory.append(response)
    i += 1
```

**Ideas to try:**
- Multi-agent voting/consensus
- Fallback agents (try secondary if primary fails)
- Response quality checking
- Helper agent integration
- Conditional task skipping
- Dynamic agent selection
- Response merging/aggregation
- Retry logic with backoff

---

## 📊 Comparison: Impact vs Difficulty

```
Impact   │
High     │  ⭐ Edit 2 (Planning Prompt)     ⭐ Edit 2 (Exec Logic)
         │     Track 1 / Medium Effort      Track 2 / Hard Effort
         │
Medium   │  Edit 1 (Agent Format)          Edit 1 (Task Helper)
         │  Track 1 / Easy Effort          Track 2 / Medium Effort
         │
Low      │
         └─────────────────────────────────────────────────────
             Easy              Medium              Hard
                        Difficulty
```

**Strategic advice:**
1. **Start with:** Track 1 Edit 2 (Planning Prompt) - Highest ROI
2. **Then do:** Track 2 Edit 2 (Execution Logic) - Robustness
3. **Polish with:** Edit 1's (Formatting + Validation)

---

## 🔍 How to Test Your Changes

### Test One Edit:

```bash
# Test Track 1 Planning
cd /Users/srutanik/AssetOpsBench_CODS/benchmark/cods_track1

# Run on single scenario
python run_track_1.py --utterance_ids "1" --generate_steps_only False

# Check result
cat /home/track1_result/trajectory/Q_1_trajectory.json | jq '.'

# Analyze failure modes
cd /Users/srutanik/AssetOpsBench_CODS/src/TrajFM
python failure_mode_extractor.py \
    --traj_directory "/home/track1_result/trajectory" \
    --summary_dir "./summary"

# View summary
cat ./summary/failure_modes_clustered.csv
```

### Test Track 2:

```bash
cd /Users/srutanik/AssetOpsBench_CODS/benchmark/cods_track2

python run_track_2.py --utterance_ids "1" --generate_steps_only False

# Same analysis as above
```

### Compare Results:

```bash
# Before your edit
python run_track_1.py --utterance_ids "1,2,3"
cp -r /home/track1_result/trajectory /tmp/baseline_trajectory

# After your edit
python run_track_1.py --utterance_ids "1,2,3"

# Compare
diff /tmp/baseline_trajectory/Q_1_trajectory.json /home/track1_result/trajectory/Q_1_trajectory.json

# Metrics
wc -c /tmp/baseline_trajectory/*.json
wc -c /home/track1_result/trajectory/*.json
```

---

## 📋 Checklist Before Submission

- [ ] Track 1 Edit 1: Agent info formatting improved
- [ ] Track 1 Edit 2: Planning prompt enhanced
- [ ] Tested Track 1: `python run_track_1.py --utterance_ids "1,5,10"`
- [ ] Track 2 Edit 1: TaskRevisionHelperAgent implemented
- [ ] Track 2 Edit 2: Execution logic enhanced
- [ ] Tested Track 2: `python run_track_2.py --utterance_ids "1,5,10"`
- [ ] Ran failure mode analysis on both tracks
- [ ] Compared baseline vs. your version (trajectories)
- [ ] No edits outside marked sections
- [ ] All regex patterns intact (#Task, #Agent, #Dependency, #ExpectedOutput)
- [ ] Safety caps in place (15 loops, <5 steps)
- [ ] Code follows existing style
- [ ] No new imports or dependencies
- [ ] Ready to submit

---

## 🆘 Common Mistakes to Avoid

❌ **Don't:**
- Edit outside the marked TODO sections
- Change workflow orchestration logic
- Modify agent or tool implementations
- Add new imports
- Remove safety constraints (max loops, max steps)
- Change the regex patterns used for plan parsing
- Modify base classes (BaseAgent, BaseWorkflow, etc.)
- Edit run_track_1.py or run_track_2.py

✅ **Do:**
- Focus on prompt engineering (Track 1 Edit 2)
- Add domain-specific guidance
- Improve clarity and structure
- Use existing imports only
- Test frequently
- Compare before/after results
- Document your changes in comments

---

## 💡 Pro Tips

1. **Test incrementally:** Edit one section, test, then move to next
2. **Use diff tools:** Compare trajectories to see impact
3. **Failure analysis:** TrajFM shows you what went wrong
4. **Prompt engineering matters:** Invest time here
5. **Domain knowledge helps:** Understand industrial maintenance domain
6. **Few-shots are powerful:** Look at agent examples in tool files
7. **Context matters:** In Track 2, design execution to maximize info reuse
8. **Logging is your friend:** Use logger to debug

---

## 🚀 Next Steps

1. **Read the full guides:**
   - `COMPETITION_EDITABLE_SECTIONS.md` - Detailed explanation
   - `CONNECTION_MAP.md` - Visual flow diagrams

2. **Examine your editable code:**
   - Open `src/agent_hive/workflows/track1_planning.py`
   - Find lines 65-89 and 167-198
   - Study current implementation

3. **Review example agents:**
   - `src/meta_agent/agents/IoT/IoTAgentFewShots.py`
   - `src/meta_agent/agents/FMSR/FMSRAgentFewShots.py`
   - Understand what each agent can do

4. **Start with Track 1 Edit 2:**
   - Most impactful
   - Highest ROI
   - Medium difficulty

5. **Test and iterate:**
   - Run baseline
   - Make edit
   - Test
   - Compare results
   - Repeat

**Good luck! 🎯**
