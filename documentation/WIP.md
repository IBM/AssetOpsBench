"# 🛠 Developer Guide: Preparing `planning_steps`, `execution_steps`, and `execution_links` for Task Planning with Tools and Agents

This guide explains how to transform a user instruction, selected components (`sampled_nodes`), and their dependencies (`sampled_links`) into a structured plan for execution in a multi-component system—where each node can be a tool or an autonomous agent.

---
## 🎯 Objective
Given:  
1. A user instruction in natural language.  
2. Component definitions and dependencies.  

Produce:  
- **`planning_steps`**: Human-readable, ordered plan.  
- **`execution_steps`**: Machine-readable list of action objects with arguments.  
- **`execution_links`**: Directed dependencies between steps.

---

## 1. `planning_steps` — Natural Language Execution Plan

A sequence of simple sentences that describe what will happen and in what order.  
- One sentence per action.  
- Mention argument values when known.  

**Example**:

\"1. First call action sites with no parameters\"  
\"2. Then run IoT node to collect data\"

---

## 2. `execution_steps` — Component Calls with Arguments

This is a list of objects, each defining one action.  

Each object contains:  
- `name`: Unique step identifier.  
- `action`: Tool or agent name.  
- `arguments`: List of `{ name, value, dynamic?, source?, transient? }`.

**Static value example**:

```json
{
  "name": "Ioi1",
  "action": "IoT",
  "arguments": [
    {
      "name": "threshold",
      "value": 0.8
    }
  ]
}
```

#### Dynamic Argument Handling

Some arguments cannot be fixed at planning time because they change per execution or depend on the output of another step.  

- Add `"dynamic": true` to mark them as unresolved until runtime.  
- If value comes from another step’s output: set `"source": "<step_name>:<output_name>"`.  
- If value is computed at runtime (e.g., current date): `"source": "runtime"`.  
- If value is unique for each run (e.g., generated file IDs): also add `"transient": true`.

**Example**:
```json
{
  "name": "file_id",
  "value": "",
  "dynamic": true,
  "transient": true,
  "source": "Ioi1:file_id"
}
```

---

## 3. `execution_links` — Directed Dependencies

A list of `{ \"source\": step_name, \"target\": step_name }` pairs defining execution order.

**Example**:

```json
[
  { "source": "Ioi1", "target": "FMSR1" }
]
```
Means: `Ioi1` runs before `FMSR1`.

---

## 5. Full Example (AHU Merge)

**Instruction**:  
> For AHU1 and AHU2, get site files and merge them **`planning_steps`**: 
\"1. Retrieve site file for AHU1\"  
\"2. Retrieve site file for AHU2\"  
\"3. Merge the two files\"

**`execution_steps`**:
```json
[
  {
    "name": "site_AHU1",
    "action": "site",
    "arguments": [
      { "name": "ahu_id", "value": "ahu1", "dynamic": false }
    ]
  },
  {
    "name": "site_AHU2",
    "action": "site",
    "arguments": [
      { "name": "ahu_id", "value": "ahu2", "dynamic": false }
    ]
  },
  {
    "name": "file_merge",
    "action": "filemerge",
    "arguments": [
      { "name": "file1", "value": "", "dynamic": true, "source": "site_AHU1:file_id", "transient": true },
      { "name": "file2", "value": "", "dynamic": true, "source": "site_AHU2:file_id", "transient": true }
    ]
  }
]
