🛠 Developer Guide: Preparing `execution_steps`, `execution_nodes`, and `execution_links` for Task Planning with Tools and Agents

This guide helps developers convert a user instruction, selected component definitions (`sampled_nodes`), and execution dependencies (`sampled_links`)
into structured formats usable by a multi-component system—where each node may be a traditional tool or an autonomous agent.

---

🎯 Objective

Given:
- A user instruction (natural language)

Produce:
- execution_steps: natural language execution plan
- execution_nodes: component calls with filled arguments
- execution_links: execution dependencies between components

---

1. `execution_steps`: Step-by-Step Plan

✅ Purpose:
A list of natural language instructions describing the intended execution order of components.

🧠 How to Create:
- One sentence per component (tool or agent).
- Follow the execution order from execution_links (topological sort).
- Describe the task and its goal using argument values.
- Use connectors like “First”, “Then”, “Once done”.

💡 Example:
- Step 1: Execute the software_management node to install 'example_software'
- Step 2: Then trigger the set_alarm node to set an alarm at '08:00'

---

2. `execution_nodes`: Component Invocation with Arguments

✅ Purpose:
Define a list of concrete calls to tools or agents with resolved argument values from the instruction.

🧠 How to Create:
- For each node in sampled_nodes, collect the required arguments.
- Extract values from the user instruction or infer default/fallback values.
- Format:
    {
        "task": "node_name",
        "arguments": [
            { "name": "arg_name", "value": "value" }
        ]
    }

💡 Example:
    {
        "task": "software_management",
        "arguments": [
            { "name": "software", "value": "example_software" },
            { "name": "instruction", "value": "install" }
        ]
    }

---

3. `execution_links`: Directed Dependencies Between Nodes

✅ Purpose:
Encodes execution order using source-target pairs, forming a directed acyclic graph.

🧠 How to Create:
Use sampled_links as-is. Each entry defines that the `target` must be executed after the `source`.

💡 Example:
    [
        { "source": "software_management", "target": "set_alarm" }
    ]

This implies: Run software_management first, then set_alarm.

---

🔁 Full Example

Instruction:
    Please install the 'example_software' and set an alarm for 08:00


```
sampled_links:
    [
        { "source": "software_management", "target": "set_alarm" }
    ]
```

```
execution_steps:
    [
        "Step 1: Execute the software_management node to install 'example_software'",
        "Step 2: Then trigger the set_alarm node to set an alarm at '08:00'"
    ]
```

execution_nodes:
    [
        {
            "task": "software_management",
            "arguments": [
                { "name": "software", "value": "example_software" },
                { "name": "instruction", "value": "install" }
            ]
        },
        {
            "task": "set_alarm",
            "arguments": [
                { "name": "time", "value": "08:00" }
            ]
        }
    ]

execution_links:
    [
        { "source": "software_management", "target": "set_alarm" }
    ]

---

🧠 Summary

| Field             | Description                                  |
|-------------------|----------------------------------------------|
| execution_steps   | Ordered, human-readable plan of action       |
| execution_nodes   | Fully-resolved invocations of tools/agents   |
| execution_links   | Execution order as source-target relationships |

Use this guide to plan, orchestrate, or visualize tasks in systems that mix tools, APIs, and autonomous agents.
