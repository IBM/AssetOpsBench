system_prompt = """
You will be provided with a multiagent system trace. Your task is to analyze the system behavior to detect any inefficiencies or failure modes.

Instructions:
- Review the trace for failures or inefficiencies based on the predefined list below.
- Only mark a failure mode as true if you can clearly identify an instance of it in the trace.
- For any detected failure, briefly explain it in the 'summary' field.
- Indicate whether the task is completed or not using a boolean.
- Answer all failure modes explicitly as true, or false.
- Optionally, suggest up to two new failure modes not on the predefined list.

Return your output as a valid JSON object only — do not include any text or explanation outside the JSON.

@@
{{
  "summary": "<One-sentence summary of problems with inefficiencies or failure modes>",
  "task_completed": <true | false>,
  "failure_modes": {{
    "1.1 Disobey Task Specification": <true | false>,
    "1.2 Disobey Role Specification": <true | false>,
    "1.3 Step Repetition": <true | false>,
    "1.4 Loss of Conversation History": <true | false>,
    "1.5 Unaware of Termination Conditions": <true | false>,
    "2.1 Conversation Reset": <true | false>,
    "2.2 Fail to Ask for Clarification": <true | false>,
    "2.3 Task Derailment": <true | false>,
    "2.4 Information Withholding": <true | false>,
    "2.5 Ignored Other Agent's Input": <true | false>,
    "2.6 Action-Reasoning Mismatch": <true | false>,
    "3.1 Premature Termination": <true | false>,
    "3.2 No or Incorrect Verification": <true | false>,
    "3.3 Weak Verification": <true | false>
  }},
  "additional_failure_modes": [
    {{
      "title": "<Title of new failure mode>",
      "description": "<1–2 sentence description with evidence from the trace>"
    }},
    {{
      "title": "<Title of new failure mode>",
      "description": "<1–2 sentence description with evidence from the trace>"
    }}
  ]
}}
@@

If no new failure modes are found, return an empty array for "additional_failure_modes".

Here is the trace:
{trace}

Below are the definitions of the known failure modes and inefficiencies:

1.1 **Disobey Task Specification**:
This error occurs when an agent or system fails to adhere to specified constraints, guidelines, or requirements associated with a particular task. Non-compliance can result from unclear, incomplete, or ambiguous instructions provided by the user, system prompts, or task descriptions. It may also arise from an agent's inadequate ability to interpret or apply constraints effectively. Consequences include incorrect or suboptimal outputs, reduced system performance, and increased resource consumption.

1.2 **Disobey Role Specification**:
Failure to adhere to the defined responsibilities and constraints of an assigned role, potentially leading to an agent behaving like another.

1.3 **Step Repetition**:
This error occurs when an agent or system unnecessarily repeats a task or phase that has already been completed. Redundancy can result from inadequate state or context tracking, inefficient workflow management, unclear instructions, or failure to recognize completed tasks.

1.4 **Loss of Conversation History**:
Unexpected context truncation, disregarding recent interactions, and reverting to a prior state in the conversation, causing loss of critical context.

1.5 **Unaware of Termination Conditions**:
This error occurs when an agent or system fails to recognize criteria designed to trigger the termination of an interaction or task. Oversight can lead to unnecessary actions, wasted resources, or incorrect behavior beyond the intended task.

2.1 **Conversation Reset**:
An unexpected or unwarranted restarting of the dialogue, potentially losing context and progress in the interaction.

2.2 **Fail to Ask for Clarification**:
Failure to request additional information when faced with unclear or incomplete data, which can result in incorrect actions.

2.3 **Task Derailment**:
Deviation from the intended task or objective, leading to irrelevant or unproductive actions.

2.4 **Information Withholding**:
This occurs when an agent possesses critical information but fails to share it with other agents or system components, which may impair the system’s overall operation or decision-making.

2.5 **Ignored Other Agent's Input**:
When an agent fails to consider or properly act on suggestions from other agents, leading to poor decisions or stalled progress.

2.6 **Action-Reasoning Mismatch**:
This happens when there’s a discrepancy between an agent’s reasoning and its actual actions, leading to unintended or counterproductive behavior.

3.1 **Premature Termination**:
Ending a task or conversation before the necessary information has been exchanged or objectives fully met.

3.2 **No or Incorrect Verification**:
The failure to properly verify task outcomes or system outputs, potentially allowing errors to propagate undetected.

3.3 **Weak Verification**:
Verification mechanisms that are insufficiently rigorous, potentially missing subtle errors or inconsistencies.

Here are example instances of those failure modes to help guide your analysis:

Example of "Step Repetition":
In the following trace, the Planner repeats the same thought twice in the workflow:
Planner's Response: Thought: To address this issue, we need to understand the root cause of the 'Line3D' object not having the '_verts3d' attribute...
Planner's Response: Thought: To address this issue, we need to understand the root cause of the 'Line3D' object not having the '_verts3d' attribute...

Example of "Unaware of Termination Conditions":
In this trace, two agents continuously repeat the same instructions despite missing information that prevents the task from progressing:
User: "Could you provide either the total length of the ribbon or the ribbon length used for each bow?"
Agent: "Continue. Please keep solving the problem until you need to query..."
User: "I'm sorry, but I really need more information to solve this problem."
Agent: "Continue. Please keep solving the problem until you need to query..."

Example of "No or Incorrect Verification":
In the following case, verification steps failed to detect an error in the game generation process:
Error: The file 'ship.bmp' was not found in the directory /Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911.
Traceback (most recent call last):
  File "/Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911/main.py", line 31, in <module>
    run_game()
  File "/Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911/main.py", line 22, in run_game
    gf.create_fleet(ai_settings, screen, aliens)
  File "/Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911/game_functions.py", line 64, in create_fleet
    alien = Alien(ai_settings, screen)
  File "/Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911/alien.py", line 13, in __init__
    self.image = pygame.image.load('alien.bmp')
FileNotFoundError: No file 'alien.bmp' found in working directory '/Users/*/Documents/*/ChatDev/'.

Example of "Action-Reasoning Mismatch":
In the following trace, the agent makes a statement that contradicts its previous reasoning:
Agent: "Note that the `_add_prefix_for_feature_names_out` method is not explicitly shown in the code snippet, but it is mentioned in the context of the `_iter` method."
[...]
However, earlier in the trace, the agent had already shown the full implementation of the `_add_prefix_for_feature_names_out` method, creating a mismatch between its reasoning and the actions taken.
"""