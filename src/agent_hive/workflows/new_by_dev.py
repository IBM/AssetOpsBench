from agent_hive.task import Task
from pydantic import Field
from typing import List
from agent_hive.enum import ContextType
import json
from agent_hive.workflows.base_workflow import Workflow
from reactxen.utils.model_inference import watsonx_llm
import re
from agent_hive.workflows.sequential import SequentialWorkflow
from agent_hive.agents.plan_reviewer_agent import PlanReviewerAgent
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

# =========================================================
# TODO: Participants can edit this section ONLY
# Add variable, dict. no more any import just any inline code
# =========================================================
# END OF EDITABLE SECTION


class NewPlanningWorkflow(Workflow):
    """
    Participant Template for Planning Review Workflow.
    ---------------------------------------------------
    📝 Instructions for participants:
    - Only modify the section marked with "TODO: Edit prompt here"
    - Do NOT change any workflow logic, agents, or execution components
    - Keep all retry, memory, and sequential execution intact
    """

    llm: str = Field(description="LLM used by the task planning.")

    def __init__(self, tasks: List[Task], llm: str):
        self.tasks = tasks
        self.memory = []
        self.max_memory = 10
        self.llm = llm
        self.max_retries = 5
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")
        if len(self.tasks) != 1:
            raise ValueError("Planning only supports one task")
        task = self.tasks[0]
        if task.agents is None or len(task.agents) < 1:
            raise ValueError("Task must have at least one agent")

    def run(self, enable_summarization=False):
        generated_steps = self.generate_steps()

        sequential_workflow = SequentialWorkflow(
            tasks=generated_steps, context_type=ContextType.SELECTED
        )

        return sequential_workflow.run()

    def generate_steps(self, save_plan=False, saved_plan_filename=""):
        task = self.tasks[0]
        agent_descriptions = ""

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

        # Extract metadata from task if available
        query_category = getattr(task, 'category', None)
        query_type = getattr(task, 'type', None)
        characteristic_form = getattr(task, 'characteristic_form', None)

        logger.info(f"[EDIT SECTION 1] Extracting metadata - Category: {query_category}, Type: {query_type}")

        for ii, aagent in enumerate(task.agents):
            agent_descriptions += f"\n({ii + 1}) 🤖 Agent name: {aagent.name}"
            agent_descriptions += f"\n📝 Agent description: {aagent.description}"
            logger.info(f"[EDIT SECTION 1] Processing agent {ii + 1}: {aagent.name}")
            if "task_examples" in aagent.__dict__ and aagent.task_examples:
                agent_descriptions += f"\n✅ Tasks that agent can solve:"
                for idx, task_example in enumerate(aagent.task_examples, start=1):
                    agent_descriptions += f"\n   {idx}. {task_example}"
            agent_descriptions += "\n"

        logger.info(f"[EDIT SECTION 1] Agent descriptions formatted with {len(task.agents)} agents")

        # =========================================================
        # END OF EDITABLE SECTION
        # 🚫 Participants should not modify code below this line
        # ❌ No new variables, functions, or workflow logic allowed
        # ✅ Only modify the section marked as TODO above
        # =========================================================

        prompt = self.get_prompt(task.description, agent_descriptions)
        logger.info(f"Plan Generation Prompt: \n{prompt}")
        llm_response = watsonx_llm(
            prompt, model_id=self.llm,
        )["generated_text"]
        logger.info(f"Plan: \n{llm_response}")

        final_plan = llm_response
        self.memory = []

        task_pattern = r"#Task\d+: (.+)"
        agent_pattern = r"#Agent\d+: (.+)"
        dependency_pattern = r"#Dependency\d+: (.+)"
        output_pattern = r"#ExpectedOutput\d+: (.+)"

        tasks = re.findall(task_pattern, final_plan)
        agents = re.findall(agent_pattern, final_plan)
        dependencies = re.findall(dependency_pattern, final_plan)
        outputs = re.findall(output_pattern, final_plan)

        if save_plan:
            if not saved_plan_filename.endswith(".txt"):
                saved_plan_filename += ".txt"

            saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
            with open(saved_plan_filename, "w") as f:
                f.write(saved_plan_text)

        planned_tasks = []
        for i in range(len(tasks)):
            task_description = tasks[i]
            if i == len(agents):
                break
            agent_name = agents[i]
            if i < len(dependencies):
                dependency = dependencies[i]
            else:
                dependency = "None"
            if i < len(outputs):
                expected_output = outputs[i]
            else:
                expected_output = ""

            selected_agent = None
            for agent in task.agents:
                if agent.name == agent_name:
                    selected_agent = agent
                    break
            if selected_agent is None:
                selected_agent = task.agents[0]

            if dependency != "None":
                numbers = re.findall(r"#S(\d+)", dependency)
                numbers = list(map(int, numbers))
                context = [planned_tasks[i - 1] for i in numbers]
            else:
                context = []

            a_task = Task(
                description=task_description,
                expected_output=expected_output,
                agents=[selected_agent],
                context=context,
            )
            planned_tasks.append(a_task)

        logger.info(f"Planned Tasks: \n{planned_tasks}")

        return planned_tasks

    def get_prompt(self, task_description, agent_descriptions):
        # =========================================================
        # TODO: Participants can edit this section ONLY
        # 🎨 Purpose: Improve prompt clarity, formatting, emojis, guidance
        # ✅ Allowed: Wording, structure, examples, emojis
        # ❌ Not allowed: Changing workflow, ReAct agent, Executor, or memory logic
        # =========================================================

        # Extract metadata from task if available
        task = self.tasks[0]
        query_category = getattr(task, 'category', None)
        query_type = getattr(task, 'type', None)
        characteristic_form = getattr(task, 'characteristic_form', None)

        logger.info(f"[EDIT SECTION 2] Building prompt with metadata - Category: {query_category}, Type: {query_type}")

        # Build query info section if metadata is available
        query_info = "### 2. Query Context & Guidance\nNo additional query information provided."
        if query_type or query_category or characteristic_form:
            query_info = "### 2. Query Context & Guidance\n"
            query_info += "Use this information to better understand the user's goal:\n"
            if query_type:
                query_info += f"- **Query Type:** {query_type}\n"
            if query_category:
                query_info += f"- **Query Category:** {query_category}\n"
            if characteristic_form:
                query_info += f"- **Expected Format:** {characteristic_form}\n"
            
            # Add explicit guidance linked to the metadata
            if query_type == "Inference Query":
                query_info += "-> **Action:** Your plan should focus on analysis and deriving insights, not just data retrieval."
            elif query_type == "Data Query":
                query_info += "-> **Action:** Your plan should focus on retrieving specific data values."
            elif query_type == "Anomaly Detection Query":
                query_info += "-> **Action:** Your plan must involve an agent capable of identifying outliers or abnormal patterns."
            
            logger.info(f"[EDIT SECTION 2] Query information section added to prompt")
        else:
            logger.info(f"[EDIT SECTION 2] No metadata available, using generic prompt")

        prompt = f"""
🚀 You are an expert AI planner. Your job is to create a step-by-step execution plan to solve a user's problem using *only* the provided agents.

## 1. Your Planning Strategy ##
1.  **Analyze the Goal:** First, read the "Problem to Solve" (Section 4) carefully.
2.  **Review Context:** Look at the "Query Context" (Section 2) for clues about the user's intent (e.g., 'Inference', 'Data Query').
3.  **Consult Agent Menu:** Review the "Available Agents" (Section 3). Pay close attention to their descriptions and "Tasks that agent can solve" examples.
4.  **Decompose the Problem:** Break the main "Problem to Solve" into a small number of logical sub-tasks (fewer than 5).
5.  **Assign Agents:** For *each* sub-task, select the *best* agent from the "Available Agents" list. The agent's name must be an *exact match*.
6.  **Set Dependencies:** Determine the data flow. If a task needs the output from a previous task, use its step number (e.g., #S1). The first task *must* have a dependency of 'None'.
7.  **Ensure a Final Answer:** The *last step* in your plan *must* be a task that synthesizes all previous results (e.g., #S1, #S2) and provides a complete, final answer to the user's "Problem to Solve".
8.  **Format the Output:** Present *only* the final plan in the required format. Do not add any other text, conversation, or explanation.

{query_info}

## 3. Available Agents (Your "Menu") ##
{agent_descriptions}

## 4. Problem to Solve (The "Goal") ##
{task_description}

## 5. Critical Rules & Output Format ##
Your response *must* follow these rules to avoid common failures:

⚠️ **1. Ensure a Final Answer:** The *last step* must synthesize all previous steps (e.g., `#Dependency<N>: #S1, #S2`) to fully answer the user's "Problem to Solve". A plan that only fetches data but doesn't present it is a failure.

⚠️ **2. Be Efficient (No Redundancy):** Do *not* create redundant tasks to fetch the same data. If data is retrieved in #S1, re-use it (via dependency) in #S3. Do not fetch it again.

⚠️ **3. Manage Data Flow (CRITICAL):** Think about data *compatibility*. When setting `#Dependency: #S1`, ensure the *output format* of #S1 matches the *input requirement* of #S2. Do not pipe a JSON file to an agent that expects plain text.

⚠️ **4. Write Actionable Sub-tasks:** Each `#Task<N>` description must be a clear, unambiguous, and *actionable* command. Use the agent's "Tasks that agent can solve" examples as a style guide.

⚠️ **5. Agent Constraint:** You *must* use *only* the agent names provided in Section 3. Do not invent agents.

⚠️ **6. Step Limit:** The plan *must* be fewer than 5 steps.

⚠️ **7. Format:** Your *entire* response *must* be *only* the plan in the format below. Do not add *any* other text (e.g., "Here is the plan:").

#Task<N>: <Describe the sub-task>
#Agent<N>: <Exact_agent_name_from_Section_3>
#Dependency<N>: <#S1, #S2, ... or None>
#ExpectedOutput<N>: <What this step will produce to be used by other steps or the final answer>

Output (your generated plan) ⬇️:
"""
        # =========================================================
        # End of participant editable section
        # =========================================================
        return prompt