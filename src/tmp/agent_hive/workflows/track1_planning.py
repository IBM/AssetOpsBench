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
        
        # =========================================================
        # TODO: Participants can edit this section ONLY
        # 🎨 Purpose: Customize LLM response post-processing
        # ❌ Not allowed: 
        #     - Modify workflow execution
        #     - Replace the base ReAct agent or Executor or Task
        #     - Change memory or retry logic
        # =========================================================
        
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

        # =========================================================
        # END OF EDITABLE SECTION
        # =========================================================

        return planned_tasks

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
