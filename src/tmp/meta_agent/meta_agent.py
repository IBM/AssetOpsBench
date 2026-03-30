from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
from termcolor import colored
from reactxen.agents.react.agents import ReflexionStrategy, ReactReflectAgent
from datetime import datetime
from meta_agent.utils import getAgent

class MetaAgent(ABC):
    def __init__(self):
        # Registries for agents/tools and their examples
        self.agents: Dict[str, object] = {}
        self.examples: Dict[str, Any] = {}

        # Load default agents/tools
        self.load_default_agents()

    @abstractmethod
    def load_default_agents(self) -> None:
        """
        Load default agents/tools into the system.
        Must be implemented by subclasses.
        """
        pass

    def add_agent(self, name: str, agent: object, examples: Any = None) -> None:
        """
        Add a new agent/tool manually and register its examples.

        Args:
            name (str): Unique name for the agent/tool.
            agent (object): The agent/tool instance.
            examples (Any, optional): Usage examples or documentation. Can be:
                - A string (description or single example)
                - A list of strings (multiple examples)
                - A callable (function that returns examples)
        """
        self.agents[name] = agent

        # Process the examples based on their type
        if callable(examples):
            # If examples is callable, call it to fetch examples
            self.examples[name] = examples()
        elif isinstance(examples, list):
            # If examples is a list of strings, store it as is
            self.examples[name] = examples
        elif isinstance(examples, str):
            # If examples is a single string, store it as is
            self.examples[name] = [examples]
        elif examples is None:
            self.examples[name] = []
        else:
            raise TypeError(
                "Examples must be a string, list of strings, or a callable function."
            )

    def show_examples(self, name: str = None) -> Any:
        """
        Show examples for a specific agent, or all examples if no name is provided.

        Args:
            name (str, optional): Name of the agent/tool.

        Returns:
            Examples for the specified agent, or all examples.
        """
        if name:
            return self.examples.get(name, f"No examples available for '{name}'.")
        return self.examples

    def list_agents(self) -> Dict[str, object]:
        """
        List all the available agents/tools in the system.

        Returns:
            Dict: Dictionary of agent names and their corresponding instances.
        """
        return self.agents

    def merge_all_examples(self) -> None:
        """Merge all example values into one consolidated example."""
        merged_examples = []

        # Iterate through all the agents in examples and merge their values
        for agent_name, examples in self.examples.items():
            if isinstance(examples, list):
                merged_examples.extend(examples)  # Merge list of examples
            elif isinstance(examples, str):
                merged_examples.append(examples)  # Add string example
            elif callable(examples):
                # If examples is callable, call it to get the examples and merge
                merged_examples.append(examples())  # Call and add result
            else:
                merged_examples.append("No examples available.")

        return "\n".join(merged_examples)

    def get_agents_list(self, shift: int = 0):
        agent_list = list(self.agents.values())
        n = len(agent_list)
        shift = shift % n  # Normalize shift
        return agent_list[shift:] + agent_list[:shift]  # Cyclic left shift

    def run(
        self,
        task,
        model_id=8,
        outer_loop_step=5,
        inner_loop_step=10,
        agent_style="ReActXen",
        in_context_examples=True,
        enable_distractor_agent=False,
        agent_order_shift=0,
    ) -> Any:
        """
        Run a specified agent/tool with given keyword arguments.

        Args:
            agent_name (str): The name of the agent/tool to run.
            kwargs: Keyword arguments to pass to the agent/tool.

        Returns:
            The result from running the agent/tool.
        """
        reflect = getAgent(
            task,
            self.get_agents_list(shift=agent_order_shift),
            inContext=self.merge_all_examples(),
            llm_model_id=model_id,
            react_step=inner_loop_step,
            reflect_step=outer_loop_step,
            enable_agent_ask=False,
            debug=True,
            agent_style=agent_style,
            #in_context_examples=in_context_examples,
            #enable_distractor_agent=enable_distractor_agent,
        )

        startTime = datetime.now()
        reflexion_strategy = ReflexionStrategy.REFLEXION
        reaction = reflect.run(reflect_strategy=reflexion_strategy)

        endTime = datetime.now()
        runTime = endTime - startTime
        runMinutes = runTime.total_seconds() / 60.0

        print("run minutes =", runMinutes)

        return reflect

    def display_agents_and_examples(self):
        """Display all agents and their examples with color formatting."""

        # Display a dashed line for separation
        print(colored("=" * 50, "cyan"))
        print(colored("Available agents in the system:", "green"))

        # Display the agents in the system
        for agent_name in self.agents.keys():
            print(colored(f"- {agent_name}", "cyan"))

        # Add a separator for clarity
        print(colored("-" * 50, "cyan"))

        # Display examples section header
        print(colored("\nExamples for each agent:", "yellow"))

        # Display examples for each agent
        for agent_name, examples in self.examples.items():
            print(colored(f"\nAgent: {agent_name}", "magenta"))
            print(colored("-" * 30, "magenta"))  # Add dashed line after agent name

            # Check if examples is a list, string, or callable
            if isinstance(examples, list):
                for example in examples:
                    print(colored(f"Example: {example}", "yellow"))
            elif isinstance(examples, str):
                print(colored(f"Example: {examples}", "yellow"))
            elif callable(examples):
                # If examples is callable, call it and print the result
                result = examples()  # Call the function to get examples
                print(colored(f"Example: {result}", "yellow"))
            else:
                print(colored("No examples available.", "yellow"))

            # Add a dashed line for separation between agents
            print(colored("-" * 50, "cyan"))
