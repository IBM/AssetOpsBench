from meta_agent.agents.pre_built_agents import load_prebuilt_agents
from meta_agent.tools.pre_built_tools import load_prebuilt_tools

def main():
    # Load pre-built tools and agents
    agents = load_prebuilt_agents()  # This will load the agents
    tools = load_prebuilt_tools()    # This will load the pre-built tools

    # Print the agents and tools (or call their base functions)
    print("Pre-built Agents:")
    for agent_name, agent in agents.items():
        print(f"{agent_name}: {agent}")

    print("\nPre-built Tools:")
    for tool_name, tool in tools.items():
        print(f"{tool_name}: {tool}")

if __name__ == "__main__":
    main()
