from meta_agent.default_meta_agent import AgentHub


def test_agent_hub_methods(add_distractor_agent=False, agent_order_shift=4):
    # Create an instance of AgentHub
    agent_hub = AgentHub()

    # 1. Load default agents and tools
    print("Loading agents and tools...")
    agent_hub.load_default_agents()

    # 2. load distractor agents
    if add_distractor_agent:
        from meta_agent.agents.distractor_agents import load_prebuilt_agents

        prebuilt_agents = load_prebuilt_agents()
        for name, agent in prebuilt_agents.items():
            agent_hub.add_agent(name, agent, agent.get_examples())

    # 2. Check if specific agents and tools are loaded
    print("Checking loaded agents and tools...")
    print(f"Loaded agents: {list(agent_hub.agents.keys())}")
    assert "IoTAgent" in agent_hub.agents
    assert "TSFMAgent" in agent_hub.agents
    assert "JSONReader" in agent_hub.agents

    # 3. Show examples for a specific agent
    agent_hub.display_agents_and_examples()
    agent_hub.run(task="What IoT sites are available?", agent_order_shift=agent_order_shift)
    # agent_hub.run(task="Get the work order of equipment CU02013 for year 2017.")
    # agent_hub.run(task="List all failure modes of asset Chiller")
    # agent_hub.run(task="What types of time series analysis are supported?")


# Execute the test script
# test_agent_hub_methods()

# Run with distractor
test_agent_hub_methods(add_distractor_agent=True, agent_order_shift=4)
