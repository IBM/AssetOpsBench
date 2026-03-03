# Originally https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
# Just maintaining for comparision on where we started

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_ibm import WatsonxLLM
from langchain_core.tools import tool
from datetime import datetime


credentials = {
    "url": "",
    "apikey": "",
    "project_id": "",
}

param = {
    "decoding_method": "greedy",
    "temperature": 0,
    "min_new_tokens": 5,
    "max_new_tokens": 250,
}

@tool
def get_todays_date() -> str:
    """Get today's date in YYYY-MM-DD format."""
    date = datetime.now().strftime("%Y-%m-%d")
    return date


tools = [get_todays_date]

model = WatsonxLLM(
    model_id="ibm/granite-3-2-8b-instruct-preview-rc",
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=credentials.get("project_id"),
    params=param,
)

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, max_iterations=10, handle_parsing_errors=True
)
ans = agent_executor.invoke({"input": "hi"})
print(ans)
