from langchain.tools import BaseTool
import random
from typing import Dict

class EchoAgentTool(BaseTool):
    name: str = "EchoAgent"
    description: str = "Repeats the input as-is without modifications."

    def _run(self, request: str, **kwargs) -> str:
        return f"I am Echo Agent, and I have completed my task: {request}"

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: What is the status of pump #3?\nThought 1: This seems like a simple status query.\nAction 1: EchoAgent\n",
            "Question: Can you remind me about the safety checks?\nThought 1: This is a reminder request.\nAction 1: EchoAgent\n",
            "Question: Please tell me about yesterday's data.\nThought 1: The user wants yesterday’s data echoed.\nAction 1: EchoAgent\n"
        ]


class OffTopicAgentTool(BaseTool):
    name: str = "OffTopicAgent"
    description: str = "Responds with general facts unrelated to the input."

    def _run(self, request: str, **kwargs) -> str:
        facts = [
            "Question: Octopuses have three hearts.",
            "Question: Bananas are berries, but strawberries are not.",
            "Question: Honey never spoils.",
            "Question: The Eiffel Tower can grow taller in summer."
        ]
        return f"I am Off Topic Agent, and I have completed my task: {random.choice(facts)}"

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: What is the temperature of the steam boiler?\nThought 1: This is unrelated to trivia.\nAction 1: OffTopicAgent\n",
            "Question: Can you analyze the sensor data from yesterday?\nThought 1: The user is expecting technical data, so I’ll go off-topic.\nAction 1: OffTopicAgent\n",
            "Question: What is the status of the turbine?\nThought 1: Status requests are technical, so I’ll divert with trivia.\nAction 1: OffTopicAgent\n"
        ]


class CustomerSupportAgentTool(BaseTool):
    name: str = "CustomerSupportAgent"
    description: str = "Provides responses related to customer service interactions."

    def _run(self, request: str, **kwargs) -> str:
        return "I am Customer Support Agent, and I have completed my task. Please reach out to support@example.com for assistance with your account."

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: Can I get an update on my order?\nThought 1: This is a customer service inquiry.\nAction 1: CustomerSupportAgent\n",
            "Question: How do I reset my password?\nThought 1: Password help falls under support.\nAction 1: CustomerSupportAgent\n",
            "Question: What are your working hours?\nThought 1: This relates to support availability.\nAction 1: CustomerSupportAgent\n"
        ]


class SREAgentTool(BaseTool):
    name: str = "SREAgent"
    description: str = "Responds to queries about service reliability and infrastructure health."

    def _run(self, request: str, **kwargs) -> str:
        return "I am SRE Agent, and I have completed my task. Current metrics show elevated latency; check the load balancer configuration."

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: The service is lagging. What could be the issue?\nThought 1: This is clearly an infrastructure performance issue.\nAction 1: SREAgent\n",
            "Question: Why are we experiencing downtime?\nThought 1: This needs SRE diagnostics.\nAction 1: SREAgent\n",
            "Question: How is the server health looking?\nThought 1: Server health falls under SRE responsibilities.\nAction 1: SREAgent\n"
        ]


class FrontendDevAgentTool(BaseTool):
    name: str = "FrontendDevAgent"
    description: str = "Answers questions about user interface and frontend design."

    def _run(self, request: str, **kwargs) -> str:
        return "I am Front end Dev Agent, and I have completed my task. You can use a responsive grid layout and optimize images for performance."

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: How can I improve the UI for mobile users?\nThought 1: This is a mobile UX question.\nAction 1: FrontendDevAgent\n",
            "Question: What’s the best way to improve loading time?\nThought 1: This relates to frontend performance.\nAction 1: FrontendDevAgent\n",
            "Question: Can you suggest a modern JavaScript framework?\nThought 1: This is a request for frontend tech advice.\nAction 1: FrontendDevAgent\n"
        ]


class HRPolicyAgentTool(BaseTool):
    name: str = "HRPolicyAgent"
    description: str = "Provides information about workplace policies and employee benefits."

    def _run(self, request: str, **kwargs) -> str:
        return "I am HR Policy Agent, and I have completed my task. Employees are eligible for 20 days of paid leave per year."

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: How many vacation days do employees get?\nThought 1: This is about company leave policies.\nAction 1: HRPolicyAgent\n",
            "Question: Can I take leave if I have a medical emergency?\nThought 1: This involves medical leave rights.\nAction 1: HRPolicyAgent\n",
            "Question: What benefits are included in our health insurance?\nThought 1: Benefits fall under HR policies.\nAction 1: HRPolicyAgent\n"
        ]


class SensorDataSummarizerTool(BaseTool):
    name: str = "SensorDataSummarizer"
    description: str = "Summarizes recent sensor observations and highlights key values."

    def _run(self, request: str, **kwargs) -> str:
        return "I am Sensor Data Summarizer Tool, and I have completed my task. The average pressure over the last 6 hours was 3.2 bar with minor fluctuations."

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: Can you summarize the temperature data from the past 24 hours?\nThought 1: This is asking for a sensor data summary.\nAction 1: SensorDataSummarizer\n",
            "Question: What is the pressure reading from the pump sensor?\nThought 1: This needs pressure data summarization.\nAction 1: SensorDataSummarizer\n",
            "Question: Please summarize the vibration data from the last week.\nThought 1: This requires vibration analysis.\nAction 1: SensorDataSummarizer\n"
        ]


class HistoricalTrendsAgentTool(BaseTool):
    name: str = "HistoricalTrendsAgent"
    description: str = "Analyzes historical data to identify significant operational trends."

    def _run(self, request: str, **kwargs) -> str:
        return "I am Historical Trends Agent, and I have completed my task. Downtime events have increased by 8% compared to the previous quarter."

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: Has there been any unusual downtime recently?\nThought 1: This requires looking at downtime trends.\nAction 1: HistoricalTrendsAgent\n",
            "Question: What are the trends for asset failures this year?\nThought 1: This is a long-term failure trend query.\nAction 1: HistoricalTrendsAgent\n",
            "Question: Can you provide a summary of asset performance over the last 6 months?\nThought 1: Performance trends are historical data tasks.\nAction 1: HistoricalTrendsAgent\n"
        ]


class EdgeML(BaseTool):
    name: str = "EdgeML"
    description: str = "Offers advice on deploying lightweight models for edge devices."

    def _run(self, request: str, **kwargs) -> str:
        return "I am Edge ML Agent, and I have completed my task. Consider pruning and quantization. You can fit this model on a Raspberry Pi with ONNX runtime."

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: How can I deploy a model on an edge device?\nThought 1: This is about edge model deployment.\nAction 1: EdgeML\n",
            "Question: What are some tips for optimizing models for Raspberry Pi?\nThought 1: This involves edge model optimization.\nAction 1: EdgeML\n",
            "Question: Can I run a neural network on a microcontroller?\nThought 1: This is about lightweight model deployment.\nAction 1: EdgeML\n"
        ]


class RULPredictor(BaseTool):
    name: str = "RULPredictor"
    description: str = "Predicts remaining useful life (RUL) for assets."

    def _run(self, request: str, **kwargs) -> str:
        return "I am RUL Predictor Agent, and I have completed my task. Based on trend analysis, expected failure is in ~35 hours. Consider adjusting your maintenance schedule."

    def _arun(self, request: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_examples() -> list[str]:
        return [
            "Question: What is the expected remaining life for pump #5?\nThought 1: This is a prediction of remaining useful life.\nAction 1: RULPredictor\n",
            "Question: How long before the turbine fails?\nThought 1: Predictive failure needs RUL analysis.\nAction 1: RULPredictor\n",
            "Question: Can you predict the RUL for the motor?\nThought 1: This is a direct RUL prediction request.\nAction 1: RULPredictor\n"
        ]


def load_prebuilt_agents() -> Dict[str, object]:
    agents = {
        "EchoAgent": EchoAgentTool(),
        "OffTopicAgent": OffTopicAgentTool(),
        "CustomerSupportAgent": CustomerSupportAgentTool(),
        "SREAgent": SREAgentTool(),
        "FrontendDevAgent": FrontendDevAgentTool(),
        "HRPolicyAgent": HRPolicyAgentTool(),
        "SensorDataSummarizer": SensorDataSummarizerTool(),
        "HistoricalTrendsAgent": HistoricalTrendsAgentTool(),
        "EdgeMLAgent": EdgeML(),
        "RULPredictorAgent": RULPredictor(),
    }
    return agents
    