from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Union

class ModelStats(BaseModel):
    """Standardized metrics for token usage and API calls at any level."""
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0
    total_cost: float = 0.0
    instance_cost: float = 0.0

class TrajectoryStep(BaseModel):
    """
    A single unit of interaction. If this step executes a sub-agent, 
    the observation and step_stats capture that nested execution.
    """
    step_n: Optional[int] = None
    thought: str = ""
    action: str = ""
    action_input: Union[str, Dict[str, Any]] = ""
    
    # Observation can be a string, structured JSON, or a nested AgentRun
    observation: Union[str, Dict[str, Any], "AgentRun"] = ""
    
    # Execution State and Feedback
    state: Optional[str] = None
    is_loop_detected: Optional[str] = None
    llm_error: bool = False
    additional_scratchpad_feedback: str = ""
    
    # Metrics for this specific step to allow for total calculation
    step_stats: Optional[ModelStats] = None

class AgentRun(BaseModel):
    """
    The standardized container for an AssetOpsBench execution.
    Optimized for recursive analysis and aggregate metric calculation.
    """
    type: str = "AssetOpsAgent"
    task: str
    environment: str = "MCP"
    system_prompt: str
    demonstration: str
    scratchpad: str
    endstate: str
    final_answer: Optional[str] = None
    
    # The primary list of parsed reasoning steps
    trajectory: List[TrajectoryStep] = Field(default_factory=list)
    
    # The flat interaction history for prompt reconstruction
    history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Global metrics for the entire run
    info: Dict[str, ModelStats] = Field(default_factory=dict)

# Rebuild the recursive reference for nested AgentRuns
AgentRun.model_rebuild()