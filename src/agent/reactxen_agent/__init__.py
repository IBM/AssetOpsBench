"""ReActXen agent runner subpackage.

Wraps :class:`~agent.stirrup_agent.StirrupAgentRunner` as the ReAct core and
adds the ReActXen agent-family loop: a self-ask tool, a Review/Critique agent,
and a Reflect agent, bounded at three rounds.

The loop never scores.  Pass/fail comes from the benchmark's program-computed
reference; Review only decides whether to spend another round.
"""

from .models import ReActXenTrajectory, ReviewVerdict, RoundRecord
from .runner import ReActXenAgentRunner

__all__ = [
    "ReActXenAgentRunner",
    "ReActXenTrajectory",
    "ReviewVerdict",
    "RoundRecord",
]
