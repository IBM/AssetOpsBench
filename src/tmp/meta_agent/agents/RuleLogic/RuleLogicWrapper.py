import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import json

from nl2anomaliesrules.agents.rule_agent import getRuleLogicAgent
#from 

logger: logging.Logger = logging.getLogger(__name__)

MAX_STEPS = 15

class ReviewMessage:
    status: str
    reasoning: str
    suggestions: str

class RuleLogicAgentResponse:
    answer: str
    review: ReviewMessage
    reflection: str
    metric: {}
    trajectory: {}

class RuleLogicAgentFunctions:

    def request(self, request: str, parent_agent=None, parent_model_id=0):

        rulelogicAgent = getRuleLogicAgent(question=request, model_id=parent_model_id) # , react_step=2, reflect_step=2)
        if parent_agent:
            rulelogicAgent.parent_agent = parent_agent
        reaction = rulelogicAgent.run()

        # ideally we like to return 
        """
            return these example
            {'final_result': ans,
             'status': ,
             'reflexion': most_recent_reflexion/None,
             'review': most recent review/None,
             'summary': I am TSFM Agent, and I completed my task, the status field denote the 
             status, i also get review agent whose suggestion is included into review field aetc. '
            }
        """

        # print('****** REACTION =', json.dumps(reaction, indent=2))
        # print('******* REFLECTION =', json.dumps(tsfmAgent.reflections, indent=2))
        # print('********** tsfmAgent.answer =', tsfmAgent.answer)

        answer = 'Agent failed to generate final answer'
        if rulelogicAgent.answer.strip() != '':
            answer = rulelogicAgent.answer

        reflection = 'None'
        if len(rulelogicAgent.reflections) > 0 and rulelogicAgent.reflections[-1].strip() != '':
            reflection = rulelogicAgent.reflections[0]

        retval = RuleLogicAgentResponse()
        retval.answer = answer
        retval.review = reaction
        retval.reflection = reflection
        retval.metric = rulelogicAgent.metric
        retval.trajectory = rulelogicAgent.trajectory

        return retval
