import json
import tempfile
from reactxen.utils.tool_description import get_tool_description
import json
from reactxen.agents.react.agents import ReactReflectAgent


def save_to_tmp(data, prefix):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".json", prefix=prefix
    ) as tmp_file:
        json.dump(data, tmp_file)
        print (tmp_file.name)
        return tmp_file.name  # Return file path as pointer


def getAgent(
    input,
    tools,
    inContext=None,
    llm_model_id=6,
    react_step=15,
    reflect_step=5,
    enable_agent_ask=False,
    debug=False,
    agent_style='ReActXen',
):

    reactXenAgent = None
    if agent_style == 'ReActXen':
        reactXenAgent = ReactReflectAgent(
            question=input,
            key="",
            # agent_prompt=tsfmagent_prompt,
            cbm_tools=tools,
            tool_desc=get_tool_description(tools),
            max_steps=react_step,
            react_llm_model_id=llm_model_id,
            reflect_llm_model_id=llm_model_id,
            # react_llm_model_id=8,
            react_example=inContext,
            # react_example="",
            num_reflect_iteration=reflect_step,
            handle_context_length_overflow=True,
            apply_loop_detection_check=True,
            log_structured_messages=True,
            early_stop=True,
            enable_agent_ask=enable_agent_ask,
            debug=debug,
        )

    return reactXenAgent
