
import os
from crewai_tools import tool
from interpreter import interpreter

@tool("Code Execution Tool")
def open_interpreter_tool(command: str) -> str:
    """This tool will take a prompt and execute code to satisfy the request.

    Args:
        command (str): The command to be executed

    Returns:
        str: The output from the command
    """
    interpreter.anonymized_telemetry = False
    interpreter.offline = True # Disables online features like Open Procedures
    interpreter.llm.model = "openai/openhermes" # Tells OI to send messages in OpenAI's format
    interpreter.llm.api_key = "fake_key" # LiteLLM, which we use to talk to LM Studio, requires this
    interpreter.llm.api_base = "http://localhost:11434/v1" # Point this at any OpenAI compatible server
    interpreter.auto_run = True
    interpreter.llm.max_tokens = 1000
    interpreter.llm.context_window = 3000
    interpreter.llm.temperature = 0 # Set to 0 for no randomness
    interpreter.debug=False
    interpreter.chat(command, display=False)
    output = interpreter.messages[-1]
    return output["content"]