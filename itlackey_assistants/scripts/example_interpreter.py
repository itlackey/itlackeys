from interpreter import interpreter

from ..tools.OpenInterpreterTool import open_interpreter_tool

# # @tool("Code Execution Tool")
# def open_interpreter_tool(prompt: str) -> str:
#     """This tool will take a prompt and execute code to satisfy the request."""
#     interpreter.offline = True # Disables online features like Open Procedures
#     interpreter.llm.model = "openai/llama3" # "openhermes" # Tells OI to send messages in OpenAI's format
#     interpreter.llm.api_key = "fake_key" # LiteLLM, which we use to talk to LM Studio, requires this
#     interpreter.llm.api_base = "http://localhost:11434/v1" # Point this at any OpenAI compatible server
#     interpreter.auto_run = True

#     for result in interpreter.chat(prompt, display=False):
#         print(result)
#     return "Tool output"

open_interpreter_tool("list the files in the current directory")
