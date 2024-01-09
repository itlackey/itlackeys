import sys
from crewai import Agent, Task
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models.openai import ChatOpenAI

from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException

os.environ.clear()
load_dotenv()

default_model_name = os.environ.get("ITL_MAIN_MODEL_NAME", "gpt-3.5-turbo")

defalut_llm = ChatOpenAI(openai_api_base=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
                        openai_api_key=os.environ.get("OPENAI_API_KEY"),
                        temperature=0.1,                        
                        model_name="oh-2.5m7b-q51",
                        top_p=0.4)


@tool("syntax_review_tool")
def syntax_review_tool(file_path: str) -> str:
    """
    A tool to review files for markdown syntax errors.

    Parameters:
    - file_path: The path to the markdown file to be reviewed.

    Returns:
    - validation_results: A list of validation results and suggestions on how to fix them.
    """
    
    print("\n\nValidating Markdown syntax...\n\n" + file_path)

    scan_result = None
    try:
        scan_result = PyMarkdownApi().scan_path(file_path)
        results = str(scan_result)
        print(results)
    except PyMarkdownApiException as this_exception:
        print(f"API Exception: {this_exception}", file=sys.stderr)
    
    syntax_validator_agent = Agent(role='Syntax Validator',
                                backstory="You are an expert markdown validator. You are an expert in formatting and structure. You following formatting guidelines strictly.",
                                goal="""
                                   Provide a detailed list of the provided markdown linting results. Give a summary with actionable tasks to 
                                    address the validation results.
                                    """, 
                                allow_delegation=False, 
                                verbose=True,
                                llm=defalut_llm)

    fix_syntax_task = Task(description=""""Give a detailed list of the validation results below. 
                           Be sure to to include suggestions on how to fix the issues.\n\nValidation Results:\n\n""" + results, 
                           agent=syntax_validator_agent)
        
    updated_markdown = fix_syntax_task.execute()

    return updated_markdown  # Return the reviewed document


# General Agent Setup
general_agent = Agent(role='General Assistant',
                        backstory="You are an expert team coach. You help team members with their tasks effectively.",
                        goal="""
                        To help your team members communicate effectively, provide feedback and action items for your team.
                        You are available to assist in formatting team members responses correctly to leverage tools or delegate tasks.
                        You remind team members to use the correct format when requesting help from a team member or accessing a tool.

                            To use a ask a question of a team member or delegate work to them, please use the following format:        
                            ```
                            Thought: Do I need to use a tool? Yes
                            Action: [Delegate work to co-worker, Ask question to co-worker]
                            Action Input: [coworker name]|['question' or 'task']|[information about the task or question]
                            ```

                            For example to ask a the Software Engineer to check the code for best practices:
                            ``` 
                                Thought: Do I need to use a tool? Yes
                                Action: Ask question to co-worker
                                Action Input: Senior Software Engineer|question|Check the code for best practices
                               
                            ```

                            To use a tool, please use the following format:        
                            ```
                            Thought: Do I need to use a tool? Yes
                            Action: [name of tool]
                            Action Input: [the value needed by the tool's arguments]
                            ```

                            For example to use the syntax review tool:
                            ```
                            Thought: Do I need to use a tool? Yes
                            Action: syntax_review_tool
                            Action Input: "print('hello world')"
                            ```
                        Be sure to complete the task once the final answer has been provided.
                        """, 
                        allow_delegation=False, 
                        verbose=True,
                        tools=[syntax_review_tool],
                        llm=defalut_llm)


# Function to Process Documents with the Crew
def process_markdown_document(filename):

    syntax_review_task = Task(description=f"""
             Tell your team how to use the syntax_review_tool to review the file(s) at this path: {filename}
             Collect the final answer from the syntax review tool and then summarize it in bullet points.
             If you already know the answer or if you do not need to use a tool, return it as your Final Answer.""",
             agent=general_agent)
    
    updated_markdown = syntax_review_task.execute()

    return updated_markdown

# Example Usage
processed_document = process_markdown_document("example.md")
print(processed_document)
