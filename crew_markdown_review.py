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
                        top_p=0.3)


@tool("markdown_validation_tool")
def markdown_validation_tool(file_path: str) -> str:
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
        syntax_validator_agent = Agent(role='Syntax Validator',
                                backstory="You are an expert markdown validator. You are an expert in formatting and structure. You following formatting guidelines strictly.",
                                goal="""
                                   Provide a detailed list of the provided markdown linting results. Give a summary with actionable tasks to 
                                    address the validation results. Write your response as if you were handing it to a developer to fix the issues.
                                    """, 
                                allow_delegation=False, 
                                verbose=True,
                                llm=defalut_llm)

        fix_syntax_task = Task(description="""Give a detailed list of the validation results below. 
                            Be sure to to include suggestions on how to fix the issues.\n\nValidation Results:\n\n""" + results, 
                            agent=syntax_validator_agent)
            
        updated_markdown = fix_syntax_task.execute()

        return updated_markdown  # Return the reviewed document
    except PyMarkdownApiException as this_exception:
        print(f"API Exception: {this_exception}", file=sys.stderr)
        return f"API Exception: {str(this_exception)}"
    


def process_markdown_document(filename):
    """
    Processes a markdown document by reviewing its syntax validation results and providing feedback on necessary changes.

    Args:
        filename (str): The path to the markdown file to be processed.

    Returns:
        str: The list of recommended changes to make to the document.

    """

    # Define general agent
    general_agent  = Agent(role='Requirements Manager',
                    goal="To use the available tools to provide execellent feedback to the team members.",
                    backstory="""You are an expert business analyst and software QA specialist. 
                        You provide high quality, thorough, insightful and actionable feedback.""",
                    allow_delegation=False, 
                    verbose=True,
                    tools=[markdown_validation_tool],
                    llm=defalut_llm)


    # Define Tasks Using Crew Tools
    syntax_review_task = Task(description=f"""
             Use the markdown_validation_tool to review the file(s) at this path: {filename}
             Be sure to pass only the file path to the markdown_validation_tool.
             Use the following format to call the markdown_validation_tool:
             Do I need to use a tool? Yes
             Action: markdown_validation_tool
             Action Input: {filename}

             Collect the final answer from the syntax review tool and then summarize it into a list of changes
             the developer should make to the document.
             
             If you already know the answer or if you do not need to use a tool, return it as your Final Answer.""",
             agent=general_agent)
    
    updated_markdown = syntax_review_task.execute()

    return updated_markdown

# If called directly from the command line take the first argument as the filename
if __name__ == "__main__":

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        processed_document = process_markdown_document(filename)
        print(processed_document)