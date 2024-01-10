import sys
from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.llms import Ollama
from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException

load_dotenv()

default_model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

default_llm = Ollama(model="openhermes")

@tool("markdown_validation_tool")
def markdown_validation_tool(file_path: str) -> str:
    """
    A tool to review files for markdown syntax errors.

    Parameters:
    - file_path: The path to the markdown file to be reviewed.

    Returns:
    - validation_results: A list of validation results 
    and suggestions on how to fix them.
    """
    
    print("\n\nValidating Markdown syntax...\n\n" + file_path)

    scan_result = None
    try:
        scan_result = PyMarkdownApi().scan_path(file_path)
        results = str(scan_result)    
        return results  # Return the reviewed document
    except PyMarkdownApiException as this_exception:
        print(f"API Exception: {this_exception}", file=sys.stderr)
        return f"API Exception: {str(this_exception)}"
    
@tool("file_editor_tool")
def file_editor_tool(file_path_and_instructions: str) -> str:
    """
    A tool to edit files based on the provided instructions.

    Parameters:
    - file_path_and_instructions:  The changes to make to the file and the path to the file to be edited.

    This string should be in the format "<file_path>|<instructions>".
    
    Returns:
    - result: The status of the edit.
    """
    import openai
    from aider.coders import Coder
    from aider import  models

    file_path, instructions = file_path_and_instructions.split("|")

    print("\n\nEditing file...\n\n" + file_path)

    result = None
    try:
        client = openai.OpenAI(api_key=os.environ.get("AIDER_OPENAI_API_KEY", os.environ["OPENAI_API_KEY"]), 
                        base_url=
                        os.environ.get("AIDER_OPENAI_API_BASE_URL", os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")))

        model_name = os.environ.get("AIDER_MODEL", "gpt-3.5-turbo") 

        model =  models.Model.create(model_name, client)
        # Create a Coder object with the file to be updated
        coder = Coder.create(client=client, main_model=model, fnames=[file_path])

        # Execute the instructions on the file
        result = coder.run(instructions)
        #print(result)
        
        return result
    except Exception as this_exception:
        print(f"File Edit Exception: {this_exception}", file=sys.stderr)
        return f"Final Answer: There was an error when editing the file:\n\n {str(this_exception)}"
   

def process_markdown_document(filename):
    """
    Processes a markdown document by reviewing its syntax validation 
    results and providing feedback on necessary changes.

    Args:
        filename (str): The path to the markdown file to be processed.

    Returns:
        str: The list of recommended changes to make to the document.

    """

    # Define general agent
    general_agent  = Agent(role='Requirements Manager',
                    goal="""Provide a detailed list of the provided markdown 
                            linting results. Give a summary with actionable 
                            tasks to address the validation results. Write your 
                            response as if you were handing it to a developer 
                            to fix the issues.
                            DO NOT provide examples of how to fix the issues.""",
                    backstory="""You are an expert business analyst 
					and software QA specialist. You provide high quality, 
                    thorough, insightful and actionable feedback.""",
                    allow_delegation=False, 
                    verbose=True,
                    tools=[markdown_validation_tool],
                    llm=default_llm)
    
    file_editor_agent = Agent(role='File Editor',
                    goal="""To take a like of changes and apply them to a file.""",
                    backstory="""You are an expert developer and content writer. 
                    You edit documents based on the provided instructions.""",
                    allow_delegation=False, 
                    verbose=True,
                    tools=[file_editor_tool],
                    llm=default_llm)


    # Define Tasks Using Crew Tools
    syntax_review_task = Task(description=f"""
			Use the markdown_validation_tool to review 
			the file(s) at this path: {filename}
            
			Be sure to pass only the file path to the markdown_validation_tool.
			Use the following format to call the markdown_validation_tool:
			Do I need to use a tool? Yes
			Action: markdown_validation_tool
			Action Input: {filename}

			Get the validation results from the tool 
			and then summarize it into a list of changes
			the developer should make to the document.
			
			If you already know the answer or if you do not need 
			to use a tool, return it as your Final Answer.""",
             agent=general_agent)

    
    edit_file_task = Task(description=f"""
			Use the instructions provided to you to edit the specified 
			the file(s) at this path: {filename}
            
            Once the file is edited, return the result as your Final Answer.
            
			Be sure to pass only the file path and the complete set of instructions
            you receiveto the file_editor_tool.

			Use the following format to call the file_editor_tool:
			Do I need to use a tool? Yes
			Action: file_editor_tool
			Action Input: {filename}|<the full set of instructions>
			""",
            agent=file_editor_agent)  
    
    file_edit_crew = Crew(tasks=[syntax_review_task, edit_file_task], 
                          agents=[general_agent,file_editor_agent], 
                          process=Process.sequential)
    
    result = file_edit_crew.kickoff()

    return result


# If called directly from the command line take the first argument as the filename
if __name__ == "__main__":

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        processed_document = process_markdown_document(filename)
        print(processed_document)


