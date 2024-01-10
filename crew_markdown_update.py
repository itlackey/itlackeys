import sys
from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv

from langchain.llms import Ollama
from tools.AiderCoderTools import file_editor_tool
from tools.MarkdownTools import markdown_validation_tool

load_dotenv()


default_llm = Ollama(model= "openhermes")

  
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
                    thorough, insightful and actionable feedback via 
                    detailed list of changes and actionable tasks.""",
                    allow_delegation=False, 
                    verbose=True,
                    tools=[markdown_validation_tool],
                    llm=default_llm)
    
    file_editor_agent = Agent(role='File Editor',
                    goal="""To take a list of changes and apply them to a file.
                    If there is an error while editing a file, you should inform 
                    the team that you cannot currently edit the file.""",
                    backstory="""You are an expert developer and content writer. 
                    You edit documents based on the provided instructions.""",
                    allow_delegation=False, 
                    verbose=True,
                    tools=[file_editor_tool, markdown_validation_tool],
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
            DO NOT include examples of how to fix the issues.
            DO NOT change any of the content of the document or
            add content to it. It is critical to your task to
            only respond with a list of changes.
			
			If you already know the answer or if you do not need 
			to use a tool, return it as your Final Answer.""",
             agent=general_agent)

    
    edit_file_task = Task(description=f"""
			Use the instructions provided to you to edit the specified 
			the file(s) at this path: {filename}
            
            Once the file is edited, return the result as your Final Answer.
            If there is an error while editing a file, you should inform 
            the team that you cannot currently edit the file.
            If there are no changes needed to the file,
            your taks is complete, and you should inform the team.

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


