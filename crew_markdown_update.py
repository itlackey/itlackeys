from crewai import Agent, Task, Crew, Process


import os
import openai
from aider.models import models
from aider.coders import Coder

def write_to_markdown(message, file_name) -> str:

    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(file_name, 'w') as file:
        file.write(f'{message}\n\n')

    return f"Message written to {file_name}."

def aider_update_document(file_name, content):

    print(f"Updating document: {file_name}")
    write_to_markdown(content, os.path.join(file_name, "_updated.md"))
    #return plan

    # Initialize Aider Client
    client = openai.OpenAI(api_key=os.environ.get("AIDER_OPENAI_API_KEY", os.environ["OPENAI_API_KEY"]),
                           base_url=os.environ.get("AIDER_OPENAI_API_BASE_URL", os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")))

    # Set Model Name
    model_name = os.environ.get("AIDER_MODEL", "gpt-3.5-turbo") 

    # Create Model and Coder instances
    model = models.Model.create(model_name, client)
    coder = Coder.create(client=client, main_model=model, fnames=[file_name])

    # Run Coder with the action plan
    try:
        response = coder.run(f"Please update this document to this content:\n\n{content}")
        return response
    except Exception as e:
        print(f"Error occurred while running Aider Coder: {e}")
        return None



# Crews as Tools Definitions
def syntax_review_tool(document):
    # Placeholder for Syntax Review Logic
    # ...
    return document  # Return the reviewed document

def syntax_correction_tool(document):
    # Placeholder for Syntax Correction Logic using Aider
    corrected_document = aider_update_document("Syntax correction plan")
    return corrected_document

def editorial_review_tool(document):
    # Placeholder for Editorial Review Logic
    # ...
    return document  # Return the editorially reviewed document

def content_update_tool(document):
    # Placeholder for Content Update Logic using Aider
    updated_document = aider_update_document("Content update plan")
    return updated_document

# General Agent Setup
general_agent = Agent(role='General Document Processor', goal='Process Markdown documents through various stages')
general_agent.tools.extend([syntax_review_tool, syntax_correction_tool, editorial_review_tool, content_update_tool])

# Define Tasks Using Crew Tools
tasks = [
    Task(description='Use syntax_review_tool to review the document', agent=general_agent),
    Task(description='Use syntax_correction_tool to correct syntax errors', agent=general_agent),
    Task(description='Use editorial_review_tool for editorial review', agent=general_agent),
    Task(description='Use content_update_tool to update document content', agent=general_agent)
]

# Instantiate and Configure a Single Crew
document_processing_crew = Crew(agents=[general_agent], tasks=tasks, process=Process.sequential)

# Function to Process Documents with the Crew
def process_markdown_document(document):
    processed_document = document_processing_crew.kickoff(document)
    return processed_document

# Example Usage
example_document = "Your Markdown document content here"
processed_document = process_markdown_document(example_document)
print(processed_document)
