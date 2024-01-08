from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv
from langchain.tools import tool, DuckDuckGoSearchRun
import markdown
import regex

from crew.AiderCoderAgent import aider_coder_agent

from langchain.chat_models.openai import ChatOpenAI

os.environ.clear()

load_dotenv()
defalut_llm = ChatOpenAI(openai_api_base=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
                        openai_api_key=os.environ.get("OPENAI_API_KEY"),
                        temperature=0,
                        top_p=0.3,
                        model_name=os.environ.get("ITL_MAIN_MODEL_NAME", "gpt-3.5-turbo"))

def write_to_markdown(message, file_name) -> str:

    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(file_name, 'w') as file:
        file.write(f'{message}\n\n')

    return f"Message written to {file_name}."


# @tool("validate_markdown")
# def validate_markdown(content: str) -> str:
#     """
#     Validates the syntax of a given Markdown content.

#     Args:
#         content (str): The entire markdown content to be validated.

#     Returns:
#         str: A message indicating whether the Markdown content is valid or not.
#     """

#     print("Validating Markdown syntax..." + content)

#     # Render the Markdown to HTML
#     rendered_html = markdown.markdown(content)

#     template_regex = regex.compile(r'\s*<p>\s*</p>\s*') #r'::: ability\n\s*<h3>\s*</h3>\s*<p>\s*</p>\s*\n:::'
#     # Validate the rendered HTML against the predefined template using regex
#     if template_regex.fullmatch(rendered_html) is not None:
#         return "Markdown content is valid."
#     else:
#         return "Markdown content is not valid."


# Crews as Tools Definitions
@tool("syntax_review_tool")
def syntax_review_tool(original_markdown: str) -> str:
    """
    A tool to review strings for markdown syntax errors.

    Parameters:
    - original_markdown: The markdown to be reviewed. This should be a string of markdown passed into the tool
    and should contain the entire string. DO NOT truncate or remove any of the context.

    For example a message in this format:
    Thought: Do I need to use a tool? Yes
    Action: syntax_review_tool
    Action Input: [insert the value of original_markdown value]
    Observation:

    Returns:
    - updated_markdown: The corrected markdown based on the syntax review. The correct markdown should not modify any of the context, only the syntax.
    """
    
    print("\n\nValidating Markdown syntax...\n\n" + original_markdown)

    rendered_html = markdown.markdown(original_markdown)

    template_regex = regex.compile(r'\s*<p>\s*</p>\s*') #r'::: ability\n\s*<h3>\s*</h3>\s*<p>\s*</p>\s*\n:::'
    # Validate the rendered HTML against the predefined template using regex
    
    is_valid = template_regex.fullmatch(rendered_html) is not None
    
    
    # validate_task = Task(description="""Review the markdown for any syntax errors.
    #                      Use the markdown_validator tool to ensure the syntax is correct. Be sure to include the 
    #                      markdown code block as the Action Input when calling the markdown_validator tool.
    #                      If you do not include the markdown code block below when calling the markdown_validator tool, 
    #                      the task will fail. It is VERY important that you pass the entire Markdown Code Block content to the validator tool!
    #                      If the markdown is not valid, then update the markdown and revalidate.
    #                      Correct any syntax issues you find in the markdown. 
    #                      It is VERY important that you do not edit the content, only the formatting and structure to ensure valid markdown syntax.
    #                      You may also use the web to reference any documenation you may need.
    #                      Once the markdown is valid, provide the updated markdown code block as your Final Answer in your response.
    #                      Here is the markdown to review and correct.
    #                      Markdown Code Block:\n\n""" + markdown, 
    #                      agent=ability_validator_agent)
    
    # validation_crew = Crew(agents=[ability_validator_agent], 
    #                        tasks=[validate_task], process=Process.sequential)
    
    # updated_markdown = validation_crew.kickoff()

    if(is_valid):
        return original_markdown
    
    fix_syntax_task = Task(description="""Review the markdown for any syntax errors.                         
                        Correct any syntax issues you find in the markdown. 
                        It is VERY important that you do not edit the content, 
                           ONLY update the formatting and structure to ensure valid markdown syntax.
                        Your job depends on preserving content, while correcting syntax errors.
                        Return the updated markdown code block as your Final Answer in your response.
                        It is VERY important to say this is your Final Answer when the markdown is valid.
                        Here is the markdown to review and correct.
                        Markdown Code Block:\n\n""" + original_markdown.lstrip().rstrip(), 
                        agent=ability_validator_agent)
    
    fix_syntax_crew = Crew(agents=[ability_validator_agent], 
                           tasks=[fix_syntax_task], 
                           process=Process.sequential)
    
    updated_markdown = fix_syntax_crew.kickoff()
    return updated_markdown  # Return the reviewed document

@tool("editorial_review_tool")
def editorial_review_tool(original_markdown: str) -> str:
    """
    Updates the content to have correct grammar and spelling.
    
    Parameters:
    - original_markdown (str): The content, a string in markdown format.
        For example a message in this format:
        Action: editorial_review_tool
        Action Input: [insert the original markdown value here]
    Returns:
    - str: The updated content in markdown format.
    """
    
    review_task = Task(description="""Review the following content for editorial errors. 
                       Respond with a list of recommended changes once you are satisfied with the content.
                       Be sure to say this is your Final Answer when you are ready to provide the updated content.
                       \n\n""" + original_markdown, 
                       agent=editorial_review_agent)
    review_crew = Crew(agents=[editorial_review_agent, research_agent], tasks=[review_task], process=Process.sequential)
    updated_markdown = review_crew.kickoff()
    return updated_markdown  # Return the editorially reviewed document

search_tool = DuckDuckGoSearchRun()
research_agent = Agent(role='Researcher', 
        goal='Find relevant information about the document. Such as documentation on proper markdown syntax, or proper grammar.', 
        backstory="""You are a world renowed researcher that uses the web to find the latest documentation for language and frameworks.""",
        allow_delegation=False, 
        verbose=True,
        llm=defalut_llm,
        tools=[search_tool])

# ability_validator_tool = MarkdownValidatorTool() #r'::: ability\n\s*<h3>\s*</h3>\s*<p>\s*</p>\s*\n:::')
# ability_validator_tool.add_template_regex( r'::: ability\n\s*<h3>\s*</h3>\s*<p>\s*</p>\s*\n:::')

ability_validator_agent = Agent(role='Ability Validator',
                                backstory="You are an expert markdown validator. You are an expert in formatting and structure. You following formatting guidelines strictly.",
                                goal="""
                                    Update the formatting and structure of the provided markdown as needed to ensure the document is valid.
                                    Once you are satisfied with the corrected markdown and it is valid, provide the corrected markdown 
                                    as your Final Answer in your response.                                    
                                    """, 
                                allow_delegation=False, 
                                verbose=True,
                                llm=defalut_llm,
                                tools=[syntax_review_tool])


editorial_review_agent = Agent(role='Editorial Reviewer',
                               backstory="You are an expert markdown reviewer.",
                               goal="""Review the Markdown document for editorial errors, such as grammar or spelling mistakes.
                                       If there are any, update the existing markdown with the corrected markdown and 
                                       return the corrected markdown. 
                                       It is VERY important that you pass in the entire content to any tools that you may use.
                                       Once the content has been updated, provide the corrected markdown as your Final Answer in your response.
                                       """,
                               allow_delegation=False,
                               llm=defalut_llm,
                               verbose=True)


aider_agent = aider_coder_agent(defalut_llm)
aider_agent.llm = defalut_llm

# General Agent Setup
general_agent = Agent(role='General Document Processor',
                      backstory="You are an expert project manager.",
                    goal="""Process Markdown documents through various stages. 
                        It is VERY important that you pass in the entire content to any tools that you may use.
                        Once you have a final answer from the crew, provide it in your response and move on to the next task.
                        """, 
                    allow_delegation=True, 
                    verbose=True, llm=defalut_llm)

#general_agent.tools.extend([syntax_review_tool, editorial_review_tool])


# Function to Process Documents with the Crew
def process_markdown_document(filename, markdown):
    # Define Tasks Using Crew Tools
    syntax_tasks = [
        Task(description='Use syntax_review_tool to review this content and reutrn the corrected markdown: \n\n Markdown: \n\n' + markdown, 
             agent=ability_validator_agent),       
    ]
    # Instantiate and Configure a Single Crew
    syntax_crew = Crew(agents=[ability_validator_agent], tasks=syntax_tasks, process=Process.sequential)

    updated_markdown = syntax_crew.kickoff()


    tasks = [
        #Task(description='Use syntax_review_tool to review the original markdown and reutrn the corrected markdown: \n\n' + markdown, agent=general_agent),
        #Task(description='Write updated markdown to the file', agent=aider_agent),
        Task(description='Use editorial_review_tool to review the following markdown and reutrn the updated markdown: \n\n' + updated_markdown, agent=editorial_review_agent),
        Task(description='Write the updated markdown to this file: ' + filename, agent=editorial_review_agent)
    ]

    # Instantiate and Configure a Single Crew
    document_processing_crew = Crew(agents=[editorial_review_agent, aider_agent], tasks=tasks, process=Process.sequential)

    processed_document = document_processing_crew.kickoff()

    return processed_document

# Example Usage
example_document = """

    ::: ability

    ### This is a level 3 header ###
> //> some crazy characters here <--

    an actual paragraph of text

    :::

    """

# set example document to the content of the ./example_document.md
example_document = open('README.md', 'r').read()
processed_document = process_markdown_document("example.md", example_document)
print(processed_document)
