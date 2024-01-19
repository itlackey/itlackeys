
import sys
import os
from dotenv import find_dotenv, load_dotenv
from langchain.llms import Ollama
from langchain.chat_models.openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool, DuckDuckGoSearchRun

from tools import write_outline_to_markdown, read_outline_from_file, write_article_to_markdown, publish_to_devto_from_file
from langchain.agents import load_tools

os.environ.clear()
load_dotenv(find_dotenv())

openai_llm = ChatOpenAI(openai_api_base=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
                    temperature=0,
                    top_p=0.3,
                    model_name=os.environ.get("MODEL_NAME", "gpt-3.5-turbo")) #"gpt-3.5-turbo")


human_tools = load_tools(['human'])

hermes_llm = Ollama(model="openhermes")
mixtral_llm = Ollama(model="dolphin-mixtral")
mistral_llm = Ollama(model="mistral")

default_llm = mistral_llm


# Create a DuckDuckGo search tool
ddg_search_tool = Tool(
    name="search_tool",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for information"
)

def write_article(tutorial_topic):
    # Define Agents
    research_analyst = Agent(
        role='Research Analyst',
        backstory="You have built a career researching and publishing of technical articles.",
        goal=f"""Create an outline for a technical article about the following topic:
        
        <BEGIN TOPIC>
        {tutorial_topic} 
        <END TOPIC>

        Look up any needed information from the web to complete the outline.
        Be sure to include a references section at the end of the outline
        to cite any sources you used.
        """,
        tools=[ddg_search_tool, write_outline_to_markdown],  # Assign the search tool
        verbose=True,
        allow_delegation=False,
        llm=default_llm
    )

    technical_writer = Agent(
        role='Technical Writer',
        backstory="You have built a career writing technical articles.",
        goal=f"""Write the technical article about the following topic:
        <BEGIN TOPIC>
        {tutorial_topic} 
        <END TOPIC>

        Look up any needed information from the web to complete the article.
        Be sure to include a references section at the end of the outline
        to cite any sources you used. Include a link to documentation or repositories when possible.

        The article should be well written and easy to read. It should contain between five and ten
        paragraphs. It should contain a single top level header that is the title of the article.

        The article should be in valid markdown format.
        """,
        tools=[ddg_search_tool, read_outline_from_file, write_article_to_markdown],  # Assign the search and writing tools
        verbose=True,
        allow_delegation=False,
        llm=default_llm
    )

    publisher = Agent(
        role='Publisher',
        goal='Publish the article to dev.to',
        backstory="You have built a career as an online content manager.",
        verbose=True,
        allow_delegation=False,
        llm=default_llm,
        tools=[publish_to_devto_from_file],  # Assuming the tool has been integrated
    )

    project_manager = Agent(
        role='Project Manager',
        goal="""Interact with the user through the human tools to request approvals.
            You are also responsible for ensuring the content is written to the correct file.
            It is VERY IMPORTANT that you DO NOT modify the content of the outline or the article.
            NEVER modify the content of the article or the outline and NEVER respond other than
            how it is described in the current task. THIS IS VERY IMPORTANT! YOUR JOB DEPENDS ON IT!!
                    
            Use the 'write_outline_to_markdown' tool like this:
            Action: write_outline_to_markdown
            Action Input:<BEGIN OUTLINE>
                        [outline_content]
                        <END OUTLINE>

            Use the 'write_article_to_markdown' tool like this:
            Action: write_article_to_markdown
            Action Input:  <BEGIN ARTICLE>
                            [article_content]
                            <END ARTICLE>
            
            """,
        backstory="You have built a career as a project manager working with customers and project teams.",
        tools=[write_article_to_markdown, write_outline_to_markdown] + human_tools,  # Assign the search tool
        verbose=True,
        llm=default_llm,
        allow_delegation=True
    )

    # Define Tasks
    task_outline_creation = Task(
        description=f"""Create an outline for an article based on the topic provided below. 
        Include a references section if online sources are used.
        
        Here is the topic: 
        
        <BEGIN TOPIC>
        {tutorial_topic}
        <END TOPIC>

        Once you have the outline, make sure to check with the human if the outline is good before returning your Final Answer.

        Return the outline as your final answer in markdown. Like this:

        Final Answer:
        <BEGIN OUTLINE>
        [outline_content]
        <END OUTLINE>


        You MUST replace [outline_content] with the outline you wrote.
        """,
        agent=research_analyst,
        tools=[ddg_search_tool]+human_tools
    )

    task_outline_verification = Task(
        description="""Verify the article outline with the human supervisor using the human tool.
            You can use the human tool like this:

            Action: human
            Action Input:\n
            <BEGIN OUTLINE>
                [outline_content]
            <END OUTLINE>\n\n
            Do you approve the outline?

            You MUST replace [outline_content] with the outline you wrote in the above example.

            The supervisor will review the outline and provide feedback.

            If the outline is approved, return the outline as your final answer in markdown. Like this:

            Final Answer:
            <BEGIN OUTLINE>
            [IMPORTANT:place outline content here]
            <END OUTLINE>

            You MUST replace [outline_content] with the outline you wrote.
            
            If the outline is rejected, ask the team to review the feedback and update the outline.
            Once the outline is updated, as the supervisor, for approval again.
        """,
        agent=project_manager,
        tools=[write_outline_to_markdown] + human_tools  # Using the 'human tool' for approval
    )
    task_outline_saving = Task(
        description="""
            If the outline is approved, use the 'write_outline_to_markdown' tool by using the following message template:
            ```
            Action: write_outline_to_markdown
            Action Input:<BEGIN OUTLINE>
                        [IMPORTANT: place outline content here!]
                        <END OUTLINE>
            
            ```
            It is VERY IMPORTANT that you place the outline content in the message template and remove the placeholder.

            Then return the outline as your final answer.

            If the outline is rejected, do NOT write the outline to a markdown file.
            Instead, return 'The outline was rejected.' as your final answer.

        """,
        agent=project_manager,
        tools=[write_outline_to_markdown]
    )

    # Define the Crew
    crew = Crew(
        agents=[research_analyst, project_manager],
        tasks=[
            task_outline_creation,
            task_outline_verification,
            task_outline_saving
        ],
        process=Process.sequential,
        verbose=True
    )


    # Code to initiate the crew's work
    outline_result = crew.kickoff()
    print(outline_result)

    task_article_writing = Task(
        description=f"""Write a detailed article based on the topic provided below.    
        Here is the topic: 
        
        <BEGIN TOPIC>
        {tutorial_topic}
        <END TOPIC>

        You can read the outline from the 'outline.md' file using the 'read_outline_from_file' tool.

        Use the 'read_outline_from_file' tool like this:
        Action: read_outline_from_file
        Action Input: [filename]

        Replace [filename] with the filename of the outline.
        
        Write a detailed article based on the topic provided above and the outline from the 'outline.md' file.
        Each section of the outline should have at least one paragraph in the article.
        Be sure to include code samples using markdown code blocks in the article.
        Be sure to include links to documentation or repositories when possible.

        Once you have the article content, yout MUST write it to a markdown file named 'article.md'.

        Return the article as your final answer in markdown. Like this:
        Final Answer:
        <BEGIN ARTICLE>
        [article_content]
        <END ARTICLE>
        
        You MUST replace [article_content] with the article you wrote.
    
        """,
        agent=technical_writer,
        tools=[read_outline_from_file]
    )

    task_article_verification = Task(
        description="""                
            Verify the article content with the human supervisor using the human tool.
            If the supervisor has already accepted the article, then return the article 
            as your final answer as described below.

            You can use the human tool like this, you MUST Replace [article_content] with the content of the article:
            
            Action: human
            Action Input:\n
            <BEGIN ARTICLE>
            [article_content]
            <END ARTICLE>\n\n
            Do you approve this article?
            
            The supervisor will review the article and provide feedback.

            If the article is rejected, continue to refine the article based on the feedback.
    
            If the article is approved, then return the content of the article as your final answer
            in the following format:

            Final Answer:
            <BEGIN ARTICLE>
            [IMPORTANT: Place the full article content here]
            <END ARTICLE>            
        """,
        agent=project_manager,
        tools=[write_article_to_markdown]+human_tools  # Using the 'human tool' for approval
    )
    task_article_saving = Task(
        description="""            
            If the article is approved, use the 'write_article_to_markdown' tool like this:
            Action: write_article_to_markdown
            Action Input:<BEGIN ARTICLE>
                        [article]
                        <END ARTICLE>
            
            Then return the article as your final answer.

            If the article is rejected, do NOT write the article to a markdown file.
            Instead, return 'The article was rejected.' as your final answer.
        """,
        agent=project_manager,
        tools=[write_article_to_markdown]
    )
    task_publishing = Task(
        description='Publish the approved article to dev.to.',
        agent=publisher,
        tools=[publish_to_devto_from_file]  # Assuming the tool has been integrated
    )

    article_crew = Crew(
        agents=[ technical_writer, project_manager],
        tasks=[
            #task_outline_creation,
            #task_outline_verification,
            task_article_writing,
            task_article_verification,
            task_article_saving
        ],
        process=Process.sequential,
        verbose=True
    )
    # Code to initiate the article writing crew's work
    article_result = article_crew.kickoff()

    print(outline_result)
    print(article_result)

if __name__ == "__main__":
    if len(sys.argv) > 1:

        print(f"Topic: {sys.argv[1]}")
        tutorial_topic = sys.argv[1]
        write_article(tutorial_topic)


