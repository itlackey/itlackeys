import glob
import os
import argparse
from dotenv import load_dotenv
from langchain.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process
from langchain.chat_models.openai import ChatOpenAI
from langchain.llms import Ollama
from langchain_community.agent_toolkits import FileManagementToolkit

hermes_llm = Ollama(model="openhermes")
mixtral_llm = Ollama(model="dolphin-mixtral")
mistral_llm = Ollama(model="mistral")
coder_llm = Ollama(model="magicoder:7b-s-cl-q5_K_M")
phi_llm = Ollama(model="phi")
gpt3_llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo") # It uses langchain.chat_models, default is GPT4

tools = FileManagementToolkit(
    root_dir=str(".cache/temp"),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()

def write_article(topic):

    search_tool = DuckDuckGoSearchRun()

    # Define your agents with roles and goals
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Uncover cutting-edge developments in AI and data science in',
        backstory="""You are a Senior Research Analyst at a leading tech think tank.
        Your expertise lies in identifying emerging trends and technologies in AI and
        data science. You have a knack for dissecting complex data and presenting
        actionable insights.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=mistral_llm)

    writer = Agent(
        role='Tech Content Strategist',
        goal="""Craft compelling content on a given technical topic. 
            Inlclude relevant links to sources and code snippets when possible.
            Write the content in markdown format to a file for the editor to review.""",
        backstory="""You are a renowned Tech Content Strategist, known for your insightful
        and engaging articles on technology and innovation. With a deep understanding of
        the tech industry, you transform complex concepts into compelling narratives.""",
        verbose=True,
        allow_delegation=True,
        llm=mistral_llm)
    
    coder = Agent(
        role='Senior Software Engineer',
        goal='Provide high quality code snippets for inclusion in blog articles.',
        backstory="""You are a renowned software engineer and content creator. You have
        a deep understanding of the tech industry and are well-versed in the latest trends.
        You have spent many years writing code and online content.""",
        verbose=True,
        allow_delegation=True,
        llm=coder_llm)
    
    editor = Agent(
        role='Chief Content Editor',
        goal='Review the article written by the Tech Content Strategist, and make any necessary adjustments',
        backstory="""You are a renowned content editor and author. You produce high quality content
        that is informative and engaging. You have a deep understanding of the tech industry and 
        are well-versed in the latest trends and innovations.""",
        verbose=True,
        allow_delegation=True,
        llm=mistral_llm)
    
    # Create tasks for your agents
    research_task = Task(
        description=f"""Conduct a comprehensive analysis of {topic}.
        Search the web for relevant information and data about this topic.
        Compile your findings in a detailed report. Your final answer MUST be a full analysis report""",
        agent=researcher
    )

    # outline_task = Task(
    #     description=f"""Using the insights from the researcher's report about {topic}, develop an 
    #     outline for and engaging blog post that highlights the most significant information about the topic.
    #     Use the write_file tool to save the outline the `.cache/draft_outline.md` file path.
    #     Call the write_file tool with the following arguments:
    #     Action: write_file
    #     Action Input: "{{"file_path": ".cache/draft_outline.md", "text": "<the content to write to the file>"}}"
        
    #     Return the name of the draft outline file in your final answer so the crew can access it.""",
    #     agent=writer
    # )


    outline_task = Task(
        description=f"""Using the insights from the researcher's report about {topic}, develop an 
        outline for and engaging blog post that highlights the most significant information about the topic.
        Continue to research the topic untile you have a clear outline that is specifc to your topic.
        Return the outline as your final answer so the crew can access it.""",
        agent=writer
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[researcher, writer, editor, coder],
        tasks=[research_task, outline_task],
        verbose=2, # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
        process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )

    # Get your crew to work!
    result = crew.kickoff()


    filename = Task(description=f"""Create a valid file name based this topic: {topic}. 
                    It should be less than 50 characters and have no special characters.
                    IT IS VERY IMPORTANT THAT YOU ONLY RETURN A VALID FILE NAME.
                    DO NOT RETURN ANYTHING ELSE OR THE JOB WILL FAIL!
                    """, agent=researcher).execute()

    print(f"Outline complete! Writing to {filename}...")
    
    # if .output dir does not exist, create it
    if not os.path.exists(".output"):
        os.makedirs(".output")

    # write to file in .output folder using topic as filename    
    with open(f".output/{filename}-draft-outline.md", "w") as f:
        f.write(result)

    # write_article_task = Task(
    #     description=f"""Using the insights from the researcher's report, and the outline
    #     in the `.cache/draft_outline.md` file to create a detailed blog post. Use the write_file tool to
    #     save your blog post to the `.cache/draft_article.md` file path.

    #     Your post should be informative yet accessible, catering to a tech-savvy audience.
    #     Aim for a narrative that captures the essence of {topic}. 
    #     Your final answer MUST be the full blog post of at least 3 paragraphs and 
    #     should include code snippets if possible.

    #     YOU MUST write the blog post to a the `.cache/draft_article.md` file path using the write_file tool.
    #     Be sure to pass the content of your article as the `text` argument to the write_file tool.

    #     Calling the write_file tool with the following arguments:
    #     Action: write_file
    #     Action Input: {{"file_path": ".cache/draft_article.md", "text": <the content to write to the file>}}
        
    #     This is VERY IMPORTANT, your job depends on it. If you do not write the file, the task will fail.
    #     Return the name of the blog post file in your final answer so the crew can access it.""",
    #     agent=writer
    # )
    
    write_article_task = Task(
            description=f"""Create a blog post about {topic} using the following outline.

            <Beging Outline>
            {result}
            </End Outline>

            Your post should be informative yet accessible, catering to a tech-savvy audience.
            Aim for a narrative that captures the essence of {topic}. Remove any text that is not relevant. 
            Your final answer MUST be the full blog post of at least 3 paragraphs and 
            should include code snippets if possible.
            
            Return the name of the blog post in your final answer so the crew can access it.
            """,
            agent=writer
        )
        
    review_task = Task(
        description=f"""Review the blog post and make any necessary adjustments. Final answer MUST be a full blog post that
        is informative and engaging. It should be at least 5 paragraphs and be well suited to be published in a tech blog
        and be related to {topic} and following this outline:
        <Begin Outline>
        {result}
        </End Outline>
        """,
        agent=editor
    )

    article_crew = Crew(
        agents=[researcher, writer, editor, coder],
        tasks=[write_article_task, review_task],
        verbose=2, # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
        process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )

    print("Outline saved, writing article...")

    result = article_crew.kickoff()

    print("Article complete! Writing to file...")

    # write to file in .output folder using topic as filename    
    with open(f".output/{filename}-draft-article.md", "w") as f:
        f.write(result)

    print("Article saved!")

    return result


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Script to perform an action on files.")
    parser.add_argument("topic", type=str, help="The action to be performed on the files.")

    args = parser.parse_args()
    write_article(args.topic)

     
if __name__ == "__main__":
    main()