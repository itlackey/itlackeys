import glob
import os
import argparse
from dotenv import load_dotenv
from langchain.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process
from langchain.chat_models.openai import ChatOpenAI

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Script to perform an action on files.")
    parser.add_argument("files", nargs='*', help="The glob pattern for file matching or list of files.")
    parser.add_argument("topic", type=str, help="The action to be performed on the files.")
 
    parser.add_argument(
        "--cache-seed",
        default=os.environ.get("ITL_CACHE_SEED", "42"),
        help="Cache seed for the action, or False to disable caching.",
    )

    args = parser.parse_args()

    env_or_file = args.oai

    if os.path.isfile(args.action):
        action = read_action_from_file(args.action)
    else:
        action = args.action

    files = []
    for file_pattern in args.files:
        if os.path.isfile(file_pattern):
            files.append(file_pattern)
        else:
            files.extend(glob.glob(file_pattern))

    if not files:
        print("No files found.")
        return
    else:
        print(f'Files: {files}')
        for file in files:
            print(f'Reviewing File: {file}')
            review_file(file, action, env_or_file=env_or_file, cache_seed=args.cache_seed)
            

def read_action_from_file(file):
    pass

def review_file(file, action, env_or_file, cache_seed):

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
    llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo") # It uses langchain.chat_models, default is GPT4
    )
    writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Tech Content Strategist, known for your insightful
    and engaging articles on technology and innovation. With a deep understanding of
    the tech industry, you transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True,
    llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo") 
    )

    # Create tasks for your agents
    task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts.
    Compile your findings in a detailed report. Your final answer MUST be a full analysis report""",
    agent=researcher
    )

    task2 = Task(
    description="""Using the insights from the researcher's report, develop an engaging blog
    post that highlights the most significant AI advancements.
    Your post should be informative yet accessible, catering to a tech-savvy audience.
    Aim for a narrative that captures the essence of these breakthroughs and their
    implications for the future. Your final answer MUST be the full blog post of at least 3 paragraphs.""",
    agent=writer
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2, # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
    process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )

    # Get your crew to work!
    result = crew.kickoff()
    return result


if __name__ == "__main__":
    main()