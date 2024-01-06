import glob
import os
import argparse
from typing import Annotated
import openai
from aider.coders import Coder
from aider import  models
from dotenv import load_dotenv
from langchain.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process
from langchain.chat_models.openai import ChatOpenAI

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Script to perform an action on files.")
    parser.add_argument("files", nargs='*', help="The glob pattern for file matching or list of files.")
    parser.add_argument("action", type=str, help="The action to be performed on the files.")
    parser.add_argument(
        "--update-files",
        action="store_true",
        help="Whether to update the files after running the action.",
    )
    #parser.add_argument("--oai", type=str, default="local.json", help="The environment variable or JSON file to load configurations from.")
    
    parser.add_argument(
        "--cache-seed",
        default=os.environ.get("ITL_CACHE_SEED", "42"),
        help="Cache seed for the action, or False to disable caching.",
    )

    args = parser.parse_args()

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
            plan = review_file(file, action, cache_seed=args.cache_seed)
            if args.update_files and plan is not None:
                print(f'Updating file: {file}')
                response = perform_action(file, action, plan)
                print(f'Response: {response}')

def write_to_markdown(message, file_name: Annotated[str, "Name of the file to write the messages to."]) -> str:

    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(file_name, 'w') as file:
        file.write(f'{message}\n\n')

    return f"Message written to {file_name}."

def read_action_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            action = file.read().strip()
        return action
    else:
        return None

def perform_action(file, action, plan): 
    """
    Executes the specified action on the given files using the OpenAI API.

    Args:
        files (List[str]): A list of file names to be processed.
        action (str): The action to be performed on the files.
        update_files (bool, optional): Whether to update the files after running the action. Defaults to False.

    Returns:
        str: The response from the coder after running the action.
    """
      
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], 
                            base_url=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"))
    
    model_name = os.environ.get("AIDER_MODEL", "gpt-3.5-turbo") 
    
    model =  models.Model.create(model_name, client)
    coder = Coder.create(client=client, main_model=model, fnames=[file])

    # print("Apply these changes:\n\n" + plan)
    #response = coder.run("Review this conversation and apply the recommended changes to the code.\n\n" + action + "\n\n" + plan)                    
    response = coder.run(plan)                    

    return response

def review_file(file, action, cache_seed):

    search_tool = DuckDuckGoSearchRun()

    # Define your agents with roles and goals
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Use the web to find the latest documentation for language and frameworks',
        backstory="""You are a Senior Research Analyst at a leading software development company.
        Your expertise lies in assisting developers with language and framework research. 
        You have a knack for dissecting complex code and presenting actionable insights.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo") # It uses langchain.chat_models, default is GPT4
    )
    coder = Agent(
        role='Senior Software Engineer',
        goal='Provide high quality code analysis with specific feedback, action items, and sample code when needed.',
        backstory="""You are a renowned Code Review Specialist, known for your insightful
        and and actionable code reviews. With a deep understanding of
        software engineering in many languages, you are able to 
        review complex code across multiple platforms, frameworks, and languages.""",
        verbose=True,
        allow_delegation=True,
        llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo") 
    )
    reviewer = Agent(
        role='Code Review Specialist',
        goal='Provide high quality code reviews with specific feedback and action items.',
        backstory="""You are a renowned Code Review Specialist, known for your insightful
        and and actionable code reviews. With a deep understanding of
        software engineering in many languages, you are able to 
        review complex code across multiple platforms, frameworks, and languages.""",
        verbose=True,
        allow_delegation=True,
        llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo") 
    )

    code = open(file, 'r').read()

    # Create tasks for your agents
    overview = Task(
        description="""Conduct a comprehensive analysis of the provided code and the documentation found
         on the web. Provide an over of the code and what frameworks and libraries were used. Here is the code: """ + code, 
        agent=researcher
    )

    research = Task(
        description="""Conduct a comprehensive analysis of the provided code. Determine the
        language used, the platform used, and any relevant libraries or frameworks used. 
        Search the web for the latest documentation on the language and framework that may be useful
        for the reviewer. Here is the code: """ + code,
        agent=researcher
    )

    review = Task(
        description="""Using the insights from the researcher's report, develop a detailed
        code review that highlights the key areas for improvement and action items.
        Your review should be informative yet concise. Include a list of changes needed to
        improve the code. If any of the changes are urgent, include them in the action items and 
        indicate that they are urgent. Your final answer MUST be the full code review summary and 
        a list of between 1 and 10 action items.
        Here is the code: """ + code,
        agent=reviewer
    )

    analyze = Task(
        description="""Using the insights from the other team members analysis,
        create e a list of changes needed to achieve the goal of """ + action + """ 
        Your final answer MUST be a list of between 1 and 10 action items that 
        accomplish the goal of """ + action + """ and any urgent changes from the code review.
        The actions cannot include creating new files. 
        Here is the code: \n\n""" + code,
        agent=coder
    )
    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[researcher, reviewer, coder],
        tasks=[analyze], #[overview, research, review, analyze],
        verbose=2, # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
        process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )

    # Get your crew to work!
    result = crew.kickoff()

    file_name = os.path.join(".cache", cache_seed, "reviews/", file + ".md")
    
    write_to_markdown(result, file_name=file_name)

    return result


if __name__ == "__main__":
    main()