import os
import glob
from typing import List, Annotated
import openai
from aider.coders import Coder
from aider import  models
from dotenv import load_dotenv
import argparse
import autogen

def write_to_markdown(messages: Annotated[List[str], "List of messages to write."], file_name: Annotated[str, "Name of the file to write the messages to."]) -> str:

    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(file_name, 'w') as file:
        for message in messages:
            file.write(f'{message}\n\n')
    return f"Messages written to {file_name}."

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
    response = coder.run("Review this conversation and apply the recommended changes to the code.\n\n" + action + "\n\n" + plan)                    

    return response


def review_file(file, action, env_or_file, cache_seed):
    """
    Use Autogen to review file based on the action prompt. Then output the output of the autogen review.
    """

    # read file content from file
    file_content = open(file, 'r').read()

    responses = []

    if cache_seed is None or cache_seed.lower() == "false":
        cache_seed = None


    config_list_local = autogen.config_list_from_json(
        env_or_file=env_or_file
    )

    llm_config = {"config_list": config_list_local, "cache_seed": cache_seed}

    review_proxy = autogen.UserProxyAgent(
        name="review_user_proxy",
        default_auto_reply="TERMINATE",
        system_message="A coordinator that works with the planner to create a set of instructions. "+
            " If the instructions are acceptable or there is nothing to do, please reply `TERMINATE`." + 
            " If you have no response, please reply `TERMINATE`.", 
        code_execution_config=False,
        human_input_mode="TERMINATE"
    )

    coder = autogen.AssistantAgent(
        name="coder",
        system_message="You are a senior software engineer. You will review provided code and provide suggested edits based on the provided action. Provide your response in markdown format and include code snippets in code blocks." ,
        llm_config=llm_config,
    )


    planner = autogen.AssistantAgent(
        name="planner",
        system_message="You are an expert technical writer and project planner. You specialize in writting a summary of code and code reviews. " + 
            " Only respond with a detailed list of changes that should be made to the code." +
            " These include a list of steps that need to be taken to complete the code changes and code samples.",
        llm_config=llm_config,
    )

    is_silent = True

    review_proxy.send(recipient=coder, request_reply=True, silent=is_silent,
                      message= "REQUEST: " + action + "\n\n" + "CODE: \n\n"  + file_content)
    
    message = coder.last_message(review_proxy)
    review_output_text = message["content"]
    responses.append(review_output_text)

    review_proxy.send(recipient=planner, request_reply=True, silent=is_silent,
        message="Read this review and reply with a list of steps that need to be taken to complete the code changes. \n\n" + str.join("\n\n", responses))

    message = planner.last_message()
    review_output_text = message["content"]
    responses.append("PLAN: \n\n" + review_output_text)


    file_name = os.path.join(".cache", cache_seed, "reviews/", file + ".md")

    write_to_markdown(responses, file_name=file_name)

    return review_output_text

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
    parser.add_argument("--oai", type=str, default="local.json", help="The environment variable or JSON file to load configurations from.")
    
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
            plan = review_file(file, action, env_or_file=env_or_file, cache_seed=args.cache_seed)
            if args.update_files and plan is not None:
                print(f'Updating file: {file}')
                response = perform_action(file, action, plan)
                print(f'Response: {response}')

if __name__ == "__main__":
    main()
