import os
import glob
from typing import List, Annotated
import openai
from aider.coders import Coder
from aider import  models
from dotenv import load_dotenv
import argparse
import autogen


config_list_gpt35 = autogen.config_list_from_json(
    env_or_file="oai.json",
    filter_dict={
        "model": {
            "deepseek",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "chatgpt-35-turbo-0301",
            "gpt-35-turbo-v0301",
        },
    },
)

config_list_local = autogen.config_list_from_json(
    env_or_file="oai.json",
    filter_dict={
        "model": {
            "local"
        },
    },
)


user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="A human admin. If the code is executing successfully, please reply `TERMINATE`.",
    code_execution_config={"last_n_messages": 2, "work_dir": ".groupchat"},
    human_input_mode="NEVER"
)
review_proxy = autogen.UserProxyAgent(
    name="review_user_proxy",
    system_message="A coordinator that works with the planner to create a set of instructions. "+
        " If the instructions are acceptable or there is nothing to do, please reply `TERMINATE`.",
    code_execution_config=False,
    human_input_mode="TERMINATE"
)


# @user_proxy.register_for_execution()
# @reviewer.register_for_llm(name="write_to_markdown", description="Write messages to a markdown file.")
def write_to_markdown(messages: Annotated[List[str], "List of messages to write."], file_name: Annotated[str, "Name of the file to write the messages to."]) -> str:
    with open(file_name, 'w') as file:
        for message in messages:
            file.write(f'{message}\n\n')
    return f"Messages written to {file_name}."

def read_action_from_file(file_path):
    with open(file_path, 'r') as file:
        action = file.read().strip()
    return action


def perform_action(file, action, update_files=False, cache_seed=None):
    """
    Executes the specified action on the given files using the OpenAI API.

    Args:
        files (List[str]): A list of file names to be processed.
        action (str): The action to be performed on the files.
        update_files (bool, optional): Whether to update the files after running the action. Defaults to False.

    Returns:
        str: The response from the coder after running the action.
    """
    review_output = review_file(file, action, cache_seed)
    

    if update_files:        
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], 
                               base_url=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"))
        
        model_name = os.environ.get("AIDER_DEFAULT_MODEL", "gpt-3.5-turbo") 
        
        model =  models.Model.create(model_name, client)
        coder = Coder.create(client=client, main_model=model, fnames=[file])

        print("Apply these changes:\n\n" + review_output)
        response = coder.run("Review this conversation and apply the recommended changes to the code.\n\n" + review_output)               
    else:
        response = review_output        

    return response


def review_file(file, action, cache_seed):
    """
    Use Autogen to review file based on the action prompt. Then output the output of the autogen review.
    """

    
    if cache_seed is None or cache_seed.lower() == "false":
        cache_seed = None

    llm_config = {"config_list": config_list_local, "cache_seed": cache_seed}

    coder = autogen.AssistantAgent(
        name="coder",
        system_message="You are a senior software engineer. Review the provide code and requirements. Provide a list of changes that should be made and the suggested code. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )
    reviewer = autogen.AssistantAgent(
        name="reviewer",    
        system_message="You are a code reviewer. When the coder is done, review their suggestions and the existing code." +
        " Ensure that the changes are of high quality and follow best practices. " +
        " Do not allow the renaming of files in the suggested changes. " +     
        " Reply `TERMINATE` in the end when everything is done." +
        " If the code is executing successfully and the changes are acceptable, please reply `TERMINATE`.",
        llm_config=llm_config,
    )

    groupchat = autogen.GroupChat(agents=[user_proxy, coder, reviewer], messages=[], max_round=12,
                                  speaker_selection_method="round_robin")
    

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # read file content from file
    file_content = open(file, 'r').read()
    
    user_proxy.initiate_chat(manager, message=action + "\n\nFile: " + file_content, clear_history=True, file_name=file + ".md")
    #responses.append(groupchat.messages)
    
    # append the content property of all the groupchat.messages to the responses list
    responses = []
    for message in groupchat.messages:
        print(message)
        if "content" in message and message["content"] and (message["name"] == "reviewer" or message["name"] == "coder"):
            responses.append(message["name"] + ": \n\n" + message["content"])
    
    planner = autogen.AssistantAgent(
        name="planner",
        system_message="You are an expert technical writer and project planner. You specialize in writting a summary of code and code reviews. " + 
            " Only respond with a detailed list of changes that should be made to the code." +
            " These include a list of steps that need to be taken to complete the code changes and code samples." +
            " Include `TERMINATE` at the end of the list. ",
        llm_config=llm_config,
    )
    review_proxy.send(recipient=planner, request_reply=True,
        message="Read this review and reply with a list of steps that need to be taken to complete the code changes. \n\n" + str.join("\n\n", responses))

    message = planner.last_message()
    #print(message)
    review_output_text = message["content"]
    responses.append("planner: \n\n" + review_output_text)
    write_to_markdown(responses, file + ".md")

    return review_output_text

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Script to perform an action on files.")
    parser.add_argument("files", nargs='*', help="The glob pattern for file matching or list of files.")
    parser.add_argument("--action", type=str, help="The action to be performed on the files.")
    parser.add_argument(
        "--update-files",
        action="store_true",
        help="Whether to update the files after running the action.",
    )

    parser.add_argument(
        "--cache-seed",
        nargs='?',
        const=False,
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
            print(f'File: {file}')    
            response = perform_action(file, action, update_files=args.update_files, cache_seed=args.cache_seed)
            print(f'Response: {response}')

if __name__ == "__main__":
    main()
