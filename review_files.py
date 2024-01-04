import os
import sys
import glob
import openai
from aider.coders import Coder
from aider import  models
from dotenv import load_dotenv
import argparse

def read_action_from_file(file_path):
    with open(file_path, 'r') as file:
        action = file.read().strip()
    return action


def perform_action(files, action, update_files=False):
    """
    Executes the specified action on the given files using the OpenAI API.

    Args:
        files (List[str]): A list of file names to be processed.
        action (str): The action to be performed on the files.
        update_files (bool, optional): Whether to update the files after running the action. Defaults to False.

    Returns:
        str: The response from the coder after running the action.
    """

    if update_files:        
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_name = os.environ.get("AIDER_DEFAULT_MODEL", "gpt-3.5-turbo") #"gpt-4-1106-preview"
        model =  models.Model.create(model_name, client)
        coder = Coder.create(client=client, main_model=model, fnames=files)
        response = coder.run(action)
    else:
        response = action
        
    return response

def main():
    parser = argparse.ArgumentParser(description="Script to perform an action on files.")
    parser.add_argument("glob_pattern", type=str, help="The glob pattern for file matching.")
    parser.add_argument("action_file_path", type=str, help="The path to the action file.")
    parser.add_argument(
        "--update-files",
        action="store_true",
        help="Whether to update the files after running the action.",
    )

    args = parser.parse_args()

    load_dotenv()
    action = read_action_from_file(args.action_file_path)
    files = glob.glob(args.glob_pattern)
    response = perform_action(files, action, update_files=args.update_files)
    print(f'Response: {response}')

if __name__ == "__main__":
    main()
