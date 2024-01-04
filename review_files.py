import os
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
        response = review_files(files, action)

    return response

def review_files(files, action):
    """
    Use Autogen to review file based on the action prompt. Then output the output of the autogen review.
    """
    pass

def main():
    parser = argparse.ArgumentParser(description="Script to perform an action on files.")
    parser.add_argument("files", nargs='*', help="The glob pattern for file matching or list of files.")
    parser.add_argument("--action", type=str, help="The action to be performed on the files.")
    parser.add_argument(
        "--update-files",
        action="store_true",
        help="Whether to update the files after running the action.",
    )

    args = parser.parse_args()

    load_dotenv()

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
            response = perform_action([file], action, update_files=args.update_files)
            print(f'Response: {response}')

if __name__ == "__main__":
    main()
