import os
import sys
import glob
import openai
from aider.coders import Coder
from aider import  models

def read_action_from_file(file_path):
    with open(file_path, 'r') as file:
        action = file.read().strip()
    return action


def perform_action(files, action):
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_name = os.environ.get("AIDER_DEFAULT_MODEL", "gpt-3.5-turbo") #"gpt-4-1106-preview"
    model =  models.Model.create(model_name, client)
    coder = Coder.create(client=client, main_model=model, fnames=files)
    response = coder.run(action)
    return response

def main(glob_pattern, action_file_path):
    action = read_action_from_file(action_file_path)
    files = glob.glob(glob_pattern)
    response = perform_action(files, action)
    print(f'Response: {response}')

if __name__ == "__main__":
    glob_pattern = sys.argv[1]
    action_file_path = sys.argv[2]
    main(glob_pattern, action_file_path)
