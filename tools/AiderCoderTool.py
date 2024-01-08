from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import os

# Import the necessary modules for the Coder object
import openai
from aider.coders import Coder
from aider import  models

# Define a Pydantic model for the input arguments
class FileUpdateInput(BaseModel):
    file_path: str = Field(description="Path to the file to be updated")
    instructions: str = Field(description="Instructions to update the file")

# Custom LangChain tool that updates files using the Coder object
class AiderFileUpdaterTool(BaseTool):
    name = "aider_file_updater"
    description = "Updates a file with specific instructions using the Aider Coder"
    args_schema = FileUpdateInput
    return_direct: bool = True  # If you want the result to be returned directly

    def _run(
        self, file_path: str, instructions: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Initialize the OpenAI client with your API key
        
        client = openai.OpenAI(api_key=os.environ.get("AIDER_OPENAI_API_KEY", os.environ["OPENAI_API_KEY"]), 
                        base_url=
                        os.environ.get("AIDER_OPENAI_API_BASE_URL", os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")))

        model_name = os.environ.get("AIDER_MODEL", "gpt-3.5-turbo") 

        model =  models.Model.create(model_name, client)
        # Create a Coder object with the file to be updated
        coder = Coder.create(client=client, main_model=model, fnames=[file_path])

        # Execute the instructions on the file
        result = coder.run(instructions)
        
        # Return the result
        return result

    # If async execution is not supported, raise NotImplementedError
    async def _arun(
        self, file_path: str, instructions: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        raise NotImplementedError("aider_file_updater does not support async execution")

# # Example usage
# file_updater_tool = AiderFileUpdaterTool()
# result = file_updater_tool.run(file_path='path/to/your/file.txt', instructions='specific instructions to update the file')
# print(result)