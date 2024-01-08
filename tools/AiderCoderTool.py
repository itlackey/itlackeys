import os
from typing import Optional, Type
from langchain.tools import BaseTool, BaseModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import openai
from aider.coders import Coder
from aider import  models
class AiderCoderToolInput():
    file_name: str
    instructions: str

class AiderCoderTool(BaseTool):
    name = "aider_coder",
    description = "Aider coding assistant. This tool will make changes to a specified file based on a description, a list of action items, or other instructions.",
    args_schema = Type[BaseModel] = AiderCoderToolInput
    
    def _run(
        self, file_name: str, instructions: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use this tool to update the file based on the provided instructions."""
      

        client = openai.OpenAI(api_key=os.environ.get("AIDER_OPENAI_API_KEY", os.environ["OPENAI_API_KEY"]), 
                        base_url=
                        os.environ.get("AIDER_OPENAI_API_BASE_URL", os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")))

        model_name = os.environ.get("AIDER_MODEL", "gpt-3.5-turbo") 
        
        model =  models.Model.create(model_name, client)
        coder = Coder.create(client=client, main_model=model, fnames=[file_name])

        response = coder.run(instructions)  
        
        return response

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("aider_coder does not support async")