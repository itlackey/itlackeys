from typing import AnyStr, Type
import markdown
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from regex import Pattern
import regex

# Define a Pydantic model for the input arguments
class MarkdownValidationInput(BaseModel):
    markdown_content: str = Field(description="Markdown content to be validated")

# Custom LangChain tool that validates Markdown syntax and compares rendered HTML to a predefined pattern
class MarkdownValidatorTool(BaseTool):
    name = "markdown_validator"
    description = "Validates Markdown syntax and compares rendered HTML to a predefined pattern"
    args_schema: Type[BaseModel] = MarkdownValidationInput 
    template_regex: Pattern[AnyStr] = regex.compile(r'\s*<p>\s*</p>\s*')

    def add_template_regex(self, pattern: str):
        self.template_regex = regex.compile(pattern)

    def _run(self, markdown_content: str) -> str:
        
        print("Validating Markdown syntax..." + markdown_content)

        # Render the Markdown to HTML
        rendered_html = markdown.markdown(markdown_content)

        # Validate the rendered HTML against the predefined template using regex
        if self.template_regex.fullmatch(rendered_html) is not None:
            return "Markdown content is valid."
        else:
            return "Markdown content is not valid."

    # If async execution is not supported, raise NotImplementedError
    async def _arun(self, markdown_content: str) -> str:
        raise NotImplementedError("markdown_validator does not support async execution")

# # Example usage
# markdown_validator_tool = MarkdownValidatorTool()
# markdown_validator_tool.add_template_regex(r'::: template\n\s*<h3>\s*</h3>\s*<p>\s*</p>\s*\n:::')
 
# # Validate a block of Markdown against the predefined pattern
# result = markdown_validator_tool.run(markdown_content='Your Markdown content here')
# print(result)
