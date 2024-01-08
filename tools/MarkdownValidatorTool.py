import markdown
import re
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Define a Pydantic model for the input arguments
class MarkdownValidationInput(BaseModel):
 markdown_content: str = Field(description="Markdown content to be validated")

# Custom LangChain tool that validates Markdown syntax and compares rendered HTML to a predefined pattern
class MarkdownValidatorTool(BaseTool):
 name = "markdown_validator"
 description = "Validates Markdown syntax and compares rendered HTML to a predefined pattern"
 args_schema = MarkdownValidationInput
 template_regex = None

 def __init__(self, template_name, html_pattern, *args, **kwargs):
     super().__init__(*args, **kwargs)
     self.name = f'{template_name}_{self.name}'  # Set the name to include the template_name
     self.template_regex = re.compile(html_pattern)

 def _run(self, markdown_content: str) -> str:
     # Render the Markdown to HTML
     rendered_html = markdown.markdown(markdown_content)

     # Validate the rendered HTML against the predefined template using regex
     return self.template_regex.fullmatch(rendered_html) is not None

 # If async execution is not supported, raise NotImplementedError
 async def _arun(self, markdown_content: str) -> str:
     raise NotImplementedError("markdown_validator does not support async execution")

# # Example usage
# markdown_validator_tool = MarkdownValidatorTool('example_template', r'<div>\s*<p>\s*</p>\s*</div>')

# # Validate a block of Markdown against the predefined pattern
# result = markdown_validator_tool.run(markdown_content='Your Markdown content here')
# print(result)
