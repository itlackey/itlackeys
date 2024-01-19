import sys
from crewai import Agent, Task
import os
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI

os.environ.clear()
load_dotenv()

# default_llm = AzureChatOpenAI(
#     openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-07-01-preview"),
#     azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt35"),
#     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://<your-endpoint>.openai.azure.com/"),
#     api_key=os.environ.get("AZURE_OPENAI_KEY")
# )

default_llm = ChatOpenAI(openai_api_base="http://localhost:8001/v1",
                        openai_api_key=os.environ.get("OPENAI_API_KEY", "local"),
                        temperature=0.1,                        
                        model_name="mistral")


def tell_story(topic):
    # Define general agent
    general_agent  = Agent(role='Story Teller',
                    goal="To create fantastic fictional stories.",
                    backstory="""You are an expert fiction writer and story teller.""",
                    allow_delegation=False, 
                    verbose=True,
                    llm=default_llm)


    # Define Tasks Using Crew Tools
    syntax_review_task = Task(description=f"""
                Write a creative story about {topic}""",
                agent=general_agent)

    print(f"Topic: {topic}")
    story = syntax_review_task.execute()

    print(f"Story: {story}")
    return story

# If called directly from the command line take the first argument as the filename
if __name__ == "__main__":

    if len(sys.argv) > 1:
        topic = sys.argv[1]
        processed_document = tell_story(topic)
        print(processed_document)




### Example Validation Results

##ikawrakow/open-hermes-2.5-mistral-7b-quantized-gguf/oh-2.5-m7b-q51.gguf
        
# model_name="oh-2.5m7b-q51",
# temperature=0.1,           
# top_p=0.3

# Here is a list of changes that the developer should make to the README.md file based on the validation results from the markdown_validation_tool:

# 1. Add a # at the beginning of the first line to make it a level 1 heading (e.g., "# My Project").
# 2. Break line 3 into two or more shorter lines, as it is currently too long (127 characters).
# 3. Break line 7 into two or more shorter lines, as it is currently too long (94 characters).
# 4. Break line 44 into multiple shorter lines or rephrase the content to make it more concise, as it is extremely long (234 characters).
# 5. Repeat steps 1-4 for the remaining PyMarkdownScanFailure entries in the validation results list.
# 6. Ensure that the README.md file follows proper Markdown syntax and is well-structured, with appropriate headings, paragraphs, and lists as needed.
# 7. Add a brief introduction to the project at the beginning of the README.md file, explaining its purpose and any key features or functionalities.
# 8. Review the overall readability and clarity of the README.md file, making adjustments as necessary to ensure it is easy for others to understand and navigate.

## TheBloke/dolphin-2.6-mistral-7b-dpo.Q4_K_M.gguf        
# model_name="dolphin-2.6-mistral-dpo-7b-q4_k_m",
# temperature=0.1,           
# top_p=0.3
        
# The markdown validation tool has identified three issues in your README.md file. Here's a summary of the changes you should make:

# 1. Add a first-line heading, such as `# Heading`, at the beginning of the README.md file to comply with Rule ID: MD041.
# 2. Break long lines into shorter ones to ensure they don't exceed 80 characters. You can use soft wraps or add line breaks where necessary. This will help you adhere to Rule ID: MD013.
# 3. Review each line in the README.md file and ensure they don't exceed 80 characters. If necessary, break up long lines or rephrase sentences to fit within the character limit. This will also help you comply with Rule ID: MD013.