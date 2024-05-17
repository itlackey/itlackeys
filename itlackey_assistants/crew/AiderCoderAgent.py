
from crewai import Agent
from tools.AiderCoderTool import AiderCoderTool

def aider_coder_agent(llm=None):
    return  Agent(
        role='Senior Software Engineer',
        goal='Update files using the Aider Coder based on the instructions provided to me.',
        backstory="""You are a renowned Software Engineer, known for your elegate and robust coding. With a deep understanding of
        software engineering in many languages, you are able to write high quality code across multiple platforms, frameworks, and languages.""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
        tools=[AiderCoderTool()],
    )