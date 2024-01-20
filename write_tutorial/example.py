import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools

from langchain.llms import Ollama

default_llm = Ollama(model="openhermes", temperature=0, top_p=0.3)
default_llm = Ollama(model="mistral")
search_tool = DuckDuckGoSearchRun()

# Loading Human Tools
human_tools = load_tools(["human"])

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science in',
  backstory="""You are a Senior Research Analyst at a leading tech think tank.
  Your expertise lies in identifying emerging trends and technologies in AI and
  data science. You have a knack for dissecting complex data and presenting
  actionable insights.""",
  verbose=False,
  allow_delegation=False,
  llm=default_llm,
  # Passing human tools to the agent
  tools=[search_tool]+human_tools
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Tech Content Strategist, known for your insightful
  and engaging articles on technology and innovation. With a deep understanding of
  the tech industry, you transform complex concepts into compelling narratives.""",
  verbose=True,
  llm=default_llm,
  allow_delegation=True
)

# Create tasks for your agents
# Being explicit on the task to ask for human feedback.
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts by
  searching the web for relevant information and latest news.
  Compile your findings in a detailed report. 
  Make sure to check with the human if the draft is good before returning your Final Answer.
  Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights from the researcher's report, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future. 
  Your final answer MUST be the full blog post of at least 3 paragraphs
  and include references to relevant sources.""",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=0
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)