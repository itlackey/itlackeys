from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from git import Repo

# Define a Pydantic model for the input arguments
class GitCommitInput(BaseModel):
   commit_message: str = Field(description="Commit message for the git commit")

# Custom LangChain tool that commits all pending changes in a Git repo
class GitCommitterTool(BaseTool):
   name = "git_committer"
   description = "Commits all pending changes in a Git repository"
   args_schema = GitCommitInput

   def __init__(self, repo_dir, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.repo_dir = repo_dir

   def _run(self, commit_message: str) -> str:
       # Initialize the repository object using GitPython
       repo = Repo(self.repo_dir)

       # Check if there are changes to commit
       if repo.is_dirty(untracked_files=True):
           # Stage all changes
           repo.git.add(A=True)

           # Commit the changes
           repo.index.commit(commit_message)
           
           return f"Changes committed to the repository at {self.repo_dir} with message: '{commit_message}'"
       else:
           return "No changes to commit."

   # If async execution is not supported, raise NotImplementedError
   async def _arun(self, commit_message: str) -> str:
       raise NotImplementedError("git_committer does not support async execution")

# # Example usage
# git_committer_tool = GitCommitterTool('/path/to/your/repo')
# result = git_committer_tool.run(commit_message='Commit all changes')
# print(result)
