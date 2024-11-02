import os
import sys
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_URL = os.getenv('OPENAI_API_URL')

# Initialize Qdrant and OpenAI clients
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_URL

# System prompt and template
SYSTEM_PROMPT = (
    "You are an AI assistant specialized in code analysis. "
    "Provide accurate and concise information based on the provided context. "
    "Do not include information not present in the context. "
    "If additional information is required, respond with 'QUERY:' followed by the query. "
    "If specific file content is needed, respond with 'FILE:' followed by the file path."
)

PROMPT_TEMPLATE = (
    "{system_prompt}\n\n"
    "Instruction: {instruction}\n\n"
    "Context:\n{context}\n\n"
    "Response:"
)

# Functions for interacting with the LLM and Qdrant
def query_llm(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        stop=None
    )
    return response.choices[0].text.strip()

def query_qdrant(collection_name, query_text, limit=5):
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_text,
        limit=limit
    )
    return [point.payload for point in search_result]

# Main processing loop
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py '<instruction>'")
        sys.exit(1)

    instruction = sys.argv[1]
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "your_collection_name")  # Set collection name as environment variable or hardcode it
    context = ""
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        # Construct prompt
        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            instruction=instruction,
            context=context
        )

        # Query the LLM
        response = query_llm(prompt)

        # Process LLM response
        if response.startswith("QUERY:"):
            query_text = response[len("QUERY:"):].strip()
            additional_data = query_qdrant(collection_name, query_text)
            context += "\n".join(additional_data)
        elif response.startswith("FILE:"):
            file_path = response[len("FILE:"):].strip()
            try:
                with open(file_path, 'r') as file:
                    file_content = file.read()
                context += file_content
            except FileNotFoundError:
                context += f"\n[Error: File '{file_path}' not found.]"
        else:
            print("Final Response:", response)
            break

        # Increment iteration count
        iteration += 1

    # Final iteration fallback
    if iteration == max_iterations:
        print("Max iterations reached. Sending final context to LLM.")
        final_prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            instruction=instruction,
            context=context
        )
        final_response = query_llm(final_prompt)
        print("Final Response:", final_response)

if __name__ == "__main__":
    main()
