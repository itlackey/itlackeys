import os
import sys
import openai
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import transformers

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'dimm-city-page')
OPENAI_API_URL = os.getenv('OPENAI_API_URL', 'http://localhost:11434/v1/')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.2')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
#encoding = tiktoken.encoding_for_model(EMBED_MODEL)
encoding = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

MAX_TOKENS = 4096
RESERVED_TOKENS = 500  # Reserve tokens for the response and other parts
MAX_CONTEXT_TOKENS = MAX_TOKENS - RESERVED_TOKENS

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY
openai.base_url = OPENAI_API_URL

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# System prompt for the LLM
SYSTEM_PROMPT = (
    "You are an AI assistant specialized in code analysis. "
    "Provide accurate and concise information based on the provided context. "
    "Do not include information not present in the context. "
    "If additional information is required, respond with 'QUERY:' followed by the query to execture for more information. "
    "If specific file content is needed, respond with 'FILE:' followed by the file path."
)

INSTUCTION_PROMPT = (
    "You are an AI assistant specialized in code analysis. "
    "Provide accurate and concise information based ONLY the provided"
    "Here are your instrctions:"
    "{instruction}"
    "Here is your context:"
    "{context}"
    "If you you need additional infomration, respond with ONLY 'QUERY:' followed by the query to execture for more information."    
)

# Function to generate embeddings using OpenAI's embedding model
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding

# Function to interact with the LLM using OpenAI's ChatCompletion endpoint
def query_llm(instruction, context):
    prompt = INSTUCTION_PROMPT.format(instruction=instruction, context=context)
    #print(prompt)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        #TODO: all settings to be set easily
        # max_tokens=500,
        # temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# Function to query the Qdrant database
def query_qdrant(collection_name, query_text, limit=5):
    query_embedding = get_embedding(query_text)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit,        
    )
    return [point.payload.get('content', '') for point in search_result]

# Function to calculate the number of tokens in a string
def num_tokens_from_string(string):
    num_tokens = len(encoding.encode(string))
    return num_tokens

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py '<instruction>'")
        sys.exit(1)

    instruction = sys.argv[1]
    context = ""
    max_iterations = 5
    iteration = 0


    while iteration < max_iterations:
        # Manage context length to stay within token limits
        context_tokens = num_tokens_from_string(context)
        if context_tokens > MAX_CONTEXT_TOKENS:
            # Truncate the context by removing the oldest information
            
            tokens = encoding.encode(context)
            truncated_tokens = tokens[-MAX_CONTEXT_TOKENS:]
            context = encoding.decode(truncated_tokens)

        response = query_llm(instruction, context)

        if response.find("QUERY:") > -1:
            query_text = response[len("QUERY:"):].strip()
            additional_data = query_qdrant(QDRANT_COLLECTION, query_text)
            context += "\n" + "\n".join(additional_data)
        elif response.find("FILE:") > -1:
            file_path = response[len("FILE:"):].strip()
            try:
                with open(file_path, 'r') as file:
                    file_content = file.read()
                context += "\n" + file_content
            except FileNotFoundError:
                context += f"\n[Error: File '{file_path}' not found.]"
        else:
            print("Final Response:", response)
            break
        print(response)
        iteration += 1

    if iteration == max_iterations:
        print("Max iterations reached. Final context and instruction sent to LLM.")
        final_response = query_llm(instruction, context)
        print("Final Response:", final_response)

if __name__ == "__main__":
    main()
