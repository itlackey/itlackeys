
def create_llm(model, model_type):
    if model_type == "openai":
        from langchain.llms import OpenAI
        return OpenAI(model_name=model)
    elif model_type == "ollama":
        from langchain.llms import Ollama
        return Ollama(model=model)
    return None
