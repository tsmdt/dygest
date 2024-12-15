import time
from typing import Optional
from openai import OpenAIError
from litellm import completion, embedding
from dygest import prompts

PROMPTS = {
    'get_topics': prompts.GET_TOPICS,
    # 'clean_topics': prompts.CLEAN_TOPICS,
    'create_toc': prompts.CREATE_TOC,
    'create_tldr': prompts.CREATE_TLDR,
    'combine_tldrs': prompts.COMBINE_TLDRS
}

def get_api_base(model_name: str) -> Optional[str]:
    """
    Determine the api_base URL based on the LLM service provider.
    """
    api_base_mappings = {
        'ollama': 'http://localhost:11434'
    }
    provider = model_name.split('/')[0].lower()
    return api_base_mappings.get(provider, None)

def call_llm(
    template: str, 
    text_input: str, 
    model: str, 
    temperature: float = 0.1, 
    api_key: str = None,
    api_base: str = None,
    sleep_time: float = 0.0
    ):
    """
    Call an LLM using litellm's completion function.
    Just set the environment variable for the chosen provider's API key.
    
    Parameters:
    - template: The prompt template key (e.g., 'summarize')
    - text_input: The input text to be summarized or processed
    - model: The LLM model (e.g., 'openai/gpt-3.5-turbo', 'anthropic/claude-2', 'huggingface/...', 'ollama/...') 
    - temperature: The LLM sampling temperature
    - api_base: Optional API base URL for some models (like Ollama or custom Hugging Face endpoints)
    """
    template_text = PROMPTS.get(template, '')
    messages = [{"role": "user", "content": f"{template_text} {text_input}"}]

    if not api_base:
        api_base = get_api_base(model)

    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base
        )

        if sleep_time > 0:
            time.sleep(sleep_time)
        
        return response.choices[0].message.content

    except OpenAIError as e:
        print(f"... An OpenAI API error occurred: {e}")
    except Exception as e:
        print(f"... An unexpected error occurred: {e}")

def get_embeddings(
    text: str, 
    model: str,
    api_base: str = None
    ):
    """
    Retrieve embeddings from a given model using litellm.embed.
    
    Parameters:
    - text: The text to embed
    - model: The embedding model (e.g., 'text-embedding-ada-002', 'huggingface/...')
    - api_base: Optional API base URL for custom endpoints
    
    Returns:
    A dictionary containing the embeddings.
    """
    if not api_base:
        api_base = get_api_base(model)
    
    try:
        response = embedding(
            input=text,
            model=model,
            api_base=api_base
        )
        
        return response.data[0]['embedding']
    
    except OpenAIError as e:
        print(f"... An OpenAI API error occurred: {e}")
    except Exception as e:
        print(f"... An unexpected error occurred: {e}")
