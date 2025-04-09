import re
import os
import time
import typer
import warnings
from rich import print
from typing import Optional
from openai import OpenAIError
from litellm import (
    completion,
    embedding,
    supports_response_schema,
    BadRequestError
    )

def get_api_base(model_name: str) -> Optional[str]:
    """
    Determine the api_base URL based on the LLM service provider.
    """
    api_base_mappings = {
        'ollama': 'http://localhost:11434'
    }
    provider = model_name.split('/')[0].lower()
    return api_base_mappings.get(provider, None)

def get_api_key(model_name: str) -> Optional[str]:
    """
    Determine and get the api_key based on the LLM service provider.
    """
    provider = model_name.split('/')[0].lower()
    if provider != 'ollama':
        api_key = os.getenv(f'{provider.upper()}_API_KEY')
        if not api_key:
            print(f"Please set your {provider.upper()}_API_KEY in your \
environment using: export {provider.upper()}_API_KEY='your_api_key'")
        return api_key
    return None

def set_llm_response_format(
        model: str,
        output_format: str, 
        json_schema: dict
    ) -> dict:
    """
    Construct the response format configuration for an LLM call based on the 
    desired output type and JSON schema.

    The function supports two output formats:
      - 'text': Returns a simple dictionary specifying plain text output.
      - 'json_schema': Checks if the current model supports structured output
        by using the supports_response_schema function. If supported:
          - If no JSON schema is provided, returns a basic 'json_schema' type.
          - If a JSON schema is provided and the provider is 'groq', returns
            the schema under the key 'json_object' (to accommodate groq API
            requirements).
          - Otherwise, returns the schema under the key 'json_schema'.
        If structured output is not supported, it falls back to plain text 
        output.

    Parameters:
        model (str): The LLM model identifier formatted as 'provider/model'.
        output_format (str): Desired output format: 'text' or 'json_schema'.
        json_schema (dict): A dictionary defining the expected JSON schema 
        when structured output is desired.

    Returns:
        dict: A dictionary that specifies the response format configuration 
        for the LLM API call.
    """
    if output_format == 'text':
        response_format = {
            'type': 'text'
            }
    elif output_format == 'json_schema':
        # Check if current model supports structured output
        provider, *_, llm_model = model.split('/')
        if supports_response_schema(llm_model, provider):
            if not json_schema:
                response_format = {
                    'type': 'json_schema'
                    }
            elif json_schema and model.split('/')[0] == 'groq':
                response_format = {
                    'type': 'json_schema',
                    'json_object': json_schema # groq API uses different key
                }
            else:
                response_format = {
                    'type': 'json_schema',
                    'json_schema': json_schema
                } 
        else:
            response_format = {
                'type': 'text'
            }

    return response_format

def remove_reasoning(llm_response: str = None) -> str:
    """
    Clean <think></think> parts from reasoning models responses.
    """
    pattern = r"<think>.*?</think>"
    return re.sub(
        pattern, 
        '', 
        llm_response.choices[0].message.content,
        flags=re.DOTALL
        )

def call_llm(
    prompt: str = None, 
    model: str = None,
    output_format: str = 'text', # 'json_schema' or 'text'
    json_schema: None | dict = None,
    temperature: float = 0.1, 
    api_key: str = None,
    api_base: str = None,
    sleep_time: float = 0.0
    ):
    """
    Call an LLM using litellm's completion function.
    Note: Set the environment variable for the chosen provider's API key.
    
    Parameters:
    - prompt: The prompt template
    - text_input: The input text to be summarized or processed
    - model: The LLM model (e.g., 'openai/gpt-3.5-turbo', 'anthropic/claude-2',
             'huggingface/...', 'ollama/...') 
    - temperature: The LLM sampling temperature
    - api_base: Optional API base URL for some models (like Ollama or custom 
                Hugging Face endpoints)
    """
    # Check for api_key and api_base
    if not api_base or api_key:
        api_base = get_api_base(model)
        api_key = get_api_key(model)

    # Set LLM response output formats ('json_schema' or 'text')
    response_format = set_llm_response_format(model, output_format, json_schema)

    try:
        # Catch httpx deprecation warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                category=DeprecationWarning, 
                module="httpx._content"
                )
            response = completion(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt
                    }],
                temperature=temperature,
                api_key=api_key,
                base_url=api_base,
                response_format=response_format
                )

        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Clean <think> responses from reasoning models like DeepSeek
        response = remove_reasoning(response)

        return response

    except BadRequestError as e:
        print(f"[purple] ...  Error: Please configure the model(s) your are \
using with the correct LLM provider (e.g. 'ollama/llama3.1:latest', \
'openai/gpt-4o-mini).")
        raise typer.Exit(code=1) 
    except OpenAIError as e:
        print(f"[purple] ... Error: {e}")
        raise typer.Exit(code=1) 
    except Exception as e:
        print(f"[purple] ... An unexpected error occurred: {e}")
        raise typer.Exit(code=1) 

def get_embeddings(
    text: str, 
    model: str,
    api_key: str = None,
    api_base: str = None
    ):
    """
    Retrieve embeddings from a given model using litellm.embed.
    
    Parameters:
    - text: The text to embed
    - model: The embedding model (e.g., 'openai/text-embedding-ada-002',
            'huggingface/...')
    - api_base: Optional API base URL for custom endpoints
    
    Returns:
    A dictionary containing the embeddings.
    """
    if not api_base or api_key:
        api_base = get_api_base(model)
        api_key = get_api_key(model)
    
    try:
        response = embedding(
            input=text,
            model=model,
            api_key=api_key,
            api_base=api_base
        )
        
        return response.data[0]['embedding']
    
    except OpenAIError as e:
        print(f"[purple] ... An OpenAI API error occurred: {e}")
        raise typer.Exit(code=1) 
    except Exception as e:
        print(f"[purple] ... An unexpected error occurred: {e}")
        raise typer.Exit(code=1) 
