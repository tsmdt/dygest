import re
import time
import typer
import asyncio
from typing import Tuple
from rich import print
from openai import OpenAIError
from litellm import (
    completion,
    acompletion,
    embedding,
    aembedding,
    supports_response_schema,
    BadRequestError
    )
from dotenv import dotenv_values
from dygest.config import ENV_FILE


def get_provider_config(model_name: str) -> Tuple[str, str, str]:
    """
    Return model, api_base, api_key
    Expects a model_name of the following structure:
    - "openai/provider/model" (for providers not in the litellm model list)
        - Example: openai/exampleprovider/qwen2.5:latest
    - "provider/model" (for litellm providers)
        - Example: ollama/qwen2.5:latest
    """
    ENV_VALUES = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    
    provider = model_name.split('/')
    
    if len(provider) == 3:
        # openai/exampleprovider/qwen2.5:latest
        prefix = provider[0]
        custom_provider = provider[1]
        model = f"{prefix}/{provider[2]}"
    elif len(provider) == 2:
        # ollama/qwen2.5:latest
        custom_provider = provider[0]
        model = f"{custom_provider}/{provider[1]}"
    else:
        typer.secho(
            "... Error: Model name must be in format 'provider/model' (e.g. 'ollama/qwen2.5:latest') "
            "or 'openai/provider/model' (e.g. 'openai/exampleprovider/qwen2.5:latest')",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
        
    # Load custom api config from .env
    api_base = ENV_VALUES.get(f'{custom_provider.upper()}_API_BASE', None)
    api_key = ENV_VALUES.get(f'{custom_provider.upper()}_API_KEY', None)
    
    return model, api_base, api_key

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
            a simplified format that Groq can handle.
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
            elif json_schema and provider == 'groq':
                # Simplified format for Groq
                response_format = {
                    'type': 'text'
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
        model, api_base, api_key = get_provider_config(model)

    # Set LLM response output formats ('json_schema' or 'text')
    response_format = set_llm_response_format(model, output_format, json_schema)

    try:
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
'openai/gpt-4o-mini)'.")
        raise typer.Exit(code=1) 
    except OpenAIError as e:
        print(f"[purple] ... Error: {e}")
        raise typer.Exit(code=1) 
    except Exception as e:
        print(f"[purple] ... An unexpected error occurred: {e}")
        raise typer.Exit(code=1) 

async def call_allm(
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
    Async version of call_llm using litellm's acompletion function.
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
        model, api_base, api_key = get_provider_config(model)

    # Set LLM response output formats ('json_schema' or 'text')
    response_format = set_llm_response_format(model, output_format, json_schema)

    try:
        response = await acompletion(
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
            await asyncio.sleep(sleep_time)
        
        # Clean <think> responses from reasoning models like DeepSeek
        response = remove_reasoning(response)

        return response

    except BadRequestError as e:
        print(f"[purple] ...  Error: Please configure the model(s) your are \
using with the correct LLM provider (e.g. 'ollama/llama3.1:latest', \
'openai/gpt-4o-mini)'.")
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
        model, api_base, api_key = get_provider_config(model)
    
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

async def get_aembeddings(
    text: str, 
    model: str,
    api_key: str = None,
    api_base: str = None
    ):
    """
    Async version of get_embeddings using litellm.aembed.
    
    Parameters:
    - text: The text to embed
    - model: The embedding model (e.g., 'openai/text-embedding-ada-002',
            'huggingface/...')
    - api_base: Optional API base URL for custom endpoints
    
    Returns:
    A dictionary containing the embeddings.
    """
    if not api_base or api_key:
        model, api_base, api_key = get_provider_config(model)
    
    try:
        response = await aembedding(
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
