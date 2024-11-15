import os
import requests
import abc

from openai import (
    OpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    )
from ollama import Client
from groq import Groq
from urllib.parse import urlparse

from dygest import prompts


### Prompts ###

PROMPTS = {
  'summarize': prompts.CREATE_SUMMARIES,
  'clean_summaries': prompts.CLEAN_SUMMARIES,
  'create_toc': prompts.CREATE_TOC,
  'create_tldr': prompts.CREATE_TLDR,
  'combine_tldrs': prompts.COMBINE_TLDRS
}


### LLMs ###

class LLMServiceBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = 'BaseService'

    def _handle_api_call(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AuthenticationError as e:
            print(
                "... Authentication Error: Please make sure that your API key "
                "... is stored in your shell environment. ('export OPENAI_API_KEY=sk-proj...') "
            )
            raise e
        except RateLimitError as e:
            print("... Rate Limit Exceeded: Please wait and try again later.")
            raise e
        except APIConnectionError as e:
            print("... Connection Error: Failed to connect to API.")
            raise e
        except OpenAIError as e:
            print(f"... An OpenAI API error occurred: {e}")
            raise e
        except Exception as e:
            print(f"... An unexpected error occurred: {e}")
            raise e

    def prompt(self, template, text_input, model, temperature):
        template_text = PROMPTS.get(template, '')
        return self._handle_api_call(
            self._call_api, template_text, text_input, model, temperature
        )

    def list_models(self):
        return self._handle_api_call(self._api_list_models)
    
    @abc.abstractmethod
    def _call_api(self, template_text, text_input, model, temperature):
        pass

    @abc.abstractmethod
    def _api_list_models(self):
        pass


class OpenAIService(LLMServiceBase):
    def __init__(self):
        super().__init__()
        self.name = 'OpenAI'
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
            )
    
    def _call_api(self, template_text, text_input, model, temperature):    
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"{template_text} {text_input}"
                }
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def _api_list_models(self):
        """
        Fetches the list of available models from the Ollama server.
        """
        response = self.client.models.list()
        model_ids = sorted([model.id for model in response.data])
        print(f'... Available OpenAI models:')
        for model in model_ids:
            print(f"... ... {model}")
        return 


class GroqService(LLMServiceBase):
    def __init__(self):
        super().__init__()
        self.name = 'Groq'
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
            )
    
    def _call_api(self, template_text, text_input, model, temperature):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{template_text} {text_input}"
                }
            ],
            model=model,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def _api_list_models(self):
        """
        Fetches the list of available models from the Ollama server.
        """
        response = self.client.models.list()
        model_ids = sorted([model.id for model in response.data])
        print(f'... Available Groq models:')
        for model in model_ids:
            print(f"... ... {model}")
        return 


class OllamaService(LLMServiceBase):
    def __init__(self, host='http://localhost:11434'):
        super().__init__()
        self.name = 'Ollama'
        self.host = host
        self.client = Client(host=self.host)
        
        if not self._is_url_valid(self.host):
            raise ValueError(f"... Host URL '{self.host}' is not valid.")
        
        if not self._is_server_available():
            raise ConnectionError(
                f"... Cannot connect to the Ollama server at '{self.host}':"
                f"... Plase check if Ollama is running and if the host URL is correct."
                )
        else:
            print(f"... LLM client successfully connected to Ollama server at {self.host}")

    def _is_url_valid(self, url):
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])

    def _is_server_available(self):
        try:
            response = requests.get(f"{self.host}/", timeout=5)
            if response.status_code == 200:
                return True
            else:
                print(f"... Health check failed with status code: {response.status_code}")
                return False
        except {} as e:
            print(f"... Error connecting to Ollama server: {e}")
            return False

    def _call_api(self, template_text, text_input, model, temperature):
        """
        Overrides the abstract method from the base class.
        Adjust parameters as needed based on your implementation.
        """
        response = self.client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"{template_text} {text_input}"
                }
            ],
            options={
                'temperature': temperature
            })
        return response['message']['content']

    
    def _api_list_models(self):
        """
        Fetches the list of available models from the Ollama server.
        """
        try:
            url = f"{self.host}/api/tags"
            response = requests.get(url, timeout=10)  # Increased timeout for reliability
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            
            data = response.json()
            
            if isinstance(data, list):
                models = data
            elif isinstance(data, dict) and 'models' in data:
                models = data['models']
                model_names = sorted([model['name'] for model in models])
                print(f'... Available Ollama models:')
                for model in model_names:
                    print(f"... ... {model}")
            else:
                print(f"... Unexpected response format: {data}")
                models = []

        except requests.exceptions.HTTPError as http_err:
            print(f"... HTTP error occurred while listing models: {http_err}")
            raise
        except requests.exceptions.ConnectionError as conn_err:
            print(f"... Connection error occurred while listing models: {conn_err}")
            raise
        except requests.exceptions.Timeout as timeout_err:
            print(f"... Timeout error occurred while listing models: {timeout_err}")
            raise
        except requests.exceptions.RequestException as req_err:
            print(f"... An error occurred while listing models: {req_err}")
            raise
        except ValueError as json_err:
            print(f"... JSON decoding failed: {json_err}")
            raise
        except Exception as e:
            print(f"... An unexpected error occurred while listing models: {e}")
            raise


### Embedders ### 

class EmbedderBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = 'BaseEmbedder'

    def embed(self, text_input, model):
        try:
            return self._call_api(text_input, model)
        except AuthenticationError as e:
            print(
                "... Authentication Error: Please make sure that your API key "
                "... is stored in your shell environment. ('export OPENAI_API_KEY=sk-...')"
            )
            raise e
        except RateLimitError as e:
            print("... Rate Limit Exceeded: Please wait and try again later.")
            raise e
        except APIConnectionError as e:
            print("... Connection Error: Failed to connect to API.")
            raise e
        except Exception as e:
            print(f"... An unexpected error occurred: {e}")
            raise e

    @abc.abstractmethod
    def _call_api(self, text_input, model):
        pass

class OpenAIEmbedder(EmbedderBase):
    def __init__(self):
        super().__init__()
        self.name = 'OpenAIEmbedder'
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
            )

    def _call_api(self, text_input, model: str = 'text-embedding-3-small'):
        response = self.client.embeddings.create(
            input=text_input,
            model=model
        )
        return {'embedding': response.data[0].embedding}
    

class OllamaEmbedder(EmbedderBase):
    def __init__(self, host='http://localhost:11434'):
        super().__init__()
        self.name = 'OllamaEmbedder'
        self.host = host
        self.client = Client(host=self.host) 

        if not self._is_url_valid(self.host):
            raise ValueError(f"... Host URL '{self.host}' is not valid.")

        if not self._is_server_available():
            raise ConnectionError(
                f"... Cannot connect to the Ollama server at '{self.host}':"
                f"... Please check if Ollama is running and if the host URL is correct."
            )
        else:
            print(f"... Embedding client successfully connected to Ollama server at {self.host}")

    def _is_url_valid(self, url):
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])

    def _is_server_available(self):
        try:
            response = requests.get(f"{self.host}/", timeout=5)
            if response.status_code == 200:
                return True
            else:
                print(f"... Health check failed with status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"... Error connecting to Ollama server: {e}")
            return False

    def _call_api(self, text_input, model: str = 'nomic-embed-text'):
        try:
            response = self.client.embeddings(
                prompt=text_input,
                model=model
                )
            return response
        except Exception as e:
            print(f"... API call failed: {e}")
            raise
