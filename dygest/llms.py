import os
import requests
import abc

from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    )
from ollama import Client
from groq import Groq
from urllib.parse import urlparse

from dygest import prompts


PROMPTS = {
  'summarize': prompts.CREATE_SUMMARIES,
  'clean_summaries': prompts.CLEAN_SUMMARIES,
  'create_toc': prompts.CREATE_TOC
}


class LLMServiceBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = 'BaseService'
    
    def prompt(self, template, text_input, model, temperature):
        template_text = PROMPTS.get(template, '')
        try:
            return self._call_api(template_text, text_input, model, temperature)
        except AuthenticationError as e:
            print(
                f"... Authentication Error: Please make soure that your API key "
                f"is stored in your shell environment. ('export OPENAI_API_KEY=sk-proj...') "
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
    def _call_api(self, template_text, text_input, model, temperature):
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


class OllamaService(LLMServiceBase):
    def __init__(self, host='http://localhost:11434'):
        super().__init__()
        self.name = 'Ollama'
        self.host = host
        self.client = Client(host=self.host)
        
        if not self._is_url_valid(self.host):
            raise ValueError(f"The URL '{self.host}' is not valid.")
        
        if not self._is_server_available():
            raise ConnectionError(f"Cannot connect to the Ollama server at '{self.host}'.")
        else:
            print(f"Successfully connected to Ollama server at '{self.host}'.")

    def _is_url_valid(self, url):
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])

    def _is_server_available(self):
        try:
            response = requests.get(f"{self.host}/", timeout=5)
            if response.status_code == 200:
                return True
            else:
                print(f"Health check failed with status code: {response.status_code}")
                return False
        except {} as e:
            print(f"Error connecting to Ollama server: {e}")
            return False

    def _call_api(self, template_text, text_input, model, temperature):
        try:
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
        except Exception as e:
            print(f"API call failed: {e}")
            raise
