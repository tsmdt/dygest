import os
import requests

from ollama import Client
from openai import OpenAI
from groq import Groq
from abc import ABC, abstractmethod

from dygest import templates


PROMPTS = {
  'summarize': templates.CREATE_SUMMARIES,
  'clean_summaries': templates.CLEAN_SUMMARIES,
}


class LLMServiceBase(ABC):
    @abstractmethod
    def prompt(self, template, text_input, model, temperature):
        pass


class GroqService(LLMServiceBase):
    def __init__(self):
        self.name = 'Groq'
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
            )
    
    def prompt(self, template, text_input, model, temperature):
        template_text = PROMPTS.get(template, '')
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
    
    def list_models(self):
        api_key = os.environ.get("GROQ_API_KEY")
        url = "https://api.groq.com/openai/v1/models"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)

        return response.json()


class OpenAIService(LLMServiceBase):
    def __init__(self):
        self.name = 'OpenAI'
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
            )
    
    def prompt(self, template, text_input, model, temperature):
        template_text = PROMPTS.get(template, '')
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


class OllamaService(LLMServiceBase):
    def __init__(self):
        self.name = 'Ollama'
        self.client = Client(
            host='http://localhost:11434'
            )
    
    def prompt(self, template, text_input, model, temperature):
        template_text = PROMPTS.get(template, '')

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
