import os
from groq import Groq
from src import templates


PROMPTS = {
  'summarize': templates.CREATE_SUMMARIES,
  'clean_summaries': templates.CLEAN_SUMMARIES,
  'ner': templates.GET_ENTITIES
}


def prompt_groq(
    template: str = 'summarize', 
    text_input: str = None, 
    model: str = 'llama-3.1-70b-versatile',
    temperature: float = 0.1
    ):
    template = PROMPTS.get(template, '')
    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{template} {text_input}"
            }
        ],
        model=model,
        temperature=temperature,
    )
    
    return response.choices[0].message.content
