import yaml
import typer
from rich import print
from enum import Enum
from pathlib import Path

DEFAULT_CONFIG = {
    'light_model': None,
    'expert_model': None,
    'embedding_model': None,
    'temperature': 0.1,
    'api_base': None,
    'chunk_size': 1000,
    'ner': False,
    'language': 'auto',
    'precise': False,
    'sleep': 2.5
}

CONFIG_FILE = Path.cwd() / "dygest_config.yaml"
CONFIG = None

def missing_config_requirements(config: dict) -> bool:
    """
    Checks if all required CONFIG fields are set by the user.
    """
    required_fields = [
        'light_model', 
        'expert_model', 
        'embedding_model'
        ]
    return any(config.get(field) is None for field in required_fields)

def load_config() -> dict:
    """
    Loads the configuration from a JSON file if it exists.
    Otherwise, returns the default configuration.
    Converts the 'language' field to the NERlanguages Enum if possible.
    """
    from dygest.ner_utils import NERlanguages

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"[purple]... Error parsing YAML configuration: {e}", err=True)
                
            # Merge with DEFAULT_CONFIG to ensure all keys are present
            merged_config = DEFAULT_CONFIG.copy()
            merged_config.update(config)

            # Convert 'language' from string to Enum if possible
            if 'language' in merged_config and isinstance(merged_config['language'], str):
                try:
                    merged_config['language'] = NERlanguages(merged_config['language'])
                except ValueError:
                    print(f"[purple]... Warning: '{merged_config['language']}' is not a valid NER language. Defaulting to 'auto'.", err=True)
                    merged_config['language'] = NERlanguages.AUTO
            else:
                merged_config['language'] = NERlanguages.AUTO

            return merged_config
            
    else:
        config = DEFAULT_CONFIG.copy()
        config['language'] = 'auto'
        return config

def save_config(config: dict):
    """
    Saves the configuration to a JSON file.
    Converts Enum fields to their string values for JSON compatibility.
    """
    config_to_save = config.copy()
    if isinstance(config_to_save.get('language'), Enum):
        config_to_save['language'] = config_to_save['language'].value

    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_to_save, f, default_flow_style=False, sort_keys=False)
    except yaml.YAMLError as e:
        typer.echo(f"Error writing YAML configuration: {e}", err=True)

def print_config(config: dict):
    """
    Prints the LLM and embedding configuration in a structured format.
    """
    # Check if requrired fields are set
    if missing_config_requirements(config):
        print("[purple]... Please configure all required fields.")
    
    language_val = config.get('language')
    if isinstance(language_val, Enum):
        language_val = language_val.value  # If Enum, get its value
    elif not language_val:
        language_val = 'auto'

    formatted_config = {
        "Light LLM (required)": config.get('light_model') or "None",
        "Expert LLM (required)": config.get('expert_model') or "None",
        "Embedding model (required)": config.get('embedding_model') or "None",
        "Temperature": config.get('temperature', 0.0),
        "Sleep": f"{config.get('sleep', 0.0)} second(s)",
        "Chunk size": config.get('chunk_size', 0),
        "NER": config.get('ner', False),
        "NER precise": config.get('precise', False),
        "NER language": language_val,
        "API Base": config.get('api_base') or "None",
    }
    
    for key, value in formatted_config.items():
        print(f"[deep_pink2]... ... {key:<27}→ [light_pink4]{value}")
