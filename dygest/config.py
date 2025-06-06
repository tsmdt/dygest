import typer
from pathlib import Path
from rich import print
from dotenv import dotenv_values, set_key
from dygest.ner_utils import NERlanguages

# Default .env values
DEFAULT_ENV_VALUES = {
    'LIGHT_MODEL': None,
    'EXPERT_MODEL': None,
    'EMBEDDING_MODEL': None,
    'TEMPERATURE': 0.1,
    'SLEEP': 2.5,
    'CHUNK_SIZE': 1000,
    'NER': False,
    'NER_LANGUAGE': 'auto',
    'NER_PRECISE': False,
    'OPENAI_API_KEY': None,
    'GROQ_API_KEY': None,
    'OLLAMA_API_BASE': 'http://localhost:11434'
}

def get_config_value(key: str, default=None, converter=None):
    """
    Helper function to get and convert environment variables.
    """
    value = ENV_VALUES.get(key, default)
    if converter and value is not None:
        try:
            # Handle string 'true'/'false' values first
            if isinstance(value, str):
                if value.lower() == 'true':
                    return True
                if value.lower() == 'false':
                    return False
            # Then handle boolean values
            if isinstance(value, bool):
                return value
            # Handle NERlanguages
            if key == 'NER_LANGUAGE':
                try:
                    return NERlanguages(value)
                except ValueError:
                    print(
                        f"[purple]... Warning: Invalid NER language '{value}'. Using 'auto' instead.",
                        err=True
                    )
                    return NERlanguages.AUTO
            return converter(value)
        except (ValueError, TypeError):
            print(
                f"[purple]... Warning: Invalid value for {key}. Using default.",
                err=True
            )
            return default
    return value

def missing_config_requirements() -> bool:
    """
    Checks if all required configuration fields are set.
    """
    required_fields = ['LIGHT_MODEL', 'EXPERT_MODEL', 'EMBEDDING_MODEL']
    return any(get_config_value(field) == '' for field in required_fields)

# def save_config(config_updates: dict):
#     """
#     Saves the configuration to a .env file using python-dotenv's set_key.
#     Preserves any existing environment variables that aren't part of the config.
#     """
#     try:
#         for key, value in config_updates.items():                
#             if value is None and key in DEFAULT_ENV_VALUES:
#                 value = DEFAULT_ENV_VALUES.get(key)
#             if value is None and key not in DEFAULT_ENV_VALUES:
#                 set_key(ENV_FILE, key, '')
#             else:
#                 set_key(ENV_FILE, key, str(value))

#         # Reload environment variables after saving
#         ENV_VALUES.update(dotenv_values(ENV_FILE))
        
#     except Exception as e:
#         typer.echo(f"Error writing .env configuration: {e}", err=True)

def print_config():
    """
    Prints the current configuration in a structured format.
    """
    # Reload environment variables after saving
    ENV_VALUES.update(dotenv_values(ENV_FILE))
    
    # Check if required fields are set
    if missing_config_requirements():
        print("[purple]... Please configure all required fields.")
    
    for key, value in ENV_VALUES.items():
        if value == '':
            value = 'None'
        if 'API_KEY' in key:
            if not value == 'None':
                print(f"[deep_pink2]... ... {key:<27}→ [light_pink4]{value[:6]}...{value[-4:]}")
            else:
                print(f"[deep_pink2]... ... {key:<27}→ [light_pink4]{value}")
        elif 'MODEL' in key:
            key_name = f"{key} (required)"
            print(f"[deep_pink2]... ... {key_name:<27}→ [light_pink4]{value}")
        else:
            print(f"[deep_pink2]... ... {key:<27}→ [light_pink4]{value}")

# Initialize environment variables
ENV_FILE = Path.cwd() / ".env"
if not ENV_FILE.exists():
    ENV_FILE.touch()
    # Write all default values to a fresh .env file
    for key, value in DEFAULT_ENV_VALUES.items():
        if value is None:
            set_key(ENV_FILE, key, '')
        else:
            set_key(ENV_FILE, key, str(value))
    print("[purple]... New .env file created with default values.")

ENV_VALUES = dotenv_values(ENV_FILE)
