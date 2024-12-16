import json
import re
import json_repair

from typing import Optional
from pathlib import Path


def resolve_input_dir(filepath: str = None, output_dir: str = None) -> Path:
    """
    Resolve a filepath (file or folder).
    """
    input_path = Path(filepath)

    if not input_path.exists():
        print(f"... Error: The path '{filepath}' does not exist.")
        return

    if output_dir is None:
        if input_path.is_dir():
            resolved_output_dir = input_path
        else:
            resolved_output_dir = input_path.parent
    else:
        resolved_output_dir = Path(output_dir)
        
    return resolved_output_dir

def load_filepath(filepath: str) -> list[Path]:
    """
    Loads a filepath (file or folder).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print("... Please provide a valid filepath.")
        return
    elif filepath.is_dir():
        files_to_process = list(filepath.rglob("*.txt"))
        if not files_to_process:
            print("... No .txt files found in the directory.")
            return
    else:
        files_to_process = [filepath]
    return files_to_process

def load_txt_file(file_path: str) -> str:
    """
    Load files but omit Byte Order Marks (BOM) at start of the string.
    """
    import codecs

    with codecs.open(file_path, 'r', encoding='utf-8-sig') as file:
        return file.read().strip()
    
def remove_hyphens(text: str) -> str:
    return re.sub(r'[=-⸗–]\n', '', text)

def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def replace_underscores_with_whitespace(text: str) -> str:
    return re.sub(r'_', ' ', text)

def chunk_text(text: str, chunk_size: int = 1000) -> tuple[list[str], int]:
    """
    Chunks string by max_tokens and returns text_chunks and token count.
    """
    import tiktoken
    from flair.splitter import SegtokSentenceSplitter

    tokenizer = tiktoken.get_encoding("gpt2") 
    
    token_count = 0
    chunks = []
    current_chunk = []
    current_chunk_length = 0

    # Split text into sentences
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)
    
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence.text)
        sentence_length = len(sentence_tokens)
        token_count += sentence_length
        
        if current_chunk_length + sentence_length > chunk_size:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = sentence_tokens
            current_chunk_length = sentence_length
        else:
            if current_chunk:  # Add a space between sentences
                current_chunk.append(tokenizer.encode(" ")[0])
            current_chunk.extend(sentence_tokens)
            current_chunk_length += sentence_length
    
    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))
    
    return chunks, token_count

def sort_summaries_by_key(summaries: list[dict], key: str) -> list[dict]:
    return sorted(summaries, key=lambda x: x[key])

def chunk_summaries(summaries: list[dict], limit: int = 15, 
                    sort_by_key: str = None) -> list[list[dict]]:
    if sort_by_key:
        summaries = sort_summaries_by_key(summaries, sort_by_key)

    chunked_summaries = []
    for i in range(0, len(summaries), limit):
        chunk = summaries[i:i + limit]
        chunked_summaries.append(chunk)
    return chunked_summaries

def is_valid_json(json_string):
    """
    Check if json_string is valid JSON.
    """
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError as e:
        print(f"... Invalid JSON: {e}")
        return False

def fix_json(json_string: str) -> str:
    """
    Cleans and repairs a string to return valid JSON.
    """
    # Remove trailing Markdown code block markers (```json and ```)
    json_string = re.sub(r"```json\s*", '', json_string, flags=re.IGNORECASE)
    json_string = re.sub(r"```\s*", '', json_string)
    
    # Repair other json errors
    json_string = json_repair.repair_json(json_string)

    return json_string

def validate_and_fix_json(json_string: str) -> Optional[dict]:
    """
    Check if a string is valid JSON. If not attempt to make it valid.
    """
    if is_valid_json(json_string):
        return json.loads(json_string)
    
    print("... Attempting to fix JSON.")
    fixed_json_str = fix_json(json_string)
    
    if is_valid_json(fixed_json_str):
        print("... JSON fixed successfully.")
        print(f'\n{json_string}\n')
        return json.loads(fixed_json_str)
    else:
        print("... Failed to fix JSON.")
        print(f'\n{json_string}\n')
        return None

def validate_summaries(llm_result):
    """
    Validate a collection of LLM summaries for correct JSON format.
    """
    try:
        summaries = validate_and_fix_json(llm_result)
    except json.JSONDecodeError:
        print("... Error decoding JSON.\n")
        print(llm_result)
    return summaries

def print_entities(entities: list[dict]) -> None:
    [print(f"... ... {entity['text']} → {entity['ner_tag']}") for entity in entities]

def print_summaries(summaries: list[dict]) -> None:
    for idx, item in enumerate(summaries):
        print(f"... ... [{idx + 1}] {item['topic']}: {item['summary']} (LOCATION: {item['location']})")
