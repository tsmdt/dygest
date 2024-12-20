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

def chunk_text(
    text: str, 
    chunk_size: int = 1000
) -> tuple[dict, int, dict]:
    """
    Chunks the text sentence-wise by max_tokens.
    
    Returns:
    - chunks: A dictionary of the form:
        {
          'chunk_01': {
              'text': '...',
              's_ids': ['S1', 'S2', ...]
          },
          'chunk_02': {...},
          ...
        }
    - token_count: Total number of tokens in the input text.
    - sentence_offsets: A dict mapping "S<id>" to the starting character 
      index of that sentence in 'text'.
    """
    import tiktoken
    from flair.splitter import SegtokSentenceSplitter
    import re

    tokenizer = tiktoken.get_encoding("gpt2")

    token_count = 0
    chunks = {}
    chunk_index = 1
    current_chunk = []
    current_chunk_length = 0
    current_chunk_s_ids = []

    # Remove all linebreaks from the input text
    text = re.sub(r'[\n\t\r]', ' ', text)

    # Split text into sentences
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)

    # Map sentence IDs to their start offsets in the input text
    sentence_offsets = {}
    current_search_pos = 0
    sentence_id = 1

    for sentence in sentences:
        # Find the offset of this sentence in the original text
        start_index = text.index(sentence.text, current_search_pos)
        sentence_offsets[f"S{sentence_id}"] = start_index
        current_search_pos = start_index + len(sentence.text)
        sentence_id += 1

    # Chunk text based on token limits
    current_chunk_s_ids = []
    global_sentence_id = 1  # reset to 1 to track sentences globally

    for sentence in sentences:
        s_id = f"S{global_sentence_id}"
        sentence_tokens = tokenizer.encode(sentence.text)
        sentence_length = len(sentence_tokens)
        token_count += sentence_length

        if current_chunk_length + sentence_length > chunk_size:
            # Finish the current chunk
            if current_chunk:
                chunk_key = f"chunk_{chunk_index:02d}"
                chunks[chunk_key] = {
                    'text': tokenizer.decode(current_chunk),
                    's_ids': current_chunk_s_ids
                }
                chunk_index += 1
            
            # Start a new chunk
            current_chunk = sentence_tokens[:]
            current_chunk_length = sentence_length
            current_chunk_s_ids = [s_id]
        else:
            # Add sentence to the current chunk
            if current_chunk:  # Add a space between sentences
                space_token = tokenizer.encode(" ")[0]
                current_chunk.append(space_token)
                current_chunk_length += 1
            current_chunk.extend(sentence_tokens)
            current_chunk_length += sentence_length
            current_chunk_s_ids.append(s_id)

        global_sentence_id += 1

    # Add the final chunk if present
    if current_chunk:
        chunk_key = f"chunk_{chunk_index:02d}"
        chunks[chunk_key] = {
            'text': tokenizer.decode(current_chunk),
            's_ids': current_chunk_s_ids
        }

    return chunks, token_count, sentence_offsets

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

def print_toc_topics(toc_part: list[dict]) -> None:
    for idx, item in enumerate(toc_part):
        print(f"... [{idx + 1}] {item['topic']}: {item['summary']} (LOCATION: {item['location']})")

def print_summaries(summary: str) -> None:
    print(f"... {summary}")
    