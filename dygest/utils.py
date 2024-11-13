import json
import re
import codecs
import tiktoken
import json_repair
import numpy as np

from itertools import combinations
from flair.splitter import SegtokSentenceSplitter
from typing import Optional
from pathlib import Path


class SummaryProcessor:
    def __init__(
            self, 
            summaries, 
            embedding_service, 
            embedding_model,
            key='topic', 
            threshold=0.8,
            verbose=False
            ):
        """
        Initialize the SummaryProcessor and process the summaries.

        Args:
            summaries (list[dict]): List of summary dictionaries.
            embedding_model (str): The name of the embedding model to use.
            key (str): The key in the dictionary to embed (e.g., 'topic').
            threshold (float): Cosine similarity threshold to consider topics as similar.
        """
        self.embedding_service = embedding_service
        self.embedding_model = embedding_model
        self.key = key
        self.threshold = threshold
        self.summaries = summaries
        self.embedded_summaries = {}
        self.filtered_summaries = []
        self.verbose = verbose        

    def embed_summaries(self):
        """
        Embed summaries with Ollama embeddings.

        Populates the `embedded_summaries` dictionary.
        """
        for summary in self.summaries:
            text = summary[self.key]
            response = self.embedding_service.embed(text, self.embedding_model)
            self.embedded_summaries[text] = np.array(response['embedding'])

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calculate the cosine similarity between two vectors.

        Args:
            vec1 (np.array): First embedding vector.
            vec2 (np.array): Second embedding vector.

        Returns:
            float: Cosine similarity score.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def remove_similar_summaries(self):
        """
        Remove similar summaries based on cosine similarity of their topics.

        Returns:
            list[dict]: Filtered list of summaries with similar topics removed.
        """
        similar_topics = []

        # Generate all unique pairs of topics
        for (topic1, emb1), (topic2, emb2) in combinations(self.embedded_summaries.items(), 2):
            similarity = self.cosine_similarity(emb1, emb2)
            if similarity >= self.threshold:
                similar_topics.append({
                    'topic_1': topic1,
                    'topic_2': topic2,
                    'similarity_score': similarity
                })

        # Identify Topics to Remove (topic_2 in each similar pair)
        topics_to_remove = set(pair['topic_2'] for pair in similar_topics)

        # Filter Out Summaries with Topics to Remove
        filtered_summaries = [summary for summary in self.summaries if summary[self.key] not in topics_to_remove]

        # Display Similar Topics Identified
        if self.verbose:
            if similar_topics:
                print("... Similar Topics Identified:")
                for pair in similar_topics:
                    print(f"... '{pair['topic_1']}' <--> '{pair['topic_2']}' with similarity score of {pair['similarity_score']:.4f}")
            else:
                print("... No similar topics found above the threshold.")

        return filtered_summaries

    def get_filtered_summaries(self):
        """
        Get the filtered summaries.

        Returns:
            list[dict]: Filtered summaries.
        """
        self.embed_summaries()
        self.filtered_summaries = self.remove_similar_summaries()
        return self.filtered_summaries


def load_filepath(filepath: str) -> list[Path]:
    """
    Loads a filepath (file or folder).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print("... Please provide a valid filepath.")
        return
    elif filepath.is_dir():
        files_to_process = list(filepath.glob("*.txt"))
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
    with codecs.open(file_path, 'r', encoding='utf-8-sig') as file:
        return file.read().strip()
    
def remove_hyphens(text: str) -> str:
    return re.sub(r'[=-⸗–]\n', '', text)

def chunk_text(text: str, chunk_size: int = 4000) -> tuple[list[str], int]:
    """
    Chunks string by max_tokens and returns text_chunks and token count.
    """
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
    [print(f"... {entity['text']} → {entity['ner_tag']}") for entity in entities]

def print_summaries(summaries: list[dict]) -> None:
    for idx, item in enumerate(summaries):
        print(f"... [{idx + 1}] {item['topic']}: {item['summary']} (LOCATION: {item['location']})")
