import typer
import re
import traceback
import numpy as np
from rich import print
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from itertools import combinations, chain
from typing import Optional
from langdetect import detect, DetectorFactory

from dygest import llms, utils, ner_utils, prompts, json_schemas
from dygest.translations import LANGUAGES
from dygest.output_utils import ExportFormats, HTML_Templates
from dygest.ner_utils import NERlanguages


@dataclass
class DygestBaseParams:
    filepath: str
    output_dir: str = "./output"
    light_model: Optional[str] = None
    expert_model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: float = 0.1
    sleep: float = 0
    chunk_size: int = 1000
    add_toc: bool = False
    add_summaries: bool = False
    add_keywords: bool = False
    add_ner: bool = True
    sim_threshold: float = 0.8
    provided_language: NERlanguages = NERlanguages.AUTO
    precise: bool = False
    verbose: bool = False
    export_metadata: bool = False
    export_format: ExportFormats = ExportFormats.HTML
    html_template: HTML_Templates = HTML_Templates.TABS


@dataclass
class DygestProcessor(DygestBaseParams):
    ner_tagger: Optional[object] = field(default=None, init=False)
    text: Optional[str] = field(default=None, init=False)
    chunks: Optional[dict] = field(default=None, init=False)
    sentence_offsets: Optional[str] = field(default=None, init=False)
    toc: Optional[list] = field(default=None, init=False)
    summaries: Optional[str] = field(default=None, init=False)
    keywords: Optional[list] = field(default=None, init=False)
    entities: Optional[list] = field(default=None, init=False)
    token_count: Optional[int] = field(default=None, init=False)
    language_ISO: NERlanguages = field(default=None, init=False)
    language_string: str = field(default=None, init=False)
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            if self.verbose:
                print(f"... Created output directory at {self.output_dir}") 

    def run_language_detection(self) -> str:
        """
        Get language of text to set the correct NER model.
        
        Returns:
            str: An ISO 639-1 language code ('en', 'de', 'es' ...)
        """
        language = self.provided_language
            
        if language == 'auto':
            DetectorFactory.seed = 0
            language_ISO = detect(self.text[:500])

            if self.verbose:
                print(f"... Detected language '{language_ISO}'")
        else:
            language_ISO = language
        return language_ISO   
        
    def run_ner(self) -> list[str] | None:
        """
        Run Named Entity Recognition with flair framework on the text.
        """
        # Load NER tagger if not already loaded or if language has changed
        if self.ner_tagger is None or self.provided_language == 'auto':
            self.ner_tagger = ner_utils.load_tagger(
                language=self.language_ISO,
                precise=self.precise
            )
            if self.verbose:
                print(f"... Loaded NER tagger for language: {self.language_ISO}")

            # Run Named Entity Recognition (NER)
            entities = ner_utils.get_flair_entities(self.text, self.ner_tagger)
            all_entities = ner_utils.update_entity_positions(entities, self.text)

            if self.verbose:
                print(f"... ENTITIES FOR DOC:")
                utils.print_entities(all_entities)
            
            return all_entities
        else:
            return []
        
    def create_toc(self) -> dict:
        """
        Create a Table of Contents (TOC) for the provided file.
        """
        def is_valid_location(location: str) -> bool:
            """
            Validate if a topic location follows this sentence id ("s_id")
            structure: "S362"
            """
            return bool(re.fullmatch(r'S[1-9]\d*$', location))
        
        def fix_wrong_location(location: str) -> str:
            """
            Try to fix a malformed topic location structure.
            
            Transforms strings like:
            - "S<019>" to "S19"
            - "S09" to "S9"
            """
            if re.fullmatch(r'S<\d+>', location):
                intermediate = re.sub(r'[<>]', '', location)
                fixed = re.sub(r'S0+', 'S', intermediate)
                return None if fixed == 'S' else fixed
            elif re.fullmatch(r'S0+\d+', location):
                fixed = re.sub(r'S0+', 'S', location)
                return None if fixed == 'S' else fixed
            return None

        def align_toc_part(toc_part: list[dict], chunk: dict) -> list[dict]:
            """
            Align the topic locations (e.g. "S7") to match them with the
            correct chunk sentence IDs. The goal is to ensure the 'location'
            is one of the actual global sentence IDs present in the chunk.
            """
            # Check for a structured json output ('topics' key is present)
            if 'topics' in toc_part:
                toc_part = toc_part['topics']

            chunk_s_ids_global = chunk['s_ids']  # List of strings like ['S47', 'S48', 'S49']
            if not chunk_s_ids_global:
                if self.verbose:
                    print("... Warning: Empty chunk_s_ids_global in align_toc_part.")
                return []

            # Convert global sentence IDs of the chunk to numbers for comparison
            try:
                chunk_s_nums_global = [int(sid[1:]) for sid in chunk_s_ids_global]
            except ValueError as e:
                if self.verbose:
                    print(f"... Warning: Could not parse sentence numbers from chunk_s_ids_global: {chunk_s_ids_global}. Error: {e}")
                return toc_part # Return original if parsing fails, or handle error appropriately

            default_location = chunk_s_ids_global[0]

            aligned_toc_part = []
            for topic in toc_part:
                original_llm_location = topic.get('location')
                current_location_str = None

                # Validate and fix the LLM's proposed location
                if original_llm_location and is_valid_location(original_llm_location):
                    current_location_str = original_llm_location
                elif original_llm_location:
                    fixed_loc = fix_wrong_location(original_llm_location)
                    if fixed_loc and is_valid_location(fixed_loc):
                        current_location_str = fixed_loc

                final_location_str = default_location # Default to the start of the chunk

                if current_location_str:
                    # Case 1: LLM's location is a valid global ID within this chunk's sentences
                    if current_location_str in chunk_s_ids_global:
                        final_location_str = current_location_str
                    else:
                        # Case 2: LLM's location is not directly in this chunk's s_ids.
                        # Attempt to find the numerically closest valid s_id from this chunk.
                        try:
                            llm_loc_num = int(current_location_str[1:])
                            if chunk_s_nums_global: # Ensure list is not empty
                                closest_s_num = min(chunk_s_nums_global, key=lambda x: abs(x - llm_loc_num))
                                final_location_str = f"S{closest_s_num}"
                            # If chunk_s_nums_global is empty, final_location_str remains default_location
                        except ValueError:
                            # Could not parse number from LLM's location string, stick to default_location
                            if self.verbose:
                                print(f"... Warning: Could not parse number from LLM location '{current_location_str}'. Using default '{default_location}'.")
                else:
                    # No valid or fixable location string from LLM, stick to default_location
                    if self.verbose and original_llm_location:
                         print(f"... Warning: LLM location '{original_llm_location}' was invalid and unfixable. Using default '{default_location}'.")


                topic['location'] = final_location_str
                aligned_toc_part.append(topic)

            return aligned_toc_part
           
        # TOC processing
        print(f'... Creating TOC with {self.light_model}')
        
        complete_toc_parts = []
        for chunk_key, chunk_data in tqdm(self.chunks.items()):
            result = llms.call_llm(
                prompt=prompts.build_prompt(
                    template=prompts.GET_TOPICS,
                    first_sentence=chunk_data['s_ids'][0],
                    last_sentence=chunk_data['s_ids'][-1],
                    language=self.language_string,
                    chunk=chunk_data['text']
                    ),
                model=self.light_model,
                output_format='json_schema',
                json_schema=json_schemas.GET_TOPICS_JSON,
                temperature=self.temperature,
                sleep_time=self.sleep
                )
            
            # Validate for correct JSON format
            toc_part = utils.validate_LLM_result(result, self.verbose)
            
            # Re-align topic locations
            toc_part = align_toc_part(toc_part, chunk_data)

            # Append toc_part
            complete_toc_parts.extend(toc_part)

            if self.verbose:
                print(f"... TOC PART FOR {chunk_key.upper()}:")
                utils.print_toc_topics(toc_part)

        # Post-Processing: Remove similar summaries
        print(f'... Removing similar TOC entries')
        sp = SummaryProcessor(
            summaries=complete_toc_parts,
            embedding_model=self.embedding_model,
            key='topic',
            threshold=self.sim_threshold,
            verbose=self.verbose
            )
        filtered_toc_parts = sp.get_filtered_summaries()

        # Post-Processing: Create TOC
        print(f'... Compiling TOC with {self.expert_model}')
        toc = llms.call_llm(
            prompt=prompts.build_prompt(
                template=prompts.CREATE_TOC,
                toc_parts=str(filtered_toc_parts),
                language=self.language_string
                ),
            model=self.expert_model,
            output_format='json_schema',
            json_schema=json_schemas.CREATE_TOC_JSON,
            temperature=self.temperature,
            sleep_time=self.sleep
            )
        
        # Validate LLM response
        toc = utils.validate_LLM_result(toc)
        final_toc = utils.validate_toc(toc)
        
        return final_toc

    def create_summaries_and_keywords(self) -> tuple[dict, dict]:
        """
        Create summaries and keywords in one go.
        """
        print(f'... Creating summary and keywords with {self.light_model}')
        
        summaries = []
        keywords = []
        for chunk_key, chunk_data in tqdm(self.chunks.items()):
            result = llms.call_llm(
                prompt=prompts.build_prompt(
                    template=prompts.CREATE_SUMMARY_AND_KEYWORDS,
                    text_chunk=chunk_data['text'],
                    language=self.language_string
                    ),
                model=self.light_model,
                output_format='json_schema',
                json_schema=json_schemas.CREATE_TOC_JSON,
                temperature=self.temperature,
                sleep_time=self.sleep
                )
            
            # Validate
            validated_result = utils.validate_LLM_result(result)
                        
            # Append
            summaries.append(validated_result['summary'])
            keywords.append(validated_result['keywords'])
            
            if self.verbose:
                print(f"... SUMMARY FOR {chunk_key.upper()}:")
                utils.print_summaries(validated_result['summary'])
                print("...")
                print(f"... KEYWORDS FOR {chunk_key.upper()}:")
                utils.print_summaries(validated_result['keywords'])
                
        print(f'... Harmonizing summaries with {self.expert_model}')
        final_summary = llms.call_llm(
            prompt=prompts.build_prompt(
                template=prompts.COMBINE_SUMMARIES,
                summaries='\n'.join(summaries),
                language=self.language_string
                ),
            model=self.expert_model,
            temperature=self.temperature,
            sleep_time=self.sleep 
            )

        # Create a unique set of keywords
        keywords_for_doc = self.clean_generated_keywords(keywords)
        
        # Remove similar keywords
        print(f'... Removing similar keywords')
        sp = SummaryProcessor(
            keywords=keywords_for_doc,
            embedding_model=self.embedding_model,
            key='topic',
            threshold=self.sim_threshold,
            verbose=self.verbose
            )
        filtered_keywords = sp.get_filtered_keywords()
        
        return final_summary, filtered_keywords
        
    def create_summaries(self) -> str:
        """
        Create summaries.
        """
        print(f'... Creating summary with {self.light_model}')
        
        summaries = []
        for chunk_key, chunk_data in tqdm(self.chunks.items()):
            summary = llms.call_llm(
                prompt=prompts.build_prompt(
                    template=prompts.CREATE_SUMMARY,
                    text_chunk=chunk_data['text'],
                    language=self.language_string
                    ),
                model=self.light_model,
                temperature=self.temperature,
                sleep_time=self.sleep
                )
            summaries.append(summary)
            
            if self.verbose:
                print(f"... SUMMARY FOR {chunk_key.upper()}:")
                utils.print_summaries(summary)
                
        print(f'... Harmonizing summaries with {self.expert_model}')
        final_summary = llms.call_llm(
            prompt=prompts.build_prompt(
                template=prompts.COMBINE_SUMMARIES,
                summaries='\n'.join(summaries),
                language=self.language_string
                ),
            model=self.expert_model,
            temperature=self.temperature,
            sleep_time=self.sleep
            )
        
        return final_summary
    
    def generate_keywords(self):
        """
        Generate keywords for the input text.
        """
        print(f'... Generating keywords with {self.light_model}')
        
        keywords_for_doc = []
        for chunk_key, chunk_data in tqdm(self.chunks.items()):
            keywords_for_chunk = llms.call_llm(
                prompt=prompts.build_prompt(
                    template=prompts.CREATE_KEYWORDS,
                    text_chunk=chunk_data['text'],
                    language=self.language_string
                    ),
                model=self.light_model,
                temperature=self.temperature,
                sleep_time=self.sleep
                )
            keywords_for_doc.append(keywords_for_chunk.split(','))
            
            if self.verbose:
                print(f"... KEYWORDS FOR CHUNK {chunk_key.upper()}:")
                utils.print_summaries(keywords_for_chunk)

        # Create a unique set of keywords
        keywords_for_doc = self.clean_generated_keywords(keywords_for_doc)

        # Remove similar keywords
        print(f'... Removing similar keywords')
        sp = SummaryProcessor(
            keywords=keywords_for_doc,
            embedding_model=self.embedding_model,
            key='topic',
            threshold=self.sim_threshold,
            verbose=self.verbose
            )
        filtered_keywords = sp.get_filtered_keywords()

        return filtered_keywords
    
    def clean_generated_keywords(
        self, 
        keywords_for_doc: str | list
        ) -> list[str]:
        """
        Return a unique list of keywords from LLM generated keyword list.
        """        
        clean_keywords = set()
        
        flattened = list(chain.from_iterable(keywords_for_doc))
        for keyword in flattened:
            keyword = utils.remove_punctuation(keyword.strip())
            keyword = utils.replace_underscores_with_whitespace(keyword)
            clean_keywords.add(keyword)

        return clean_keywords
    
    def reset_processing_vals(self):
        """
        Reset processing values for each file.
        """
        self.text = None
        self.chunks = None
        self.toc = None
        self.summaries = None
        self.keywords = None
        self.entities = None
        self.token_count = None
        self.language_ISO = None
        self.language_string = None
        self.sentence_offsets = None

    def process_file(self, file: Path):
        """
        Main function for processing files and creating TOCs, summaries, 
        keywords as well as running NER.
        """
        # Reset processing values
        self.reset_processing_vals()

        # Get filename and output filepath
        self.filename = file.stem
        self.output_filepath = self.output_dir.joinpath(self.filename)

        print(f'[bold][orange]... Dygesting {file.name}')

        # Load file
        self.text = self.run_with_error_handling(
            utils.load_file,
            file,
            self.verbose,
            error_message="Error during file loading"
        )
        
        # Chunk file
        self.chunks, self.token_count, self.sentence_offsets = (
            self.run_with_error_handling(
                utils.chunk_text, 
                text=self.text, 
                chunk_size=self.chunk_size, 
                error_message="Error during text chunking"
                )
            )

        if self.verbose:
            print(f"... Total tokens in file: {self.token_count}")
            print(f"... Number of chunks: {len(self.chunks)}")
            
        # Run language detection 
        if not self.language_ISO:
            self.language_ISO = self.run_with_error_handling(
                self.run_language_detection,
                error_message="Error during language detection"
            )
            
        # Transform ISO code to string ('en' → 'English')
        self.language_string = LANGUAGES.get(self.language_ISO).title()
        
        # Run Named Entity Recognition (NER)
        if self.add_ner:
            self.entities = self.run_with_error_handling(
                self.run_ner,
                error_message="Error during NER task"
            )
        
        # Create TOC (Table of Contents)
        if self.add_toc:
            self.toc = self.run_with_error_handling(
                self.create_toc,
                error_message="Error during TOC creation"
            )
         
        # Create summary and keywords in one go       
        if self.add_summaries and self.add_keywords:
            self.summaries, self.keywords = self.run_with_error_handling(
                self.create_summaries_and_keywords, 
                error_message="Error during creating summaries and keywords"
            )

        # Create only summary
        elif self.add_summaries:
            
            self.summaries = self.run_with_error_handling(
                self.create_summaries, 
                error_message="Error during summary creation"
            )

        # Create only keywords
        elif self.add_keywords:
            self.keywords = self.run_with_error_handling(
                self.generate_keywords,
                error_message="Error during keyword generation"
            )

    def run_with_error_handling(
        self, 
        func, 
        *args, 
        error_message="", 
        **kwargs
        ):
        """
        Helper function to handle exceptions uniformly.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[purple]... {error_message}: {e}")
            print(f"[purple]{traceback.format_exc()}")
            raise typer.Exit(code=1)
            
            
class SummaryProcessor:
    def __init__(
            self, 
            summaries: Optional[dict[str]] = None, 
            keywords: Optional[set[str]] = None,
            embedding_model: str = None,
            key: str = 'topic', 
            threshold: float = 0.8,
            verbose: bool = False
            ):
        """
        Initialize the SummaryProcessor and process the summaries.

        Args:
            summaries (list[dict]): List of summary dictionaries.
            embedding_model (str): The name of the embedding model to use.
            key (str): The key in the dictionary to embed (e.g., 'topic').
            threshold (float): Cosine similarity threshold to consider topics as similar.
        """
        self.embedding_model = embedding_model
        self.key = key
        self.threshold = threshold
        self.summaries = summaries
        self.keywords = keywords
        self.verbose = verbose  

        self.embedded_summaries = {}
        self.filtered_summaries = []  
        self.embedded_keywords = {}
        self.filtered_keywords = []

    def embed_summaries(self):
        """
        Embed summaries.
        """
        for summary in self.summaries:
            text = summary[self.key]
            response = llms.get_embeddings(text, model=self.embedding_model)
            self.embedded_summaries[text] = np.array(response)

    def embed_keywords(self):
        """
        Embed strings (like keywords).
        """
        for keyword in self.keywords:
            response = llms.get_embeddings(keyword, model=self.embedding_model)
            self.embedded_keywords[keyword] = np.array(response)

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
    
    def remove_similar_keywords(self):
        """
        Remove similar keywords based on cosine similarity of their embeddings.

        Returns:
            list[str]: A filtered list of keywords with similar ones removed.
        """
        similar_pairs = []

        # Generate all unique pairs of keywords and check their similarity
        for (kw1, emb1), (kw2, emb2) in combinations(self.embedded_keywords.items(), 2):
            similarity = self.cosine_similarity(emb1, emb2)
            if similarity >= self.threshold:
                similar_pairs.append({
                    'keyword_1': kw1,
                    'keyword_2': kw2,
                    'similarity_score': similarity
                })

        # Identify keywords to remove.
        keywords_to_remove = set(pair['keyword_2'] for pair in similar_pairs)

        # Filter out the similar keywords
        filtered_keywords = [k for k in self.keywords if k not in keywords_to_remove]

        # Display similar keywords if verbose is enabled
        if self.verbose:
            if similar_pairs:
                print("... Similar Keywords Identified:")
                for pair in similar_pairs:
                    print(f"... '{pair['keyword_1']}' <--> '{pair['keyword_2']}' with similarity score of {pair['similarity_score']:.4f}")
            else:
                print("... No similar keywords found above the threshold.")

        return filtered_keywords
    
    def get_filtered_summaries(self):
        """
        Get the filtered summaries.

        Returns:
            list[dict]: Filtered summaries.
        """
        self.embed_summaries()
        self.filtered_summaries = self.remove_similar_summaries()
        return self.filtered_summaries
    
    def get_filtered_keywords(self):
        """
        Get the filtered summaries.

        Returns:
            list[dict]: Filtered summaries.
        """
        self.embed_keywords()
        self.filtered_keywords = self.remove_similar_keywords()
        return self.filtered_keywords
