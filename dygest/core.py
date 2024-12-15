import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
from langdetect import detect, DetectorFactory

from dygest import llms, utils, ner_utils
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
    add_ner: bool = True
    sim_threshold: float = 0.8
    provided_language: NERlanguages = NERlanguages.AUTO
    precise: bool = False
    verbose: bool = False
    export_metadata: bool = False


@dataclass
class DygestProcessor(DygestBaseParams):
    ner_tagger: Optional[object] = field(default=None, init=False)
    text: Optional[str] = field(default=None, init=False)
    toc: Optional[list] = field(default=None, init=False)
    summaries: Optional[str] = field(default=None, init=False)
    entities: Optional[list] = field(default=None, init=False)
    files_to_process: Optional[list] = field(default=None, init=False)
    token_count: Optional[int] = field(default=None, init=False)
    language: NERlanguages = field(default=None, init=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            if self.verbose:
                print(f"... Created output directory at {self.output_dir}")

        # Load files to process
        self.files_to_process = utils.load_filepath(self.filepath)
        
    def create_toc(self, chunks):
        """
        Create a Table of Contents (TOC) for the provided file.
        """
        print(f'... Creating TOC with {self.light_model}')
        
        complete_toc_parts = []
        for idx, chunk in enumerate(tqdm(chunks)):
            result = llms.call_llm(
                template='get_topics',
                text_input=chunk,
                model=self.light_model,
                temperature=self.temperature,
                sleep_time=self.sleep
            )

            toc_part = utils.validate_summaries(result)
            complete_toc_parts.extend(toc_part)

            if self.verbose:
                print(f"... TOC PART FOR CHUNK {idx + 1}:")
                utils.print_summaries(toc_part)

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
            template='create_toc',
            text_input=filtered_toc_parts,
            model=self.expert_model,
            temperature=self.temperature,
            sleep_time=self.sleep
        )
        final_toc = utils.validate_summaries(toc)
        return final_toc
    
    def create_summaries(self, chunks):
        """
        Create summaries.
        """
        print(f'... Creating summary with {self.light_model}')
        
        tldrs = []
        for idx, chunk in enumerate(tqdm(chunks)):
            tldr = llms.call_llm(
                template='create_tldr',
                text_input=chunk,
                model=self.light_model,
                temperature=self.temperature,
                sleep_time=self.sleep
            )
            tldrs.append(tldr)
            
            if self.verbose:
                print(f"... SUMMARY FOR CHUNK {idx + 1}:")
                utils.print_summaries(tldr)
                
        print(f'... Assembling summaries with {self.expert_model}')
        combined_tldrs = llms.call_llm(
                template='combine_tldrs',
                text_input='\n'.join(tldrs),
                model=self.expert_model,
                temperature=self.temperature,
                sleep_time=self.sleep
            )
        
        return combined_tldrs
    
    def run_language_detection(self, file: Path) -> str:
        """
        Get language of text to set the correct NER model.
        """
        language = self.provided_language
        if language == 'auto':
            DetectorFactory.seed = 0
            language = detect(self.text[:500])
            print(f"... Detected language '{language}' for {file.name}")
        return language   
        
    def run_ner(self, file: Path) -> tuple[str, list]:
        """
        Run Named Entity Recognition with flair framework on the file.
        """
        language = self.run_language_detection(file)

        # Load NER tagger if not already loaded or if language has changed
        if self.ner_tagger is None or self.provided_language == 'auto':
            self.ner_tagger = ner_utils.load_tagger(
                language=language,
                precise=self.precise
            )
            if self.verbose:
                print(f"... Loaded NER tagger for language: {language}")

            # Run Named Entity Recognition (NER)
            entities = ner_utils.get_flair_entities(self.text, self.ner_tagger)
            all_entities = ner_utils.update_entity_positions(entities, self.text)

            if self.verbose:
                print(f"... ENTITIES FOR DOC:")
                utils.print_entities(all_entities)
            
            return language, all_entities
        else:
            return language, []
            
    def process(self):
        for file in self.files_to_process:
            self.process_file(file)

    def process_file(self, file: Path):
        # Get filename and output filepath
        self.filename = file.stem
        self.output_filepath = self.output_dir.joinpath(self.filename)

        # Load and chunk the file
        self.text = utils.load_txt_file(file)
        chunks, self.token_count = utils.chunk_text(
            text=self.text, 
            chunk_size=self.chunk_size
        )
 
        if self.verbose:
            print(f"... Processing file: {file}")
            print(f"... Total tokens in file: {self.token_count}")
            print(f"... Number of chunks: {len(chunks)}")

        # Run Named Entity Recognition (NER)
        if self.add_ner:
            self.language, self.entities = self.run_ner(file)
        
        # Create TOC
        if self.add_toc:
            self.toc = self.create_toc(chunks)
        
        # Post-Processing: Clean summaries
        if self.add_summaries:
            self.summaries = self.create_summaries(chunks)
            
            
class SummaryProcessor:
    def __init__(
            self, 
            summaries, 
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
        self.embedding_model = embedding_model
        self.key = key
        self.threshold = threshold
        self.summaries = summaries
        self.embedded_summaries = {}
        self.filtered_summaries = []
        self.verbose = verbose        

    def embed_summaries(self):
        """
        Embed summaries.
        """
        for summary in self.summaries:
            text = summary[self.key]
            response = llms.get_embeddings(text, model=self.embedding_model)
            self.embedded_summaries[text] = np.array(response)

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