import typer
from typing import Optional
from enum import Enum
from pathlib import Path

app = typer.Typer()

class LLMService(Enum):
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    GROQ = 'groq'

class EmbeddingService(Enum):
    OLLAMA = 'ollama'
    OPENAI = 'openai'

class NERlanguages(str, Enum):
    AUTO = 'auto'
    AR = 'ar'
    DE = 'de'
    DA = 'da'
    EN = 'en'
    FR = 'fr'
    ES = 'es'
    NL = 'nl'
    
def resolve_input_dir(filepath: str = None, output_dir: str = None) -> Path:
    input_path = Path(filepath)
    
    if not input_path.exists():
        typer.echo(f"Error: The path '{filepath}' does not exist.", err=True)
        raise typer.Exit(code=1)

    if output_dir is None:
        if input_path.is_dir():
            resolved_output_dir = input_path
        else:
            resolved_output_dir = input_path.parent
    else:
        resolved_output_dir = Path(output_dir)
    return resolved_output_dir
    
@app.command(no_args_is_help=True)
def main(
    filepath: str = typer.Option(
        None,
        "--files",
        "-f",
        help="Path to the input folder or .txt file."
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output_dir",
        "-o",
        help="If not provided, outputs will be saved in the input folder.",
    ),
    llm_service: LLMService = typer.Option(
        None,
        "--llm_service",
        "-llm",
        help='Select the LLM service for creating digests.',
    ),
    llm_model: str = typer.Option(
        None,
        "--llm_model",
        "-m",
        help="LLM model name. Defaults to 'llama-3.1-70b-versatile' (Groq), 'gpt-4o-mini' (OpenAI) or 'llama3.1' (Ollama).",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-t",
        help='Temperature of LLM.',
    ),
    embedding_service: EmbeddingService = typer.Option(
        None,
        "--embedding_service",
        "-emb",
        help='Select the Embedding service for creating digests.',
    ),
    embedding_model: str = typer.Option(
        None,
        "--embedding_model",
        "-e",
        help="Embedding model name. Defaults to 'text-embedding-3-small' (OpenAI) or 'nomic-embed-text' (Ollama).",
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk_size",
        "-c",
        help="Maximum number of tokens per chunk."
    ),
    sim_threshold: float = typer.Option(
        0.85,
        "--sim_threshold",
        "-sim",
        help="Similarity threshold for removing duplicate summaries."
    ),
    ner: bool = typer.Option(
        False,
        "--ner",
        help="Enable Named Entity Recognition (NER). Defaults to False.",
    ),
    language: NERlanguages = typer.Option(
        NERlanguages.AUTO,
        "--lang",
        "-l",
        help='Language of file(s) for NER. Defaults to auto-detection.',
    ),
    precise: bool = typer.Option(
        False,
        "--precise",
        "-p",
        help="Enable precise mode for NER. Defaults to fast mode.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output. Defaults to False.",
    ),
    export_metadata: bool = typer.Option(
        False,
        "--export_metadata",
        "-meta",
        help="Enable exporting metadata to output file(s). Defaults to False.",
    ),
    list_models: bool = typer.Option(
        False,
        "--list_models",
        help="List all available models for a LLM service.",
    ),
):
    """
    🌞 DYGEST: Document Insights Generator 🌞
    """
    from tqdm import tqdm
    from langdetect import detect, DetectorFactory
    from dygest import llms, utils, output_utils, ner_utils
    from dygest.llms import LLMServiceBase, EmbedderBase

    class DygestProcessor:
        def __init__(
            self,
            filepath: str,
            output_dir: str = "./output",
            llm_service: LLMService = LLMService.GROQ,
            llm_model: str = None,
            embedding_service: EmbeddingService = EmbeddingService.OPENAI,
            embedding_model: str = None,
            temperature: float = 0.1,
            chunk_size: int = 1000,
            sim_threshold: float = 0.8,
            ner: bool = True,
            language: NERlanguages = NERlanguages.AUTO,
            precise: bool = False,
            verbose: bool = False,
            export_metadata: bool = False
        ):
            self.filepath = filepath
            self.output_dir = output_dir
            self.llm_service = llm_service
            self.llm_model = llm_model
            self.llm_client = None
            self.embedding_service = embedding_service
            self.embedding_model = embedding_model
            self.embedding_client = None
            self.ner_tagger = None
            self.token_count = None
            self.temperature = temperature
            self.chunk_size = chunk_size
            self.sim_threshold = sim_threshold
            self.ner = ner
            self.language = language.value 
            self.precise = precise
            self.verbose = verbose
            self.export_metadata = export_metadata

            def get_llm_service_instance(self) -> LLMServiceBase:
                if self.llm_service == LLMService.GROQ:
                    if self.llm_model is None:
                        self.llm_model = 'llama-3.1-70b-versatile'
                    return llms.GroqService()
                elif self.llm_service == LLMService.OPENAI:
                    if self.llm_model is None:
                        self.llm_model = 'gpt-4o-mini'
                    return llms.OpenAIService()
                elif self.llm_service == LLMService.OLLAMA:
                    if self.llm_model is None:
                        self.llm_model = 'llama3.1:latest'
                    return llms.OllamaService()
                else:
                    raise ValueError(
                        f"... Unknown LLM service: {self.llm_service}"
                    )

            def get_embedding_service_instance(self) -> EmbedderBase:
                if self.embedding_service == EmbeddingService.OPENAI:
                    if self.embedding_model is None:
                        self.embedding_model = 'text-embedding-3-small'
                    return llms.OpenAIEmbedder()
                elif self.embedding_service == EmbeddingService.OLLAMA:
                    if self.embedding_model is None:
                        self.embedding_model = 'nomic-embed-text'
                    return llms.OllamaEmbedder()
                else:
                    raise ValueError(
                        f"... Unknown Embedding service: {self.embedding_service}"
                    )

            # Check for or create output_dir
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)
                if self.verbose:
                    print(f"... Created output directory at {self.output_dir}")

            # Load files to process
            self.files_to_process = utils.load_filepath(self.filepath)

            # Instantiate the LLM / Embedding service(s)
            self.llm_client = get_llm_service_instance(self)
            self.embedding_client = get_embedding_service_instance(self)

        def process(self):
            for file in self.files_to_process:
                self.process_file(file)

        def process_file(self, file: Path):
            # Get filename and output filepath
            filename = file.stem
            output_filepath = self.output_dir.joinpath(filename)

            # Load and chunk the file
            text = utils.load_txt_file(file)
            chunks, self.token_count = utils.chunk_text(text, chunk_size=self.chunk_size)
            if self.verbose:
                print(f"... Processing file: {file}")
                print(f"... Total tokens in file: {self.token_count}")
                print(f"... Number of chunks: {len(chunks)}")

            # Auto language detection and NER setup
            language = self.language
            if self.ner:
                if language == 'auto':
                    DetectorFactory.seed = 0
                    language = detect(text[:500])
                    print(f"... Detected language '{language}' for {file.name}")

                # Load NER tagger if not already loaded or if language has changed
                if self.ner_tagger is None or self.language == 'auto':
                    self.ner_tagger = ner_utils.load_tagger(
                        language=language,
                        precise=self.precise
                    )
                    if self.verbose:
                        print(f"... Loaded NER tagger for language: {language}")

                # Run Named Entity Recognition (NER)
                entities = ner_utils.get_flair_entities(text, self.ner_tagger)
                all_entities = ner_utils.update_entity_positions(entities, text)

                if self.verbose:
                    print(f"... ENTITIES FOR DOC:")
                    utils.print_entities(all_entities)
            else:
                all_entities = []

            # Retrieve LLM summaries for text chunks
            all_summaries = []
            print(f'... Generating insights')
            for idx, chunk in enumerate(tqdm(chunks, desc=str(self.llm_service))):
                result = self.llm_client.prompt(
                    template='summarize',
                    text_input=chunk,
                    model=self.llm_model,
                    temperature=self.temperature
                )

                summaries = utils.validate_summaries(result)
                all_summaries.extend(summaries)

                if self.verbose:
                    print(f"... SUMMARIES FOR CHUNK {idx + 1}:")
                    utils.print_summaries(summaries)

            # Post-Processing: Remove similar summaries
            print(f'... Removing similar summaries')
            sp = utils.SummaryProcessor(
                summaries=all_summaries,
                embedding_service=self.embedding_client,
                embedding_model=self.embedding_model,
                key='topic',
                threshold=self.sim_threshold,
                verbose=self.verbose
            )
            filtered_summaries = sp.get_filtered_summaries()

            print(f'... Creating TOC')
            toc_summaries = self.llm_client.prompt(
                template='create_toc',
                text_input=filtered_summaries,
                model=self.llm_model,
                temperature=self.temperature
            )
            processed_summaries = utils.validate_summaries(toc_summaries)

            # Write Output
            html_writer = output_utils.HTMLWriter(
                filename=filename,
                output_filepath=output_filepath.with_suffix('.html'),
                text=text,
                named_entities=all_entities,
                summaries=processed_summaries,
                language=language,
                llm_service=self.llm_service,
                model=self.llm_model,
                mode='create_toc',
                token_count=self.token_count,
                export_metadata=self.export_metadata
            )
            html_writer.write_html()
            
    # List models
    if list_models:
        if llm_service is None:
            print(f'... Please provide a LLM service for listing available models.')
            return
        elif llm_service in [llm for llm in LLMService]:
            if llm_service.value == 'ollama':
                ollama_client = llms.OllamaService()
                ollama_client.list_models()
                return
            elif llm_service.value == 'groq':
                groq_client = llms.GroqService()
                groq_client.list_models()
                return
            elif llm_service.value == 'openai':
                openai_client = llms.OpenAIService()
                openai_client.list_models()
                return
            else:
                print(f'... Unknown LLM service.')
                return
            
    # Check for file
    filepath = Path(filepath)
    if filepath is None:
        print('... Please provide a file or folder (--files, -f).')
        return
                
    # Pass input_filepath dir as output if None was provided
    resolved_output_dir = resolve_input_dir(filepath, output_dir)

    processor = DygestProcessor(
        filepath=filepath,
        output_dir=resolved_output_dir,
        llm_service=LLMService(llm_service),
        llm_model=llm_model,
        embedding_service=EmbeddingService(embedding_service),
        embedding_model=embedding_model,
        temperature=temperature,
        chunk_size=chunk_size,
        sim_threshold=sim_threshold,
        ner=ner,
        language=language,
        precise=precise,
        verbose=verbose,
        export_metadata=export_metadata
    )
    processor.process()

if __name__ == '__main__':
    app()
