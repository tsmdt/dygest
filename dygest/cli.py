import typer
import json
from pprint import pprint
from typing import Optional
from enum import Enum
from pathlib import Path

app = typer.Typer(
    no_args_is_help=True, 
    add_completion=False,
    help='🌞 DYGEST: Document Insights Generator 🌞'
)

class NERlanguages(str, Enum):
    AUTO = 'auto'
    AR = 'ar'
    DE = 'de'
    DA = 'da'
    EN = 'en'
    FR = 'fr'
    ES = 'es'
    NL = 'nl'

DEFAULT_CONFIG = {
    'llm_model': None,
    'embedding_model': None,
    'temperature': 0.1,
    'api_base': None,
    'chunk_size': 1000,
    'ner': False,
    'language': NERlanguages.AUTO.value,    # string for JSON compatibility
    'precise': False,
    'sleep': 2.5    # to prevent rate limit errors (token per minute)
}

CONFIG_FILE = Path.cwd() / ".dygest_config.json"

def load_config() -> dict:
    """
    Loads the configuration from a JSON file if it exists.
    Otherwise, returns the default configuration.
    Converts the 'language' field to the NERlanguages Enum.
    """
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Convert 'language' from string to Enum
            if 'language' in config and isinstance(config['language'], str):
                try:
                    config['language'] = NERlanguages(config['language'])
                except ValueError:
                    typer.echo(f"... Warning: '{config['language']}' is not a valid NER language. Defaulting to 'auto'.", err=True)
                    config['language'] = NERlanguages.AUTO
            else:
                config['language'] = NERlanguages.AUTO
            return config
    else:
        return DEFAULT_CONFIG.copy()

def save_config(config: dict):
    """
    Saves the configuration to a JSON file.
    Converts Enum fields to their string values for JSON compatibility.
    """
    config_to_save = config.copy()
    if isinstance(config_to_save.get('language'), Enum):
        config_to_save['language'] = config_to_save['language'].value
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_to_save, f, indent=4)

def print_config(config: dict):
    """
    Prints the LLM and embedding configuration in a structured format.
    """
    formatted_config = {
        "LLM model": config.get('llm_model') or "None",
        "Embedding model": config.get('embedding_model') or "None",
        "Temperature": config.get('temperature', 0.0),
        "Sleep": f"{config.get('sleep', 0.0)} second(s)",
        "Chunk size": config.get('chunk_size', 0),
        "NER": config.get('ner', False),
        "NER precise": config.get('precise', False),
        "NER language": config.get(
            'language'
            ).value if isinstance(config.get('language'), Enum) else "auto",
        "API Base": config.get('api_base') or "None",
    }
    
    print("... LLM and embedding configuration:")
    for key, value in formatted_config.items():
        print(f"... ... {key}: {value}")

CONFIG = load_config()

def resolve_input_dir(filepath: str = None, output_dir: str = None) -> Path:
    input_path = Path(filepath)

    if not input_path.exists():
        typer.echo(f"... Error: The path '{filepath}' does not exist.", err=True)
        raise typer.Exit(code=1)

    if output_dir is None:
        if input_path.is_dir():
            resolved_output_dir = input_path
        else:
            resolved_output_dir = input_path.parent
    else:
        resolved_output_dir = Path(output_dir)
    return resolved_output_dir

@app.command("config", no_args_is_help=True)
def configure(
    llm_model: str = typer.Option(
        None,
        "--llm_model",
        "-m",
        help="LLM model name.",
    ),
    embedding_model: str = typer.Option(
        None,
        "--embedding_model",
        "-e",
        help="Embedding model name.",
    ),
    temperature: float = typer.Option(
        None,
        "--temperature",
        "-t",
        help='Temperature of LLM.',
    ),
    sleep: float = typer.Option(
        None,
        "--sleep",
        help='Increase this value if you experience rate limit errors (token per minute).',
    ),
    chunk_size: int = typer.Option(
        None,
        "--chunk_size",
        "-c",
        help="Maximum number of tokens per chunk."
    ),
    ner: bool = typer.Option(
        None,
        "--ner",
        "-n",
        help="Enable Named Entity Recognition (NER). Defaults to False.",
    ),
    language: NERlanguages = typer.Option(
        None,
        "--lang",
        "-l",
        help='Language of file(s) for NER. Defaults to auto-detection.',
    ),
    precise: bool = typer.Option(
        None,
        "--precise",
        "-p",
        help="Enable precise mode for NER. Defaults to fast mode.",
    ),
    api_base: str = typer.Option(
        None,
        "--api_base",
        help="Set custom API base url for providers like Ollama and Hugginface."
    ),
    show_config: bool = typer.Option(
        False,
        "--show_config",
        "-show",
        help="Show loaded config parameters.",
    )
):
    """
    Configure LLMs, Embeddings and Named Entity Recognition.
    """
    global CONFIG
    CONFIG['llm_model'] = llm_model if llm_model is not None else CONFIG.get('llm_model')
    CONFIG['embedding_model'] = embedding_model if embedding_model is not None else CONFIG.get('embedding_model')
    CONFIG['temperature'] = temperature if temperature is not None else CONFIG.get('temperature')
    CONFIG['sleep'] = sleep if sleep is not None else CONFIG.get('sleep')
    CONFIG['api_base'] = api_base if api_base is not None else CONFIG.get('api_base')
    CONFIG['chunk_size'] = chunk_size if chunk_size is not None else CONFIG.get('chunk_size')
    CONFIG['ner'] = ner if ner is not None else CONFIG.get('ner')
    CONFIG['language'] = language if language is not None else CONFIG.get('language')
    CONFIG['precise'] = precise if precise is not None else CONFIG.get('precise')

    if show_config:
        print_config(CONFIG)
        return

    # Save the updated CONFIG to a file
    save_config(CONFIG)

    # Print the configuration using the print_config function
    print_config(CONFIG)

@app.command("run", no_args_is_help=True)
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
    toc: bool = typer.Option(
        False,
        "--toc",
        "-t",
        help="Create a Table of Contents (TOC) for the text. Defaults to False.",
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize",
        "-s",
        help="Include a short summary for the whole text. Defaults to False.",
    ),
    sim_threshold: float = typer.Option(
        0.85,
        "--sim_threshold",
        "-sim",
        help="Similarity threshold for removing duplicate topics."
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
    )
):
    """
    Create insights for your documents (summaries, keywords, TOCs).
    """
    global CONFIG
    CONFIG = load_config()

    from tqdm import tqdm
    from langdetect import detect, DetectorFactory
    from dygest import llms, utils, output_utils, ner_utils

    class DygestProcessor:
        def __init__(
            self,
            filepath: str,
            output_dir: str = "./output",
            llm_model: str = None,
            embedding_model: str = None,
            temperature: float = 0.1,
            sleep: float = 0,
            chunk_size: int = 1000,
            toc: bool = False,
            summarize: bool = False,
            sim_threshold: float = 0.8,
            ner: bool = True,
            language: NERlanguages = NERlanguages.AUTO,
            precise: bool = False,
            verbose: bool = False,
            export_metadata: bool = False
        ):
            self.filepath = filepath
            self.output_dir = Path(output_dir)
            self.llm_model = llm_model
            self.embedding_model = embedding_model
            self.ner_tagger = None
            self.token_count = None
            self.temperature = temperature
            self.sleep = sleep
            self.chunk_size = chunk_size
            self.toc = toc
            self.summarize = summarize
            self.sim_threshold = sim_threshold
            self.ner = ner
            self.language = language
            self.precise = precise
            self.verbose = verbose
            self.export_metadata = export_metadata

            # Check for or create output_dir
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)
                if self.verbose:
                    print(f"... Created output directory at {self.output_dir}")

            # Load files to process
            self.files_to_process = utils.load_filepath(self.filepath)
            
        def create_toc(self, chunks):
            """
            Create a Table of Contents (TOC) for the provided file
            """
            print(f'... Creating TOC with {self.llm_model}')
            
            complete_toc_parts = []
            for idx, chunk in enumerate(tqdm(chunks)):
                result = llms.call_llm(
                    template='summarize',
                    text_input=chunk,
                    model=self.llm_model,
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
            sp = utils.SummaryProcessor(
                summaries=complete_toc_parts,
                embedding_model=self.embedding_model,
                key='topic',
                threshold=self.sim_threshold,
                verbose=self.verbose
            )
            filtered_toc_parts = sp.get_filtered_summaries()

            # Post-Processing: Create TOC
            toc = llms.call_llm(
                template='create_toc',
                text_input=filtered_toc_parts,
                model=self.llm_model,
                temperature=self.temperature,
                sleep_time=self.sleep
            )
            final_toc = utils.validate_summaries(toc)
            return final_toc
        
        def create_summaries(self, chunks):
            """
            Create summaries.
            """
            print(f'... Creating summary with {self.llm_model}')
            
            tldrs = []
            for idx, chunk in enumerate(tqdm(chunks)):
                tldr = llms.call_llm(
                    template='create_tldr',
                    text_input=chunk,
                    model=self.llm_model,
                    temperature=self.temperature,
                    sleep_time=self.sleep
                )
                tldrs.append(tldr)
                
                if self.verbose:
                    print(f"... SUMMARY FOR CHUNK {idx + 1}:")
                    utils.print_summaries(tldr)
                    
            combined_tldrs = llms.call_llm(
                    template='combine_tldrs',
                    text_input='\n'.join(tldrs),
                    model=self.llm_model,
                    temperature=self.temperature,
                    sleep_time=self.sleep
                )
            
            return combined_tldrs
        
        def run_language_detection(self, text: str, file: Path) -> str:
            """
            Get language of text to set the correct NER model.
            """
            language = self.language
            if language == 'auto':
                DetectorFactory.seed = 0
                language = detect(text[:500])
                print(f"... Detected language '{language}' for {file.name}")
            return language   
            
        def run_ner(self, text: str, file: Path) -> tuple[str, list]:
            """
            Run Named Entity Recognition with flair framework on the file.
            """
            language = self.run_language_detection(text, file)

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
                
                return language, all_entities
            else:
                return language, []
                
        def process(self):
            for file in self.files_to_process:
                self.process_file(file)

        def process_file(self, file: Path):
            # Get filename and output filepath
            filename = file.stem
            output_filepath = self.output_dir.joinpath(filename)

            # Load and chunk the file
            text = utils.load_txt_file(file)
            chunks, self.token_count = utils.chunk_text(
                text, 
                chunk_size=self.chunk_size
            )
            
            if self.verbose:
                print(f"... Processing file: {file}")
                print(f"... Total tokens in file: {self.token_count}")
                print(f"... Number of chunks: {len(chunks)}")

            # Run Named Entity Recognition (NER)
            if self.ner:
                language, entities = self.run_ner(text, file)
           
            # Create TOC
            if self.toc:
                final_toc = self.create_toc(chunks)
            
            # Post-Processing: Clean TLDR
            if self.summarize:
                tldrs = self.create_summaries(chunks)

            # Write Output
            print(f'... Writing HTML')
            html_writer = output_utils.HTMLWriter(
                filename=filename,
                output_filepath=output_filepath.with_suffix('.html'),
                text=text,
                named_entities=entities if self.ner else None,
                toc=final_toc if self.toc else None,
                tldrs=tldrs if self.summarize else None,
                language=language if self.ner else None,
                model=self.llm_model,
                mode='create_toc',
                token_count=self.token_count,
                export_metadata=self.export_metadata
            )
            html_writer.write_html()
            
    # Check for file
    if filepath is None:
        print('... Please provide a file or folder (--files, -f).')
        return

    filepath = Path(filepath)
    resolved_output_dir = resolve_input_dir(filepath, output_dir)
    
    processor = DygestProcessor(
        filepath=filepath,
        output_dir=resolved_output_dir,
        llm_model=CONFIG['llm_model'],
        embedding_model=CONFIG['embedding_model'],
        temperature=CONFIG['temperature'],
        sleep=CONFIG['sleep'],
        chunk_size=CONFIG['chunk_size'],
        toc=toc,
        summarize=summarize,
        sim_threshold=sim_threshold,
        ner=CONFIG['ner'],
        language=CONFIG['language'],
        precise=CONFIG['precise'],
        verbose=verbose,
        export_metadata=export_metadata
    )
    processor.process()

if __name__ == '__main__':
    app()
    