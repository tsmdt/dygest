import typer
from typing import Optional
from enum import Enum
from pathlib import Path
import json

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
    'language': NERlanguages.AUTO,
    'precise': False
}

CONFIG_FILE = Path.cwd() / ".dygest_config.json"

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG.copy()

def save_config(config: dict):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

CONFIG = load_config()

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
        0.1,
        "--temperature",
        "-t",
        help='Temperature of LLM.',
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk_size",
        "-c",
        help="Maximum number of tokens per chunk."
    ),
    ner: bool = typer.Option(
        False,
        "--ner",
        "-n",
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
    api_base: str = typer.Option(
        None,
        "--api_base",
        help="Set custom API base url for providers like Ollama and Hugginface."
    )
):
    """
    Configure LLMs, Embeddings and Named Entity Recognition.
    """
    global CONFIG
    CONFIG['llm_model'] = llm_model
    CONFIG['embedding_model'] = embedding_model
    CONFIG['temperature'] = temperature
    CONFIG['api_base'] = api_base
    CONFIG['chunk_size'] = chunk_size
    CONFIG['ner'] = ner
    CONFIG['language'] = language
    CONFIG['precise'] = precise

    # Save the updated CONFIG to a file
    save_config(CONFIG)

    print("... LLM and embedding configuration set:")
    print(f"... ... LLM model: {llm_model if llm_model else 'None'}")
    print(f"... ... Embedding model: {embedding_model if embedding_model else 'None'}")
    print(f"... ... Temperature: {temperature}")
    print(f"... ... Chunk size: {chunk_size}")
    print(f"... ... NER: {ner if ner else False}")
    print(f"... ... NER precise: {precise if precise else False}")
    print(f"... ... NER language: {language if language else 'auto'}")
    print(f"... ... API Base: {api_base if api_base else 'None'}") 

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
                    temperature=self.temperature
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
                temperature=self.temperature
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
                    temperature=self.temperature
                )
                tldrs.append(tldr)
                
                if self.verbose:
                    print(f"... SUMMARY FOR CHUNK {idx + 1}:")
                    utils.print_summaries(tldr)
                    
            combined_tldrs = llms.call_llm(
                    template='combine_tldrs',
                    text_input='\n'.join(tldrs),
                    model=self.llm_model,
                    temperature=self.temperature
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
    