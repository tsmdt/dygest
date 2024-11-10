import typer
from enum import Enum
from pathlib import Path
from tqdm import tqdm
from flair.nn import Classifier
from langdetect import detect, DetectorFactory

from dygest import llms, utils, output_utils, ner_utils
from dygest.llms import LLMServiceBase
    
    
app = typer.Typer()


class MODE(Enum):
    summarize = 'sum'
    create_toc = 'toc'


class LLMService(Enum):
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    GROQ = 'groq'


class NERlanguages(str, Enum):
    AUTO = 'auto'
    AR = 'ar'
    DE = 'de'
    DA = 'da'
    EN = 'en'
    FR = 'fr'
    ES = 'es'
    NL = 'nl'


def get_llm_service_instance(service: LLMService) -> LLMServiceBase:
    if service == LLMService.GROQ:
        return llms.GroqService()
    elif service == LLMService.OPENAI:
        return llms.OpenAIService()
    elif service == LLMService.OLLAMA:
        return llms.OllamaService()
    else:
        raise ValueError(f"Unknown LLM service: {service}")

def run(
    filepath: Path = None,
    output_dir: Path = None,
    llm_service: LLMServiceBase = None,
    model: str = None,
    temperature: float = 0.1,
    mode: str = None,
    ner_tagger: Classifier = None,
    precise: bool = False,
    max_tokens: int = 1000, 
    language: str = 'en',
    verbose: str = False
    ):
    """
    Processes a text file by chunking it, performing named entity recognition (NER),
    summarizing each chunk using a specified language model, and generating a final HTML 
    report containing all detected entities and summaries.

    Args:
        filepath (Path): Path to the input text file.
        output_dir (Path): Directory to save the generated HTML output.
        llm_service (LLMServiceBase): Service for interacting with a language model 
            to generate summaries.
        model (str): The language model to use for summarization.
        temperature (float): Temperature setting for the language model's response, 
            influencing response creativity (default is 0.1).
        ner_tagger (Classifier): NER tagger for extracting named entities; if None, 
            a tagger will be selected based on the detected or provided language.
        precise (bool): If True, enables precise mode for NER tagging, which may increase accuracy.
        max_tokens (int): Maximum number of tokens per chunk; used for dividing the text 
            into manageable sections.
        language (str): Language code for the input text (e.g., 'en' for English). 
            If set to 'auto', language is detected automatically.
        verbose (str): If True, prints detailed output for each chunk, including detected 
            entities and summaries.

    Returns:
        None: Generates an HTML file in the specified output directory containing:
            - Extracted named entities (with duplicates removed)
            - Summarized content for each chunk (with duplicates removed)
            - Original text for reference
    """
    # Get filename
    filename = filepath.stem

    # Create output_filepath
    output_filepath = Path(f'{output_dir.joinpath(filename)}')

    # Chunk file
    text = utils.load_txt_file(filepath)
    chunks, token_count = utils.chunk_text(text, max_tokens=max_tokens)

    # Auto language detection
    if language == 'auto':
        DetectorFactory.seed = 0
        language = detect(text[:500])
        print(f"... Detected language '{language}' for {filename}")
        ner_tagger = ner_utils.load_tagger(
            language=language, 
            precise=precise
            )
    
    # Run Named Entity Recognition (NER)
    entities = ner_utils.get_flair_entities(text, ner_tagger)
    all_entities = ner_utils.update_entity_positions(entities, text)

    if verbose:
        print(f"\n\nENTITIES FOR DOC:\n")
        utils.print_entities(all_entities)

    # Retrieve LLM summaries for text chunks
    all_summaries = []
    for idx, chunk in enumerate(tqdm(chunks)):
        result = llm_service.prompt(
            template='summarize',
            text_input=chunk,
            model=model,
            temperature=temperature)
        
        summaries = utils.validate_summaries(result)
        all_summaries.extend(summaries)
        
        if verbose:
            print(f"SUMMARIES FOR CHUNK {idx + 1}:\n")
            utils.print_summaries(summaries)
            print("===============\n")
    
    # Post-Processing: Clean summaries or create TOC
    temp_summaries = llm_service.prompt(
        template=mode, # Sets the post-processing mode
        text_input=all_summaries,
        model=model,
        temperature=temperature)
    all_summaries = utils.validate_summaries(temp_summaries)

    # print(temp_summaries)
    
    if verbose:
        print(f"COMPLETE SUMMARIES FOR DOC:\n")
        utils.print_summaries(all_summaries)

    # Write Output
    html_writer = output_utils.HTMLWriter(
        filename=filename,
        output_filepath=output_filepath.with_suffix('.html'),
        text=text,
        named_entities=all_entities,
        summaries=all_summaries,
        language=language,
        llm_service=llm_service,
        model=model,
        mode=mode,
        token_count=token_count
    )
    html_writer.write_html()

@app.command(no_args_is_help=True)
def main(
    filepath: str = typer.Option(
        ...,
        "--files",
        "-f",
        help="Path to the input folder or .txt file."
    ),
    output_dir: str = typer.Option(
        Path("./output"),
        "--output_dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help="Folder where digests should be saved.",
    ),
    service: LLMService = typer.Option(
        LLMService.GROQ,
        "--service",
        "-s",
        help='Select the LLM service.',
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="""
        Provide the LLM model name for creating digests. 
        Defaults to "llama-3.1-70b-versatile" (Groq), "gpt-4o-mini (OpenAI) or "llama3.1" (Ollama).""",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-t",
        help='Temperature of LLM.',
    ),
    dygest_mode: MODE = typer.Option(
        MODE.create_toc.value,
        "--dygest",
        '-d',
        help='Create summaries or a table of contents (TOC).',
    ),
    max_tokens: int = typer.Option(
        1000,
        "--max_tokens", 
        help="Maximum number of tokens per chunk."
    ),
    ner: bool = typer.Option(
        True,
        "--ner",
        help="Enable Named Entity Recognition (NER). Defaults to True.",
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
        help="Enable verbose output.",
    )):
    """
    🌞 DYGEST: Document Insights Generator 🌞 

    -----------------------------------------
    
    DYGEST is a designed to extract meaningful insights from your text documents.
    It can generate summaries, create tables of contents (TOC), and perform Named Entity Recognition (NER)
    to identify and categorize key information within your documents.
    """
    # Check for or create output_dir
    output_dir = Path(output_dir)    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Load files
    files_to_process = utils.load_filepath(filepath)

    # Instantiate the LLM service
    llm_service_instance = get_llm_service_instance(service)

    # Set default LLM model if none is provided
    if model is None:
        if service == LLMService.GROQ:
            model = 'llama-3.1-70b-versatile'
        elif service == LLMService.OPENAI:
            model = 'gpt-4o-mini'
        elif service == LLMService.OLLAMA:
            model = 'llama3.1:latest'
        else:
            raise ValueError(f"Unknown LLM service: {service}")

    # Load NER tagger
    if ner and not language.value == 'auto':
        tagger = ner_utils.load_tagger(language=language.value, precise=precise)
    else:
        tagger = None

    # Map mode to prompt templates
    mode = {"sum": "clean_summaries", "toc": "create_toc"}[dygest_mode.value]

    # Run processing
    for file in files_to_process:
        run(
            filepath=file,
            output_dir=output_dir,
            llm_service=llm_service_instance,
            model=model,
            temperature=temperature,
            mode=mode,
            ner_tagger=tagger,
            precise=precise,
            max_tokens=max_tokens,
            language=language.value,
            verbose=verbose
        )

if __name__ == "__main__":
    app()
    