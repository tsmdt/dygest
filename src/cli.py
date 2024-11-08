import re
import tiktoken
import typer

from pathlib import Path
from tqdm import tqdm
from src import llms, utils, output_utils
    
    
# Typer
app = typer.Typer()


def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def chunk_text(text, max_tokens=6000):
    tokenizer = tiktoken.get_encoding("gpt2")  # Tokenizer used by OpenAI models
    chunks = []
    
    current_chunk = []
    current_chunk_length = 0
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
    
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        sentence_length = len(sentence_tokens)
        
        if current_chunk_length + sentence_length > max_tokens:
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
    
    return chunks


def get_flair_entities(text_chunk, tagger):
    from flair.data import Sentence
    
    sentence = Sentence(text_chunk)
    tagger.predict(sentence)
    
    entities = []
    for entity in sentence.get_spans('ner'):
        flair_label = entity.get_label('ner').value
        if flair_label:
            entities.append({
                'entity': entity.text,
                'category': flair_label
            })
    return entities


def remove_duplicate_entities(entities: list) -> list:
    seen_entities = set()
    all_unique_entities = []
    
    for item in entities:
        entity = item['entity']
        if entity not in seen_entities:
            all_unique_entities.append(item)
            seen_entities.add(entity)
    
    return all_unique_entities


def run(
    filepath: Path = None,
    output_dir: Path = None,
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 1000, 
    verbose: str = False
    ):
    from flair.models import SequenceTagger
    
    # Load NER tagger
    tagger = SequenceTagger.load('de-ner')
    
    # Get filename
    filename = filepath.stem

    # Chunk file
    text = load_txt_file(filepath)
    chunks = chunk_text(text, max_tokens=max_tokens)
    
    # Retrieve entities and summaries
    all_entities = []
    all_summaries = []

    for idx, chunk in enumerate(tqdm(chunks)):
        # Retrieve entities using Flair
        entities = get_flair_entities(chunk, tagger)
        all_entities.extend(entities)
        
        # Retrieve Summaries
        result = llms.prompt_groq(
            template='summarize',
            text_input=chunk,
            model=model,
            temperature=temperature)
        summaries = utils.validate_groq_summaries(result)
        all_summaries.extend(summaries)
        
        if verbose:
            print(f"\n\nENTITIES IN CHUNK {idx + 1}:\n")
            utils.print_entities(entities)
            print()
            print(f"SUMMARIES FOR CHUNK {idx + 1}:\n")
            utils.print_summaries(summaries)
            print("===============\n")
    
    # Remove duplicate entities        
    all_entities = remove_duplicate_entities(all_entities)
    
    # Remove duplicate summaries
    temp_summaries = llms.prompt_groq(
            template='clean_summaries',
            text_input=all_summaries,
            model=model,
            temperature=temperature)
    all_summaries = utils.validate_groq_summaries(temp_summaries)
    
    if verbose:
        utils.print_summaries(all_summaries)
    
    html_content = output_utils.create_html(
        filename=filename, 
        text=text, 
        summaries=all_summaries, 
        named_entities=all_entities)
    output_utils.save_html(html_content, filepath=f'{output_dir.joinpath(filename).with_suffix('.html')}')
    

@app.command(no_args_is_help=True)
def main(
    filepath: str = typer.Option(
        ...,
        "--files",
        "-f",
        help="Path to the input text file."
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
    model: str = typer.Option(
        "llama-3.1-70b-versatile",
        "--model",
        "-m",
        help='Provide the LLM model name for processing.',
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-t",
        help='Temperature of LLM.',
    ),
    language: str = typer.Option(
        "de",
        "--lang",
        "-l",
        help='Language of provided file(s) (Example: "de", "en").',
    ),
    max_tokens: int = typer.Option(
        1000,
        "--max_tokens",
        "-t", 
        help="Maximum number of tokens per chunk."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v", 
        help="Enable verbose output.",
    )):
    """
    🌞 Get insights in your content with DYGEST 🌞 
    """
    # Check for or create output_dir
    output_dir = Path(output_dir)    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Load files
    filepath = Path(filepath)
    if not filepath.exists():
        typer.echo("Please provide a valid filepath.")
        raise typer.Exit()
    elif filepath.is_dir():
        files_to_process = list(filepath.glob("*.txt"))
        if not files_to_process:
            typer.echo("No .txt files found in the directory.")
            raise typer.Exit()
    else:
        files_to_process = [filepath]

    # Run processing
    for file in files_to_process:
        run(
            filepath=file,
            output_dir=output_dir,
            model=model,
            max_tokens=max_tokens,
            verbose=verbose
        )

if __name__ == "__main__":
    app()
    