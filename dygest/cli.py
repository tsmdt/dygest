import typer
import json
from typing import Optional
from pathlib import Path
from rich import print

from dygest.output_utils import ExportFormats
from dygest.config import CONFIG, DEFAULT_CONFIG

app = typer.Typer(
    no_args_is_help=True,
    help='🌞 DYGEST: Document Insights Generator 🌞'
)

@app.command("config", no_args_is_help=True)
def configure(
    light_model: str = typer.Option(
        None,
        "--light_model",
        "-l",
        help="LLM model name for lighter tasks (summarization, keywords)",
    ),
    expert_model: str = typer.Option(
        None,
        "--expert_model",
        "-x",
        help="LLM model name for heavier tasks (TOCs).",
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
        "-s",
        help='Pause LLM requests to prevent rate limit errors (in seconds).',
    ),
    chunk_size: int = typer.Option(
        None,
        "--chunk_size",
        "-c",
        help="Maximum number of tokens per chunk."
    ),
    ner: Optional[bool] = typer.Option(
        None,
        "--ner/--no-ner",
        help="Enable Named Entity Recognition (NER). Defaults to False.",
    ),
    precise: Optional[bool] = typer.Option(
        None,
        "--precise/--fast",
        help="Enable precise mode for NER. Defaults to fast mode.",
    ),
    language: str = typer.Option(
        None,
        "--lang",
        "-lang",
        help='Language of file(s) for NER. Defaults to auto-detection.',
    ),
    api_base: str = typer.Option(
        None,
        "--api_base",
        '-api',
        help="Set custom API base url for providers like Ollama and Hugginface."
    ),
    view_config: bool = typer.Option(
        False,
        "--view_config",
        "-v",
        help="View loaded config parameters.",
    )
):
    """
    Configure LLMs, Embeddings and Named Entity Recognition.
    """
    from dygest.config import load_config, save_config, print_config
    from dygest.ner_utils import NERlanguages
    
    global CONFIG
    if CONFIG is None:
        CONFIG = load_config()
    
    if view_config:
        print_config(CONFIG)
        return
        
    CONFIG['light_model'] = light_model if light_model is not None else CONFIG.get('light_model')
    CONFIG['expert_model'] = expert_model if expert_model is not None else CONFIG.get('expert_model')
    CONFIG['embedding_model'] = embedding_model if embedding_model is not None else CONFIG.get('embedding_model')
    CONFIG['temperature'] = temperature if temperature is not None else CONFIG.get('temperature')
    CONFIG['sleep'] = sleep if sleep is not None else CONFIG.get('sleep')
    CONFIG['api_base'] = api_base if api_base is not None else CONFIG.get('api_base')
    CONFIG['chunk_size'] = chunk_size if chunk_size is not None else CONFIG.get('chunk_size')
    CONFIG['ner'] = ner if ner is not None else CONFIG.get('ner')
    CONFIG['precise'] = precise if precise is not None else CONFIG.get('precise')

    # Convert language if provided
    if language is not None:
        try:
            CONFIG['language'] = NERlanguages(language)
        except ValueError:
            typer.echo(f"... Warning: '{language}' is not a valid NER language. Using 'auto' instead.", err=True)
            CONFIG['language'] = NERlanguages.AUTO
    
    # Save and print config
    save_config(CONFIG)
    print_config(CONFIG)

@app.command("run", no_args_is_help=True)
def main(
    filepath: str = typer.Option(
        None,
        "--files",
        "-f",
        help="Path to the input folder or file."
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output_dir",
        "-o",
        help="If not provided, outputs will be saved in the input folder.",
    ),
    export_format: Optional[ExportFormats] = typer.Option(
        ExportFormats.HTML,
        "--export_format",
        "-ex",
        help="Set the data format for exporting.",
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
        help="Include a short summary for the text. Defaults to False.",
    ),
    keywords: bool = typer.Option(
        False,
        "--keywords",
        "-k",
        help="Create descriptive keywords for the text. Defaults to False.",
    ),
    sim_threshold: float = typer.Option(
        0.85,
        "--sim_threshold",
        "-sim",
        help="Similarity threshold for removing duplicate topics."
    ),
    html_template: Path = typer.Option(
        'templates/plain',
        "--html_template",
        "-ht",
        help="Specify a folder with an HTML template, CSS and (optional) JavaScript.",
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    skip_html: bool = typer.Option(
        False,
        "--skip_html",
        "-skip",
        help="Skip files if HTML already exists in the same folder. Defaults to False.",
    ),
    export_metadata: bool = typer.Option(
        False,
        "--export_metadata",
        "-meta",
        help="Enable exporting metadata to output file(s). Defaults to False.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output. Defaults to False.",
    )
):
    """
    Create insights for your documents (summaries, keywords, TOCs).
    """
    from dygest.config import load_config
    from dygest import core, output_utils, utils, translations
    global CONFIG
    
    # Load config and make sure it is correctly set
    CONFIG = load_config()
    if CONFIG == DEFAULT_CONFIG:
        print(f"[purple]... Please configure dygest first by running *dygest \
config* and set your LLMs.")
        raise typer.Exit(code=1)
    
    # Validate HTML template path if HTML export is requested
    if export_format in [ExportFormats.HTML, ExportFormats.ALL]:
        if not html_template.exists():
            typer.secho(
                f"... Error: HTML template folder does not exist: {html_template}",
                fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        html_file = next(html_template.glob('*.html'), None)
        if not html_file:
            typer.secho(
                f"... Error: No HTML file found in template path: {html_template}",
                fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
    
    # Create a list of all files to process
    files_to_process = utils.load_filepath(filepath, skip_html=skip_html)
        
    # Process files
    for file in files_to_process:
        # Handle dygest JSON input files (and do not run LLM processing)
        if file.suffix.lower() == '.json':
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                # Validate JSON structure
                if not utils.validate_json_input(json_data):
                    raise ValueError("Invalid JSON structure")
                    
                # Create processor with minimal required attributes
                proc = core.DygestProcessor(
                    filepath=filepath,
                    output_dir=utils.resolve_input_dir(Path(filepath), output_dir),
                    light_model=json_data['light_model'],
                    expert_model=json_data['expert_model'],
                    embedding_model=CONFIG['embedding_model'],
                    temperature=CONFIG['temperature'],
                    sleep=CONFIG['sleep'],
                    chunk_size=json_data['chunk_size'],
                    add_toc='toc' in json_data,
                    add_summaries='summary' in json_data,
                    add_keywords='keywords' in json_data,
                    add_ner=CONFIG['ner'],
                    sim_threshold=sim_threshold,
                    provided_language=json_data['language'],
                    precise=CONFIG['precise'],
                    verbose=verbose,
                    export_metadata=export_metadata,
                    export_format=export_format,
                    html_template_path=html_template
                )
                
                # Set the data from JSON
                proc.filename = json_data['filename']
                proc.output_filepath = Path(json_data['output_filepath'])
                proc.text = '\n'.join(chunk['text'] for chunk in json_data['chunks'].values())
                proc.chunks = json_data['chunks']
                proc.token_count = json_data['token_count']
                proc.language_ISO = json_data['language']
                proc.language_string = translations.LANGUAGES.get(json_data['language']).title()
                proc.sentence_offsets = json_data['sentence_offsets']
                
                if 'summary' in json_data:
                    proc.summaries = json_data['summary']
                if 'keywords' in json_data:
                    proc.keywords = json_data['keywords']
                if 'toc' in json_data:
                    proc.toc = json_data['toc']
                
            except json.JSONDecodeError as e:
                print(f"[purple]... Error: Invalid JSON file: {e}")
                continue
            except Exception as e:
                print(f"[purple]... Error processing JSON file: {e}")
                continue
        else:
            # Regular file processing
            proc = core.DygestProcessor(
                filepath=filepath,
                output_dir=utils.resolve_input_dir(Path(filepath), output_dir),
                light_model=CONFIG['light_model'],
                expert_model=CONFIG['expert_model'],
                embedding_model=CONFIG['embedding_model'],
                temperature=CONFIG['temperature'],
                sleep=CONFIG['sleep'],
                chunk_size=CONFIG['chunk_size'],
                add_toc=toc,
                add_summaries=summarize,
                add_keywords=keywords,
                add_ner=CONFIG['ner'],
                sim_threshold=sim_threshold,
                provided_language=CONFIG['language'],
                precise=CONFIG['precise'],
                verbose=verbose,
                export_metadata=export_metadata,
                export_format=export_format,
                html_template_path=html_template
            )
            
            # Process file
            proc.process_file(file)

        # Write output
        try:
            if file.suffix.lower() != '.json':
                formats_to_export = (
                    [ExportFormats.CSV, ExportFormats.JSON, ExportFormats.HTML]
                    if proc.export_format == ExportFormats.ALL
                    else [proc.export_format, ExportFormats.JSON]
                )
            else:
                formats_to_export = (
                    [ExportFormats.CSV, ExportFormats.HTML]
                    if proc.export_format == ExportFormats.ALL
                    else [proc.export_format]
                )
            
            for format in formats_to_export:
                proc.export_format = format
                writer = output_utils.get_writer(proc)
                writer.write()
            
            print('[blue][bold]... DONE')

        except ValueError as ve:
            print(f'... {ve}')
        except Exception as e:
            print(f'... An unexpected error occurred: {e}')

if __name__ == '__main__':
    app()
    