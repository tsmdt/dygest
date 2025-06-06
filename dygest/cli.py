import typer
import json
from typing import Optional
from pathlib import Path
from rich import print
from dygest import utils
from dygest.output_utils import ExportFormats

app = typer.Typer(
    no_args_is_help=True,
    help='DYGEST: Document Insights Generator 🌞'
)

@app.command("config", no_args_is_help=True)
def configure(
    add_custom: Optional[str] = typer.Option(
        None,
        "--add_custom",
        "-add",
        help="Add a custom key-value pair to the config .env (format: KEY=VALUE).",
    ),
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
    view_config: bool = typer.Option(
        False,
        "--view_config",
        "-v",
        help="View loaded config parameters.",
    )
):
    """
    Configure LLMs, Embeddings and Named Entity Recognition. (Config file: .env)
    """
    from dygest.config import print_config, ENV_FILE, set_key
    from dygest.ner_utils import NERlanguages
    
    if view_config:
        print_config()
        return
    
    # Update individual config values if provided
    if light_model is not None:
        set_key(ENV_FILE, 'LIGHT_MODEL', light_model)
    
    if expert_model is not None:
        set_key(ENV_FILE, 'EXPERT_MODEL', expert_model)
    
    if embedding_model is not None:
        set_key(ENV_FILE, 'EMBEDDING_MODEL', embedding_model)
    
    if temperature is not None:
        set_key(ENV_FILE, 'TEMPERATURE', str(temperature))
    
    if sleep is not None:
        set_key(ENV_FILE, 'SLEEP', str(sleep))
    
    if chunk_size is not None:
        set_key(ENV_FILE, 'CHUNK_SIZE', str(chunk_size))
    
    if ner is not None:
        set_key(ENV_FILE, 'NER', str(ner).lower())
    
    if precise is not None:
        set_key(ENV_FILE, 'NER_PRECISE', str(precise).lower())
    
    # Add language to NER CONFIG if provided
    if language is not None:
        try:
            lang_value = NERlanguages(language).value
            set_key(ENV_FILE, 'NER_LANGUAGE', lang_value)
        except ValueError:
            typer.secho(
                f"... Warning: '{language}' is not a valid NER language. Using 'auto' instead.",
                fg=typer.colors.MAGENTA
            )
            set_key(ENV_FILE, 'NER_LANGUAGE', NERlanguages.AUTO.value)
    
    # Handle custom key-value pair if provided
    if add_custom is not None:
        if '=' not in add_custom:
            typer.secho(
                "... Error: Custom key-value pair must be in format KEY=VALUE",
                fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        
        key, value = add_custom.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        if not key:
            typer.secho(
                "... Error: Key cannot be empty",
                fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        
        # Add custom key-value pair
        set_key(ENV_FILE, key, value)
    
    # Print updated config
    print_config()

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
    html_template_default: utils.DefaultTemplates = typer.Option(
        utils.DefaultTemplates.tabs,
        "--default_template",
        "-dt",
        help="Choose a built-in HTML template ('tabs' or 'plain')."
    ),
    html_template_user: Optional[Path] = typer.Option(
        None,
        "--user_template",
        "-ut",
        help="Provide a custom folder path for an HTML template.",
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
    from dygest import core, output_utils, utils, translations
    from dygest.ner_utils import NERlanguages
    from dygest.config import missing_config_requirements, get_config_value
    
    # Check if required configuration is set
    if missing_config_requirements():
        print(f"[purple]... Please configure dygest first by running *dygest config* and set your LLMs.")
        raise typer.Exit(code=1)
    
    # Determine which HTML template folder to use
    if html_template_user is not None:
        chosen_template = html_template_user
    else:
        # Use built-in template based on the name
        try:
            chosen_template = utils.default_html_template(html_template_default.value)
        except ValueError as e:
            typer.secho(f"... Error: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    
    # Validate HTML template path if HTML export is requested
    if export_format in [ExportFormats.HTML, ExportFormats.ALL]:
        if not chosen_template.exists():
            typer.secho(
                f"... Error: HTML template folder does not exist: {chosen_template}",
                fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        html_file = next(chosen_template.glob('*.html'), None)
        if not html_file:
            typer.secho(
                f"... Error: No HTML file found in template path: {chosen_template}",
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
                    typer.secho(
                        f"... Error: The provided JSON does not follow the dygest JSON format.",
                        fg=typer.colors.RED
                    )
                    raise typer.Exit(code=1)
                    
                # Create processor with minimal required attributes
                proc = core.DygestProcessor(
                    filepath=filepath,
                    output_dir=utils.resolve_input_dir(Path(filepath), output_dir),
                    light_model=json_data['light_model'],
                    expert_model=json_data['expert_model'],
                    embedding_model=get_config_value('EMBEDDING_MODEL'),
                    temperature=get_config_value('TEMPERATURE', 0.0, float),
                    sleep=get_config_value('SLEEP', 0.0, float),
                    chunk_size=json_data['chunk_size'],
                    add_toc='toc' in json_data,
                    add_summaries='summary' in json_data,
                    add_keywords='keywords' in json_data,
                    add_ner=get_config_value('NER', False, bool),
                    sim_threshold=sim_threshold,
                    provided_language=json_data['language'],
                    precise=get_config_value('NER_PRECISE', False, bool),
                    verbose=verbose,
                    export_metadata=export_metadata,
                    export_format=export_format,
                    html_template_path=chosen_template
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
                light_model=get_config_value('LIGHT_MODEL'),
                expert_model=get_config_value('EXPERT_MODEL'),
                embedding_model=get_config_value('EMBEDDING_MODEL'),
                temperature=get_config_value('TEMPERATURE', 0.0, float),
                sleep=get_config_value('SLEEP', 0.0, float),
                chunk_size=get_config_value('CHUNK_SIZE', 0, int),
                add_toc=toc,
                add_summaries=summarize,
                add_keywords=keywords,
                add_ner=get_config_value('NER', False, bool),
                sim_threshold=sim_threshold,
                provided_language=get_config_value('NER_LANGUAGE', NERlanguages.AUTO),
                precise=get_config_value('NER_PRECISE', False, bool),
                verbose=verbose,
                export_metadata=export_metadata,
                export_format=export_format,
                html_template_path=chosen_template
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
