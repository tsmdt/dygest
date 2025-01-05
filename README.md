[![PyPI version](https://badge.fury.io/py/dygest.svg)](https://badge.fury.io/py/dygest)

# 🌞 dygest: Document Insights Generator
> [!NOTE] 
**dygest** is a text analysis tool that extracts insights from documents, generating summaries, keywords, TOCs, and performing Named Entity Recognition (NER).

<p align="center">
  <img src="./assets/dygest_html.png" width="100%">
</p>

## Info
**dygest** was created to gain fast insights into longer transcripts of audio and video content by retrieving relevant topics and providing an easy to use HTML interface with short cuts from summaries to corresponding text chunks. NER processing further enhances those insights by identifying names of individuals, organisations, locations etc.

## Features 🧩
- **Text insights**  
  Generate concise insights for your text files using various LLM services by creating *summaries*, *keywords*, *table of contents* (TOC) and *named entities* (NER).
- **Unified LLM Interface**  
  dygest uses [litellm](https://github.com/BerriAI/litellm) and provides integration for various LLM service providers: `OpenAI`, `Anthropic`, `HuggingFace`, `Groq`, `Ollama` etc. Check the [complete provider list](https://docs.litellm.ai/docs/providers) for all available services. 
- **Token Friendly**  
  dygest performs token-heavy text analysis and summarization tasks. Therefore, the underlying LLM pipeline can be tailored to your needs and specific rate limits using a *mixed experts approach*.
- **Mixed Experts Approach**  
  dygest utilizes two fully customizable LLMs to handle different processing tasks. The first, referred to as the `light_model`, is designed for lighter tasks such as summarization and keyword extraction. The second, called the `expert_model`, is optimized for more complex tasks like constructing Tables of Contents (TOCs).  

  This flexibility allows for various pipeline configurations. For example, the `light_model` can run locally using `Ollama`, while the `expert_model` can leverage an external API service like `OpenAI` or `Groq`. This approach ensures efficiency and adaptability based on specific requirements.

> [!TIP]
> As the `expert_model` is dealing with a lot of input content it is recommended to use a larger LLM (`>=32B`) for this task. Smaller LLMs (`3B` to `7B`) perform well as `light_model`.

- **Named Entity Recognition (NER)**  
  Named Entity Recognition via fast and reliable `flair` framework (identifies persons, organisations, locations etc.).
  
- **User-friendly HTML Editor**  
  By default `dygest` will create a `.html` file that can be viewed in standard browsers and combines summaries, keywords, TOC and NER for your text. It features a text editor for you to make further changes.

- **Input Formats**: `.txt`, `.csv`, `.xlsx`, `.doc`, `.docx`, `.pdf`, `.json`, `.html`, `.xml`
  
- **Export Formats**: `.json`, `.csv`, `.html`

## Requirements
- 🐍 Python `>=3.10` 
- 🔑 API keys for LLM services like `OpenAI`, `Anthropic` and `Groq` *and / or* a running `Ollama` instance

> [!NOTE]
> API Keys have to be stored in your environment (e.g. `export $OPENAI_API_KEY=skj....`)

## Installation

### Install with `pip`

#### Create a Python virtual environment
```shell
python3 -m venv venv
```

#### Activate the environment
```shell
source venv/bin/activate
```

#### Install dygest
```shell
pip install dygest
```

### Install from source

#### Clone this repository
```shell
git clone https://github.com/tsmdt/dygest.git
cd dygest
```

#### Create a Python virtual environment
```shell
python3 -m venv venv
```

#### Activate the environment
```shell
source venv/bin/activate
```

#### Install dygest
```shell
pip install .
```

## Usage

### Configuration
Customize the **dygest** LLM pipeline by running the `dygest config` command:

```shell
 Usage: dygest config [OPTIONS]

 Configure LLMs, Embeddings and Named Entity Recognition.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --light_model      -l                 TEXT     LLM model name for lighter tasks (summarization, keywords) [default: None]                │
│ --expert_model     -x                 TEXT     LLM model name for heavier tasks (TOCs). [default: None]                                  │
│ --embedding_model  -e                 TEXT     Embedding model name. [default: None]                                                     │
│ --temperature      -t                 FLOAT    Temperature of LLM. [default: None]                                                       │
│ --sleep            -s                 FLOAT    Pause LLM requests to prevent rate limit errors (in seconds). [default: None]             │
│ --chunk_size       -c                 INTEGER  Maximum number of tokens per chunk. [default: None]                                       │
│ --ner                     --no-ner             Enable Named Entity Recognition (NER). Defaults to False. [default: no-ner]               │
│ --precise                 --fast               Enable precise mode for NER. Defaults to fast mode. [default: fast]                       │
│ --lang             -lang              TEXT     Language of file(s) for NER. Defaults to auto-detection. [default: None]                  │
│ --api_base         -api               TEXT     Set custom API base url for providers like Ollama and Hugginface. [default: None]         │
│ --view_config      -v                          View loaded config parameters.                                                            │
│ --help                                         Show this message and exit.                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
The configuration is saved as `dygest_config.yaml` in the project directory. The `.yaml` config looks like this:

```yaml
light_model: ollama/mistral:latest
expert_model: groq/llama-3.3-70b-versatile
embedding_model: ollama/nomic-embed-text:latest
temperature: 0.4
chunk_size: 1000
ner: true
language: auto
precise: false
api_base: null
sleep: 0
```

### Processing
Run the **dygest** LLM pipeline with the `dygest run` command:

```shell
 Usage: dygest run [OPTIONS]

 Create insights for your documents (summaries, keywords, TOCs).

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --files            -f         TEXT                 Path to the input folder or .txt file. [default: None]                       │
│ --output_dir       -o         TEXT                 If not provided, outputs will be saved in the input folder. [default: None]  │
│ --export_format    -ex        [all|json|csv|html]  Set the data format for exporting. [default: html]                           │
│ --toc              -t                              Create a Table of Contents (TOC) for the text. Defaults to False.            │
│ --summarize        -s                              Include a short summary for the text. Defaults to False.                     │
│ --keywords         -k                              Create descriptive keywords for the text. Defaults to False.                 │
│ --sim_threshold    -sim       FLOAT                Similarity threshold for removing duplicate topics. [default: 0.85]          │
│ --verbose          -v                              Enable verbose output. Defaults to False.                                    │
│ --export_metadata  -meta                           Enable exporting metadata to output file(s). Defaults to False.              │
│ --help                                             Show this message and exit.                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Export formats

#### `json`

Find an example `.json` output in the [examples folder](examples/embeddings_What_they_are_and_why_they_matter_en.json).

## Acknowledgments

**dygest** uses great python packages:

- `litellm`: https://github.com/BerriAI/litellm
- `flair`: https://github.com/flairNLP/flair
- `typer`: https://github.com/fastapi/typer
- `json_repair`: https://github.com/mangiucugna/json_repair
- `markitdown`: https://github.com/microsoft/markitdown
