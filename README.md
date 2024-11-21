# 🌞 DYGEST: Document Insights Generator
**dygest** is a command-line tool designed to extract meaningful insights from `.txt` files. It generates summaries, creates tables of contents (TOC), and performs Named Entity Recognition (NER) to identify and categorize key information within your documents. It creates a `.html` for further reviewing and editing.

## Features 🧩 
- **Text insights:** Generate concise insights for your text files using various LLM services by creating summaries, table of contents (TOC) and Named Entity Recognition (NER).
- **LLM APIs:** Integration for `OpenAI`, `Groq` and `Ollama` available.
- **Embedding APIs:** Integretation for `OpenAI` and `Ollama` available. Used for creating better TOCs.
- **NER:** Named Entity Recognition via fast and reliable `flair` framework (identifies persons, organisations, locations etc.).
- **HTML Editor**: By default **dygest** will create a `.html` file that can be viewed in standard browsers and combines summaries, TOC and NER for your text. It features a text editor for you to make further changes.

## How it works 🌞
**dygest** was created to gain fast insights into longer transcripts of audio and video content by retrieving relevant topics and providing an easy to use HTML interface with short cuts from summaries to corresponding text chunks. NER processing further enhances those insights by identifying names of individuals, organisations, locations etc.

### Workflow
1. **Chunking**: `.txt` input files are firstly chunked using the `--chunk_size` option (default: 1000 tokens).
2. **Summary Creation**: For each chunk 1-2 summaries are generated using a LLM service of your choice (`OpenAI`, `Groq`, `Ollama`); all summaries focus on the most relevant topics discussed in the corresponding chunk. (**Hint**: a larger LLM (`70b` compared to `8b`) generally means better results.)
3. **Duplicate Removal**: After retrieving all summaries a similarity comparison is run to detect identical or very similar summaries that will then get removed. This detection can be controlled via the `--sim_threshold` option. A higher float number means a higher threshold for detecting a duplicate. A lower float number means that similar summaries are much more loosly identified.
4. **TOC Creation**: After the duplicate removal a table of contents is created using the LLM service. (**Hint**: a larger LLM (`70b` compared to `8b`) generally means better results.)
5. **Document-wise Summary**: A short summary for the whole document can be generated too.
6. **HTML**: By default **dygest** will create a `.html` file that combines TOC, NER result and the provided text with a focus on usability and quick access to information. The provided `.txt` can be edited in the browser to make further changes (e.g. correction of a transcript).

<p align="center">
  <img src="./assets/dygest_html.png" width="100%">
</p>

## Requirements
- `Python >= 3.10` 
- API Keys for `OpenAI` and/or `Groq` *or* a running `Ollama` instance
- API Keys have to be stored in your environment (e.g. `export $OPENAI_API_KEY=skj....`)

## Installation

### Clone this repository
```shell
git clone https://github.com/tsmdt/dygest.git
```

### Change to folder
```shell
cd dygest
```

### Create a virtual environment
```shell
python3 -m venv venv
```

### Activate the environment
```shell
source venv/bin/activate
```

### Install dygest
```shell
pip install .
```

## Usage
```shell
>>> dygest

 Usage: dygest [OPTIONS]

 🌞 DYGEST: Document Insights Generator 🌞

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --files               -f        TEXT                         Path to the input folder or .txt file. [default: None]                                        │
│ --output_dir          -o        TEXT                         If not provided, outputs will be saved in the input folder. [default: None]                   │
│ --llm_service         -llm      [ollama|openai|groq]         Select the LLM service for creating digests. [default: None]                                  │
│ --llm_model           -m        TEXT                         LLM model name. Defaults to 'llama-3.1-70b-versatile' (Groq), 'gpt-4o-mini' (OpenAI) or       │
│                                                              'llama3.1' (Ollama).                                                                          │
│                                                              [default: None]                                                                               │
│ --temperature         -t        FLOAT                        Temperature of LLM. [default: 0.1]                                                            │
│ --embedding_service   -emb      [ollama|openai]              Select the Embedding service for creating digests. [default: None]                            │
│ --embedding_model     -e        TEXT                         Embedding model name. Defaults to 'text-embedding-3-small' (OpenAI) or 'nomic-embed-text'     │
│                                                              (Ollama).                                                                                     │
│                                                              [default: None]                                                                               │
│ --chunk_size          -c        INTEGER                      Maximum number of tokens per chunk. [default: 1000]                                           │
│ --summarize           -s                                     Include a short summary for the whole text. Defaults to False.                                │
│ --sim_threshold       -t        FLOAT                        Similarity threshold for removing duplicate topics. [default: 0.85]                           │
│ --ner                 -n                                     Enable Named Entity Recognition (NER). Defaults to False.                                     │
│ --lang                -l        [auto|ar|de|da|en|fr|es|nl]  Language of file(s) for NER. Defaults to auto-detection. [default: auto]                      │
│ --precise             -p                                     Enable precise mode for NER. Defaults to fast mode.                                           │
│ --verbose             -v                                     Enable verbose output. Defaults to False.                                                     │
│ --export_metadata                                            Enable exporting metadata to output file(s). Defaults to False.                               │
│ --list_models                                                List all available models for a LLM service.                                                  │
│ --install-completion                                         Install completion for the current shell.                                                     │
│ --show-completion                                            Show completion for the current shell, to copy it or customize the installation.              │
│ --help                                                       Show this message and exit.                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Examples

### Generate dygest with default settings:
```shell
dygest --files ./documents/my_txt.txt -llm groq -emb openai 
```
Creates dygest using `Groq API` with default model `llama-3.1-70b-versatile` and `OpenAI Embeddings` model `text-embedding-3-small`.

### Generate dygest with NER using local LLMs:
```shell
dygest --files ./documents/my_txt.txt -llm ollama -m llama3.1:8b-instruct-q8_0 -emb ollama -e chroma/all-minilm-l6-v2-f32:latest -n -v --export_metadata
```
Creates a `.html` using `Ollama` with LLM `llama3.1:8b-instruct-q8_0` and embeddings model `chroma/all-minilm-l6-v2-f32:latest` while enabling `NER`, `verbose` output and exporting processing `metadata` to the `.html`. Make sure that you have the models you want to use pulled with `Ollama` first.

### Generate dygest OpenAI with NER and genereous duplicate removal:
```shell
dygest --files ./documents/my_txt.txt -llm openai -emb openai -n -p --sim_threshold 0.6
```
Creates a `.html` using `OpenAI` with default LLM `gpt-4o-mini` and default embedding model `text-embedding-3-small` while enabling `NER` in `precise` mode. The similarity threshold set with `sim_threshold` is generous and will remove many summaries that are somewhat comparable to other ones found in the TOC.

### List available models for a LLM service:
```shell
dygest -llm openai --list_models
```
Lists all available `OpenaAI` models.
