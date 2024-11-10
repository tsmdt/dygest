# 🌞 DYGEST: Document Insights Generator

**dygest** is a command-line tool designed to extract meaningful insights from your text documents. It can generate summaries, create tables of contents (TOC), and perform Named Entity Recognition (NER) to identify and categorize key information within your documents.

## Features 🧩 
- **Text insights:** Generate concise insights for your text files using various LLM services by creating summaries, table of contents (TOC) and Named Entity Recognition (NER).
- **LLM APIs:** Integration for `OpenAI`, `Groq` and `Ollama` available.
- **NER:** Named Entity Recognition via fast and reliable `flair` framework.
- **HTML Editor**: By default **dygest** will create a `.html` file that can be viewed in standard browsers and combines summaries, TOC and NER for your text. It features a text editor for you to make further changes.

## Requirements
- `Python >= 3.10` 
- API Keys for `OpenAI` and/or `Groq` *or* a running `Ollama` instance

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
 -----------------------------------------
 DYGEST is a designed to extract meaningful insights from your text documents. It can generate summaries, create tables of contents (TOC), and
 perform Named Entity Recognition (NER) to identify and categorize key information within your documents.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --files               -f      TEXT                         Path to the input folder or .txt file. [default: None] [required]                  │
│    --output_dir          -o      TEXT                         Folder where digests should be saved. [default: output]                            │
│    --service             -s      [ollama|openai|groq]         Select the LLM service. [default: groq]                                            │
│    --model               -m      TEXT                         Provide the LLM model name for creating digests.  Defaults to                      │
│                                                               "llama-3.1-70b-versatile" (Groq), "gpt-4o-mini (OpenAI) or "llama3.1" (Ollama).    │
│                                                               [default: None]                                                                    │
│    --temperature         -t      FLOAT                        Temperature of LLM. [default: 0.1]                                                 │
│    --dygest              -d      [sum|toc]                    Create summaries or a table of contents (TOC). [default: toc]                      │
│    --max_tokens                  INTEGER                      Maximum number of tokens per chunk. [default: 1000]                                │
│    --ner                                                      Enable Named Entity Recognition (NER). Defaults to True. [default: True]           │
│    --lang                -l      [auto|ar|de|da|en|fr|es|nl]  Language of file(s) for NER. Defaults to auto-detection. [default: auto]           │
│    --precise             -p                                   Enable precise mode for NER. Defaults to fast mode.                                │
│    --verbose             -v                                   Enable verbose output.                                                             │
│    --install-completion                                       Install completion for the current shell.                                          │
│    --show-completion                                          Show completion for the current shell, to copy it or customize the installation.   │
│    --help                                                     Show this message and exit.                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Examples

### Generate Summaries with Default Settings:
```shell
dygest --files ./documents
```

### Create Table of Contents and Enable Verbose Output:
```shell
dygest --files ./documents --dygest toc --verbose
```

### Perform NER with Precise Mode and Specify Language:
```shell
dygest --files ./documents --ner --precise --lang en
```

### Use OpenAI Service with a Specific Model and Increased Temperature:
```shell
dygest --files ./documents --service openai --model gpt-4 --temperature 0.2
```
