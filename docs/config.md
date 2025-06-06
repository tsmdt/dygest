---
layout: default
title: Configuration
nav_order: 3
description: "dygest: Configuration"
permalink: /config
---

# Configuration ⚙️

Before running `dygest` you need to set up your LLM configuration first.

**All configuration parameters are stored in a `.env` file in the projects root directory**. If you have used `pip` for installing `dygest`, this root directory is inside your `venv`.

## Table of Contents

- [How to configure your `.env`](#how-to-configure-your-env)
- [`.env` settings](#env-settings)
  - [Model Settings](#model-settings)
    - [Custom LLM Providers](#custom-llm-providers)
    - [LIGHT_MODEL](#light_model-required)
    - [EXPERT_MODEL](#expert_model-required)
    - [EMBEDDING_MODEL](#embedding_model-required)
  - [LLM Parameters](#llm-parameters)
    - [TEMPERATURE](#temperature)
    - [SLEEP](#sleep)
    - [CHUNK_SIZE](#chunk_size)
  - [Named Entity Recognition (NER)](#named-entity-recognition-ner)
    - [NER](#ner)
    - [NER_LANGUAGE](#ner_language)
    - [NER_PRECISE](#ner_precise)
  - [API Keys](#api-keys)
    - [OPENAI_API_KEY](#openai_api_key)
    - [GROQ_API_KEY](#groq_api_key)
  - [Custom Settings](#custom-settings)
    - [OLLAMA_API_BASE](#ollama_api_base)
    - [Adding custom settings](#adding-custom-settings)
- [Manually creating your `.env`](#manually-creating-your-env)
- [Configuring with `dygest config`](#configuring-with-dygest-config)
  - [Options](#options)
  - [Model Options](#model-options)
  - [LLM Parameter Options](#llm-parameter-options)
  - [NER Options](#ner-options)
  - [Viewing Configuration](#viewing-configuration)
  - [Example](#example)
- [Recap: putting it together](#recap-putting-it-together)
- [Troubleshooting & Tips](#troubleshooting--tips)

## How to configure your `.env`

There are 2 ways to configure your `.env`:

1. Manually editing the `.env` with a code editor.
2. Running `dygest config` in CLI.

Use the provided `.env.example` (root directory) as a blueprint.

## `.env` settings

The `.env.example` file serves as a template for your own `.env` file. Each key corresponds to a setting used by **dygest** at runtime.

```dotenv
LIGHT_MODEL='ollama/gemma3:12b'
EXPERT_MODEL='groq/llama-3.3-70b-versatile'
EMBEDDING_MODEL='ollama/nomic-embed-text:latest'
TEMPERATURE='0.1'
SLEEP='0'
CHUNK_SIZE='1000'
NER='True'
NER_LANGUAGE='auto'
NER_PRECISE='False'

# API KEYS
OPENAI_API_KEY=''
GROQ_API_KEY=''

# CUSTOM SETTINGS
OLLAMA_API_BASE='http://localhost:11434'
```

### Model Settings

The LLM setup follows the `litellm` notation:

- Pattern: `LLM_PROVIDER_NAME/MODEL_NAME`
- Example 1: `openai/gpt-4o-mini`
- Example 2: `ollama/qwen2.5:3b-instruct`

#### Custom LLM Providers

If you are using a *custom openAI compatible* API provider tho is not part of `litellm` (see their providers list here: https://docs.litellm.ai/docs/providers) you have to **specify an `API base url`, an `api key` and use a slightly different `model name`**:

Example:

```dotenv
MYOWN_API_BASE='http://example.org/llm-service/v1'
MYOWN_API_KEY=proj_123...
LIGHT_MODEL=openai/myown/gemma3:12b
EXPERT_MODEL=openai/myown/gemma3:24b
```

#### LIGHT_MODEL (required)

Model to use for lighter-weight tasks (e.g., summarization, keyword extraction).

*Example:*

```dotenv
LIGHT_MODEL='ollama/gemma3:12b'
```  

#### EXPERT_MODEL (required)

Model to use for heavier tasks (e.g., generating a Table of Contents).

*Example:*

```dotenv
EXPERT_MODEL='groq/llama-3.3-70b-versatile'
```

#### EMBEDDING_MODEL (required)

Model used to generate embeddings (e.g., for clustering or similarity).

*Example:*

```dotenv
EMBEDDING_MODEL='ollama/nomic-embed-text:latest'
```

### LLM Parameters

#### TEMPERATURE

Sampling temperature for LLM calls (float between 0.0 and 1.0).

Lower values → more deterministic output; higher values → more creative.  

*Example:*

```dotenv
TEMPERATURE='0.1'
  ```

#### SLEEP

Time (in seconds) to wait between LLM requests. Useful for rate-limit throttling.

*Example:*

```dotenv
SLEEP='0'
```

#### CHUNK_SIZE

Maximum number of tokens per chunk when splitting large documents.  

*Example:*

```dotenv
CHUNK_SIZE='1000'
```

### Named Entity Recognition (NER)

#### NER

Enable or disable NER altogether. Accepts `True` or `False`.  
  
*Example:*

```dotenv
NER='True'
```

#### NER_LANGUAGE

Language code for the NER pipeline (e.g., `en`, `de`, or `auto`). If you pass an invalid code, **dygest** falls back to `auto`.

*Example:*

```dotenv
NER_LANGUAGE='auto'
```

#### NER_PRECISE

Enable precise (slower) NER mode.

- `True` → Precise mode (higher accuracy, slower)  
- `False` → Fast mode (lower accuracy, faster)

*Example:*

```dotenv
NER_PRECISE='False'
```

### API Keys

#### OPENAI_API_KEY
  
Your OpenAI API key (if using OpenAI).

*Example:*

```dotenv
OPENAI_API_KEY='sk-xyz...'
```

#### GROQ_API_KEY

Your Groq API key (if using Groq).  
  
*Example:*

```dotenv
GROQ_API_KEY='groq-abc...'
```

### Custom Settings

#### OLLAMA_API_BASE

Base URL for your Ollama API instance.

*Example:*

```dotenv
OLLAMA_API_BASE='http://localhost:11434'
```

#### Adding custom settings

You can also add arbitrary key–value pairs at the end of `.env` if you have custom configuration needs. For example:

```dotenv
MY_API_KEY='some-value'
NEW_API_BASE='some-value'
```

---

## Manually creating your `.env`

1. **Copy** the example template in the root directory of the project and **rename** it to `.env`:  

```bash
cp .env.example .env
```

2. **Open** `.env` in your preferred editor and **fill in** all required values:

    - Ensure `LIGHT_MODEL`, `EXPERT_MODEL`, and `EMBEDDING_MODEL` are non‐empty.  
    - If you plan to use NER, set `NER='True'`; otherwise, set `NER='False'`.  
    - If you enable NER, pick a valid `NER_LANGUAGE` (e.g., `en` for English).  
    - Provide your `OPENAI_API_KEY` or `GROQ_API_KEY` as needed (leave blank if unused).

3. **Save** `.env` and re-run any `dygest` commands. **dygest** will automatically load the values from `.env`.

> **Note:**  
> - If `.env` does not exist when **dygest** starts, it will create a new `.env` populated with default values (as defined in `dygest/config.py`).  
> - After the file is created, you can overwrite any value with either direct editing or via the CLI (`dygest config ...`).

---

## Configuring with `dygest config`

Instead of manually editing `.env` each time, you can run:

```bash
dygest config [OPTIONS]
```

This command uses [`python-dotenv`](https://pypi.org/project/python-dotenv/) under the hood to read/write individual keys in `.env`. If you pass no options, it shows help.

### Options

`--view_config` / `-v`

- Print all current configuration values (as read from `.env`). Useful to verify what you have set. Example: `dygest config -v`

`--add_custom KEY=VALUE` / `-add KEY=VALUE`

- Add or overwrite a custom key–value pair that is not already defined. The format must be `KEY=VALUE` (no spaces around `=`).

    ```bash
    dygest config --add_custom GROQ_API_KEY=groq-abc...
    ```

### Model Options

`--light_model MODEL_NAME` / `-l MODEL_NAME`

- Set `LIGHT_MODEL` to the specified model string.

    ```bash
    dygest config --light_model "openai/gpt-4o-mini"
    ```

`--expert_model MODEL_NAME` / `-x MODEL_NAME`

- Set `EXPERT_MODEL` to the specified model string.

    ```bash
    dygest config --expert_model "groq/llama3.3-70b-versatile"
    ```

`--embedding_model MODEL_NAME` / `-e MODEL_NAME`

- Set `EMBEDDING_MODEL` to the specified model string.

    ```bash
    dygest config -e "openai/text-embedding-3"
    ```

### LLM Parameter Options

`--temperature FLOAT` / `-t FLOAT`

- Set `TEMPERATURE` (e.g., `0.1`, `0.7`, etc.). Must be parseable as a float.

    ```bash
    dygest config -t 0.2
    ```

`--sleep FLOAT` / `-s FLOAT`

- Set `SLEEP` (seconds to pause between requests).

    ```bash
    dygest config --sleep 8
    ```

`--chunk_size INT` / `-c INT`

- Set `CHUNK_SIZE` (maximum tokens per chunk). Must be an integer.

    ```bash
    dygest config --chunk_size 800
    ```

### NER Options

`--ner` / `--no-ner`

- Enable or disable NER. If you pass `--ner`, it sets `NER=True`. If you pass `--no-ner`, it sets `NER=False`.

    ```bash
    dygest config --ner          # turns NER on
    dygest config --no-ner       # turns NER off
    ```

`--precise` / `--fast`

- Toggle precise‐mode NER.

  - `--precise` sets `NER_PRECISE=True`
  - `--fast` sets `NER_PRECISE=False`

    ```bash
    dygest config --precise   # choose more accurate (but slower) NER
    dygest config --fast      # choose faster (but less accurate) NER
    ```

`--lang LANG_CODE` / `-lang LANG_CODE`

- Set `NER_LANGUAGE`. Must be one of the `NERlanguages` enum values (e.g., `en`, `de`, `auto`). If you supply an invalid code, **dygest** will warn you and fall back to `auto`.

    ```bash
    dygest config --lang en
    dygest config --lang auto
    ```

### Viewing Configuration

To simply view what's currently in `.env`, run:

```bash
dygest config --view_config
```

or:

```bash
dygest config -v
```

This prints all keys and their current values in a user‐friendly format. Required fields are marked `(required)`.

### Example

Suppose you want to:

1. Switch to OpenAI's GPT-4o as your lighter model  
2. Disable NER entirely  
3. Increase chunk size to 1200  

You can run:

```bash
dygest config --light_model "openai/gpt-4o" --no-ner --chunk_size 1200
```

After execution, the CLI will print the updated configuration and save those three keys to your `.env`.

---

## Recap: putting it together

1. **Copy the template**:

```bash
cp .env.example .env
```

2. **Either manually edit** `.env` OR use `dygest config` to set keys one at a time. For example:

```bash
dygest config \
    --light_model "openai/gpt-4"
    --expert_model "groq/llama-3.3-70b"
    --embedding_model "openai/text-embedding-3"
    --temperature 0.2
    --sleep 1
    --chunk_size 1200
    --no-ner
    --lang en
```

3. **Verify** by running:

```bash
dygest config -v
```

Make sure none of the required fields (`LIGHT_MODEL`, `EXPERT_MODEL`, `EMBEDDING_MODEL`) are blank.

4. **Run dygest** on your documents:

```bash
dygest run --files path/to/your/documents --summarize --keywords --toc
```

or:

```bash
dygest run --files path/to/your/documents -skt
```

At that point, **dygest** will read your `.env` values at startup and proceed accordingly.

---

## Troubleshooting & Tips

- If you ever see:
  
  ```bash
  … Please configure dygest first by running *dygest config* and set your LLMs.
  ```

  it means one or more of the required keys (`LIGHT_MODEL`, `EXPERT_MODEL`, `EMBEDDING_MODEL`) is still empty. Either edit `.env` manually or set them with `--light_model`, `--expert_model`, `--embedding_model`.

- If you pass an invalid `--lang` value for NER, you'll get a warning:
  
  ```bash
  … Warning: '<your‐lang>' is not a valid NER language. Using 'auto' instead.
  ```

- If your API keys change, just run, for example:

  ```bash
  dygest config --add_custom OPENAI_API_KEY=sk-newapikey
  ```

  This overwrites the existing key without affecting the other settings.
