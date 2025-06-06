---
layout: default
title: Setup
nav_order: 2
description: "dygest: Setup"
permalink: /setup
---

# Setup 🛠️

- [Requirements](#requirements)
- [Installation](#installation)
  - [Install with `pip`](#install-with-pip)
    - [Create a Python virtual environment](#create-a-python-virtual-environment)
    - [Activate the environment](#activate-the-environment)
    - [Install dygest](#install-dygest)
  - [Install from source](#install-from-source)
    - [Clone this repository](#clone-this-repository)
    - [Create a Python virtual environment](#create-a-python-virtual-environment-1)
    - [Activate the environment](#activate-the-environment-1)
    - [Install dygest](#install-dygest-1)

## Requirements

- 🐍 Python `>=3.10` 
- 🔑 API keys for LLM services like `OpenAI`, `Anthropic` and `Groq` *and / or* a running `Ollama` instance

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
