<BEGIN OUTLINE>
# How to Use Ollama LLM

## Introduction
- The Ollama Language Model (LLM) is a powerful tool for natural language processing.
- This guide will walk you through the process of setting up and using Ollama LLM.

## Prerequisites
- Familiarity with command line interface (CLI) on your operating system.
- An account on ollama.ai to download the app.

## Setup for Different Operating Systems
### Linux (WSL) Users
1. Open terminal and execute the following command: `curl https://ollama.ai/install.sh | sh`
2. Install the desired LLM by running: `ollama run [model_name]` or `ollama pull [model_name]`.
### macOS Users
1. Download the app from ollama.ai.
2. Open terminal and execute the following command to install Ollama CLI: `curl https://ollama.ai/install.sh | sh`
3. Install the desired LLM by running: `ollama run [model_name]` or `ollama pull [model_name]`.
### Windows Users
1. Download the app from ollama.ai.
2. Open PowerShell and execute the following command to install Ollama CLI: `iwr https://ollama.ai/install.sh -OutFile .\install.ps1; .\install.ps1`.
3. Install the desired LLM by running: `ollama run [model_name]` or `ollama pull [model_name]`.

## Deploying Mistral/Llama 2 or Other LLMs
1. Install the LLM which you want to use locally.
2. Open your terminal and execute the following command to deploy: `ollama run mistral` or `ollama run [model_name]`.

## References
- Ollama AI: <https://ollama.ai/>
- [Ollama CLI on GitHub](https://github.com/ollama-ai/ollama-cli)
<END OUTLINE>