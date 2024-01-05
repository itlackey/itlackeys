## ITLackey's Scripts

This repository contains a set of tools for various things, such as reviewing and modifying code files using autogen and aider.

### Configuration

The behavior of the scripts can be configured using the `.env` file and the `local.json` file.

#### .env.example

This file contains environment variable definitions that configure various aspects of the tooling:

```plaintext
#itl config
ITL_CACHE_SEED=42

# aider config
AIDER_MODEL="gpt-4-1106-preview"
OPENAI_API_KEY=sk-...
# OPENAI_API_BASE_URL="http://localhost:8000/v1" #used for local LLMs with aider
```

`ITL_CACHE_SEED` is used to set a seed for caching purposes.
`AIDER_MODEL` specifies the model of the AI to be used.
`OPENAI_API_KEY` is your OpenAI API key.
`OPENAI_API_BASE_URL` is the base URL for the OpenAI API, which can be pointed to a local server if needed.

#### local.json

This file provides configuration for local instances of language models:

```json
[{
    "api_key": "llm-studio",
    "model": "local",
    "base_url": "http://localhost:8000/v1"
}]
```

Each object in the array represents a different configuration, specifying the `api_key`, `model`, and `base_url` for a local language model instance.

### Usage

To use the tools in this repository, you will need to set up your environment by copying `.env.example` to `.env` and filling in your details. You can also modify `local.json` as needed to point to your local language model instances.

Once configured, you can run the `review_files.py` script to perform actions on your code files as specified by the command-line arguments or action files.

For more detailed usage instructions, refer to the comments and documentation within the `review_files.py` script.

### Contributing

Contributions to this repository are welcome. Please ensure that you follow the existing code conventions and include appropriate tests and documentation with your pull requests.

### License

This repository is licensed under the MIT License.
