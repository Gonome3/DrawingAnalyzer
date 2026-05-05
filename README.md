## Requirements

Python:

   Pymupdf


## Installation

Just copy down the repo

The program is designed to communicate with an ollama instance.

The repo comes with a config.example.json file, the config file is inputted when the program is run to let you have premade configs for different models and API, inside the config file you can also set parameters such as the context window of the model being called.

Important: Be wary of https timeouts due to large models needing to load or taking long times to respond.

## Usage commands:

`python pdf_reader.py input.pdf --coords`               # print JSON to stdout

`python pdf_reader.py input.pdf --coords -o output.json`     # write to file

`python pdf_reader.py input.pdf`                          # original plain-text mode still works

Example command:

`python pdf_reader.py input.pdf --coords --compact --extract --config config-qwen3-vl.json`
call pdf reader, input test.pdf extract coords of text, compact the prompt for less token usage use the qwen3-vl config

`python pdf_reader.py "SAP A\Axel\21015-A.pdf" --compact --extract --config config-qwen3-vl.json -v -o SAP-A-Axel-21015-A.json`
