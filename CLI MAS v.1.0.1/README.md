Please extract the CLI MAS folder to any directory,
put some .txt agents in the same directory as mas.py,
and run mas.py, using python 3.9 or higher, after installing these requirements:

ollama
PyYAML
colorama
pyttsx3
sentence-transformers
faiss-cpu 

Thanks, and have fun. - Solomon

# CLI-MAS

**CLI Multi-Agent System (CLI-MAS)** is a sophisticated, command-line-based multi-agent AI conversation simulator. It leverages local language models via Ollama to create dynamic and engaging interactions between multiple, customizable AI personas. The system is designed to be highly extensible, configurable, and observable.

## Key Features

- **Multi-Agent Conversations**: Simulate roundtable discussions between multiple AI agents, each with a unique persona defined by a simple text file.
- **Local LLM Support**: Powered by Ollama, allowing you to run powerful language models entirely on your local machine.
- **Configurable Personas**: Easily create new agents by adding `.txt` files. Personas can have custom system prompts, response temperatures, and example-driven response styles.
- **Retrieval-Augmented Generation (RAG)**: Enhance agent responses by providing them with a set of example phrases. The system uses a FAISS vector index to retrieve the most relevant examples for a given context, leading to more consistent and in-character replies.
- **Tool Integration**: Agents can use simple tools to perform actions like calculations. The tool system is easily extensible.
- **Persistence**:
    - **Conversation History**: Save and resume conversations.
    - **Session Logging**: Detailed logs of each session are saved to timestamped files for analysis.
    - **UI Customization**: Agent name colors and TTS voices are remembered across sessions.
- **Text-to-Speech (TTS)**: Hear the conversation unfold with `pyttsx3` integration.
- **Resource Management**: Includes garbage collection intervals to manage memory during long conversations.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3**: The core runtime for the application.
2.  **Ollama**: The platform for running local language models. You can download it from [ollama.com](https://ollama.com/).
3.  **Ollama Models**: At least one language model installed. By default, the application uses `dolphin-mistral:latest`. You can pull models using the command:
    ```sh
    ollama pull dolphin-mistral:latest
    ```

## Installation

1.  **Clone the repository or download the source code.**

2.  **Navigate to the project directory**:
    ```sh
    cd Realgoblin MAS/
    ```

3.  **Install the required Python packages**:
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created containing the necessary packages like `py-ollama`, `PyYAML`, `colorama`, `pyttsx3`, `sentence-transformers`, `faiss-cpu` etc.)*

## Usage

Run the simulation from within the `MAS` directory:

```sh
cd MAS
python mas.py [OPTIONS]
```

### Command-Line Options

| Argument              | Default Value                       | Description                                                                                              |
| --------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `--turns`             | `6`                                 | The total number of turns in the conversation.                                                           |
| `--agents`            | `"all"`                             | Comma-separated list of agent personas to load (e.g., `alex,jordan`). Use "all" to load all `.txt` files. |
| `--model`             | (Per-agent default)                 | Override the Ollama model for all agents.                                                                |
| `--history`           | `False`                             | Enable saving and loading of the full conversation history.                                              |
| `--clear-history`     | (N/A)                               | Deletes the saved conversation history and exits.                                                        |
| `--session-log`       | `True`                              | Enables detailed session logging to a timestamped file in the `history/` directory.                      |
| `--no-session-log`    | (N/A)                               | Disables session logging.                                                                                |
| `--no-tts`            | (N/A)                               | Disable Text-to-Speech output for the session.                                                           |
| `--help-tools`        | (N/A)                               | Display a list of available agent tools and exit.                                                        |
| **RAG Options**       |                                     |                                                                                                          |
| `--rag` / `--no-rag`  | `True` (enabled)                    | Enable or disable Retrieval-Augmented Generation (RAG).                                                  |
| `--examples-k`        | `3`                                 | The number of relevant examples to inject into the context for each agent turn.                          |
| `--embed-model`       | `all-mpnet-base-v2`                 | The sentence-transformer model to use for embedding RAG examples.                                        |

### Example

To run a 10-turn conversation with agents named "Alex" and "Jordan":

```sh
python mas.py --agents alex,jordan --turns 10
```

## Creating Agents

Creating a new agent is simple:

1.  Create a new `.txt` file in the `MAS/` directory (e.g., `my_agent.txt`).
2.  The content of the file serves as the agent's **system prompt**.

### Advanced Persona Configuration

You can add special sections to the `.txt` file for more control.

#### Front Matter

To set a specific temperature for an agent (controlling the randomness of its responses), add a front-matter block at the very top of the file:

```yaml
---
temp: 0.7
---
You are a calm and calculated AI assistant.
```

#### RAG Examples

To provide in-character response examples for RAG, add an `### EXAMPLES` section. Separate each example with a blank line.

```
You are a pirate.

### EXAMPLES

Ahoy, matey! Shiver me timbers!

Walk the plank, ye scallywag!

I've got a treasure chest full o' gold!
```

## Configuration

Global settings are managed in the `MAS/config.yaml` file. This file is created with default values on the first run. Here you can configure default models, blacklist certain models, set history length, and toggle features on or off. 
