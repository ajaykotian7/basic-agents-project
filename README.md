# Basic Agents Project

Hands-on examples of the building blocks behind LLM agents and the common agentic workflow patterns, written while following [this video](https://youtu.be/bZzyPscbtI8). Every example uses the OpenAI Python SDK with Pydantic models for structured output.

## Project structure

```
basics/
  structured_output.py   # Extract structured data into Pydantic models
  tools.py               # Function calling: the model calls a live weather API (Open-Meteo)
  retrieval.py           # Tool-based retrieval from a small JSON knowledge base
  kb.json                # Knowledge base used by retrieval.py
workflows/
  prompt-chaining.py     # Sequential LLM calls with a gate between steps
  routing.py             # Classify the request, then route it to a specialised handler
  parallelization.py     # Run independent checks concurrently with asyncio
  orchestration.py       # Orchestrator plans a blog post, workers write sections, a reviewer polishes it
  output.md              # Sample run of orchestration.py (plan, draft, review, final post)
```

## Concepts covered

### Basics

| Script | What it shows |
| --- | --- |
| `structured_output.py` | Using `client.chat.completions.parse` with a Pydantic `response_format` to pull calendar events out of free text. |
| `tools.py` | The full tool-calling loop. The model asks for `get_weather`, the script calls the Open-Meteo API, sends back the result, and receives a structured `WeatherReport`. |
| `retrieval.py` | The same loop, but the tool searches a knowledge base. The answer is returned with the id of the record it came from, and off-topic questions are handled. |

### Workflows

| Script | Pattern | Example |
| --- | --- | --- |
| `prompt-chaining.py` | **Prompt chaining** | Extract → validate (confidence gate) → parse event details → generate a confirmation email. |
| `routing.py` | **Routing** | Classify a request as `new_event`, `modify_event`, or `other`, then send it to the matching parser and confirmation generator. |
| `parallelization.py` | **Parallelization** | Run a calendar-intent check and a prompt-injection security check concurrently with `asyncio.gather`. Continue only if both pass. |
| `orchestration.py` | **Orchestrator–workers** | An orchestrator plans the sections of a blog post. A worker call writes each section in turn. A reviewer call then scores how well the sections fit together, suggests edits, and returns a polished final version. |

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies into .venv
uv sync

# Add your OpenAI API key
cp .env.example .env
# then edit .env and set MY_OPEN_AI_KEY
```

The scripts read the key from the `MY_OPEN_AI_KEY` environment variable (loaded from `.env` via `python-dotenv`).

## Running the examples

Run each script on its own:

```bash
uv run basics/structured_output.py
uv run basics/tools.py
uv run basics/retrieval.py

uv run workflows/prompt-chaining.py
uv run workflows/routing.py
uv run workflows/parallelization.py
uv run workflows/orchestration.py
```

Each script prints its results as labelled sections: a heading for each request, then the intermediate steps (tool calls, routing decisions, check results, parsed details) and the final answer. `routing.py` and `orchestration.py` also log progress lines with timestamps while they run.

To see what a full orchestrator–workers run looks like without spending tokens, open [`workflows/output.md`](workflows/output.md). That sample was generated with `target_length=1000`. The script is set to 200 words by default to keep test runs cheap.

> **Note:** Every run makes real OpenAI API calls and costs tokens. `orchestration.py` is the most expensive. It makes one planning call, one call per blog section, and one review call that rewrites the whole post.

## Model

The examples use the `gpt-6-astra` model. To try a different model, change the `model=` argument in the scripts.

## Credits

I wrote all of the base code while following [the video](https://youtu.be/bZzyPscbtI8). [Claude](https://claude.com/claude-code) was used to clean up the print statements so the output is easier to read. It also helped with a code review, a few small bug fixes, and this README.
