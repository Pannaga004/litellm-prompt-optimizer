# Prompt Optimizer

A provider-agnostic prompt optimization pipeline powered by [LiteLLM](https://github.com/BerriAI/litellm) and [DeepEval](https://github.com/confident-ai/deepeval).

Give it an initial prompt, a set of tasks, and an evaluation rubric — it will generate responses, score them with an LLM judge, and automatically rewrite the prompt to fix recurring weaknesses across multiple optimization loops.

## Features

- **Provider-agnostic** — works with any LiteLLM-supported model (Gemini, Groq, OpenAI, Anthropic, etc.) for both generation and judging, just by changing a model string.
- **Automated feedback loop** — generates responses, scores them, and rewrites the prompt to target only *recurring* weaknesses (not one-off mistakes).
- **Regression-safe** — always keeps the best-scoring prompt version found so far, even if a later candidate performs worse.
- **Blended scoring** — combines a custom rubric (`GEval`) with a generic relevance check (`AnswerRelevancyMetric`).
- **Deterministic section checks** — optionally verifies required sections are present and in order, independent of the LLM judge's leniency.
- **Full telemetry** — tracks token usage, timing, and per-task score deltas across every loop.

## Installation

```bash
pip install prompt_optimizer
```

## Setup

1. Create a `.env` file in your project root with your model names and API key(s):

```env
    GENERATION_MODEL_NAME=gemini/gemini-2.0-flash
    EVAL_MODEL_NAME=gemini/gemini-2.0-flash
    GEMINI_API_KEY=your-api-key-here
```

    Any [LiteLLM-supported model string](https://docs.litellm.ai/docs/providers) works here (e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`, `groq/llama-3.3-70b-versatile`), as long as the matching API key env var is also set.

2. Create a `config.json` describing your prompt, tasks, and evaluation rubric:

```json
    {
      "max_loops": 3,
      "initial_prompt": "You are a helpful assistant. Answer: {user_query}",
      "optimizer_prompt": "You are a Prompt Optimization Expert...",
      "evaluation_steps": [
        "Check that the response directly answers the question."
      ],
      "tasks": [
        { "id": 1, "query": "What is the capital of France?" }
      ]
    }
```

## Usage

```bash
prompt-optimizer
```

Optional flags:

```bash
prompt-optimizer --config config.json --env .env --output results.json
```

| Flag | Description |
|------|-------------|
| `-c`, `--config` | Path to config JSON (default: `config.json`) |
| `-e`, `--env` | Path to `.env` file (default: `.env`) |
| `-o`, `--output` | Save results as JSON (generated responses are terminal-only and never saved to disk) |

## How it works

1. **Generate** — the current prompt is run against every task using the generation model.
2. **Evaluate** — each response is scored by an LLM judge using your custom rubric, blended with a generic relevance check, plus an optional deterministic required-section check.
3. **Compare** — the new average score is compared against the best version seen so far. Regressions are kept on record but never become the new baseline.
4. **Optimize** — if the score is below threshold, an optimizer model analyzes recurring weaknesses and proposes new prompt instructions, which are merged into a bounded instruction set.
5. Repeat until the score threshold is hit or `max_loops` is reached.

The final result includes the best-scoring prompt version, its score, and full telemetry (token usage, timing, prompt version history).

## License

MIT