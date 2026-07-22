import argparse
import json
import os
import shutil
import sys
import warnings

from dotenv import load_dotenv

from .pipeline import run_optimization_pipeline

warnings.filterwarnings("ignore")


def copy_readme_if_missing():
    """
    Drops a copy of the README into the user's current working directory,
    the first time they run the CLI from that folder. Never overwrites an
    existing copy, so it only happens once per project folder.

    The README has to be bundled INSIDE the installed package for this to
    work (see setup.py's package_data + include_package_data=True, and
    make sure README.md is physically copied into the prompt_optimizer/
    package folder before building -- see build_readme_copy.py / build
    step notes below).
    """
    dest = os.path.join(os.getcwd(), "PROMPT_OPTIMIZER_README.md")
    if os.path.exists(dest):
        return  # already dropped here before, don't overwrite

    source = os.path.join(os.path.dirname(__file__), "README.md")
    if not os.path.exists(source):
        return  # README wasn't bundled into the installed package -- skip silently

    try:
        shutil.copy(source, dest)
        print(f"[+] Docs copied to {dest} (see it for full setup instructions)")
    except OSError:
        pass  # e.g. read-only directory -- not worth failing the whole run over


def wrap_text(text, width=63, indent="    "):
    """Simple word-wrap for terminal-friendly printing."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return "\n".join(f"{indent}{line}" for line in lines)


def print_formatted_results(result):
    telemetry = result.get("telemetry", {})
    runs = telemetry.get("runs", [])
    prompt_versions = telemetry.get("prompt_versions", [])
    token_usage = telemetry.get("token_usage", {})
    timing = telemetry.get("timing", {})

    print("\n" + "=" * 65)
    print("                    FINAL BENCHMARK RESULTS                    ")
    print("=" * 65)
    print(f"Loops executed: {telemetry.get('loops_executed', 0)}")

    for run in runs:
        print("\n" + "-" * 65)
        print(f"LOOP {run['loop']}  |  Prompt Version: {run.get('prompt_version', '?')}  "
              f"|  Average Score: {run['average_score']:.2f}")
        print("-" * 65)

        for t in run.get("task_results", []):
            print(f"\n  [Task {t['task_id']}] {t['task_name']}")
            print(f"  Score: {t['score']:.2f}  |  Time: {t.get('time_seconds', 0):.1f}s")

            print(f"\n  {'=' * 61}")
            print(f"  GENERATED RESPONSE")
            print(f"  {'=' * 61}")
            for line in t["response"].splitlines():
                print(wrap_text(line, width=59, indent="  "))

            print(f"\n  {'-' * 61}")
            print(f"  JUDGE FEEDBACK")
            print(f"  {'-' * 61}")
            for line in t["feedback"].splitlines():
                print(wrap_text(line, width=59, indent="      "))
            print(f"  {'.' * 61}")

    if len(prompt_versions) > 1:
        print("\n" + "=" * 65)
        print("                  PROMPT VERSION HISTORY                       ")
        print("=" * 65)
        version_scores = {r.get("prompt_version"): r["average_score"] for r in runs}
        for v in prompt_versions:
            score_str = f"{version_scores[v['version']]:.2f}" if v["version"] in version_scores else "n/a"
            print(f"\n  {v['version']}  (score: {score_str})")
            print(f"  {'-' * 61}")
            for line in v["prompt"].splitlines():
                print(wrap_text(line, width=59, indent="  "))

    print("\n" + "=" * 65)
    print(f"          FINAL OPTIMIZED PROMPT ({result.get('final_prompt_version', '?')}"
          f"  |  score: {result.get('final_average_score', 0):.2f})")
    print("=" * 65)
    print(result.get("final_optimized_prompt", ""))
    print("=" * 65)

    print("\n" + "=" * 65)
    print("                        RUN SUMMARY                             ")
    print("=" * 65)
    gen = token_usage.get("generation", {})
    judge = token_usage.get("judge", {})
    opt = token_usage.get("optimizer", {})
    total = token_usage.get("total", {})
    print(f"  Total tokens used:      {total.get('total_tokens', 0):,}  "
          f"(prompt: {total.get('prompt_tokens', 0):,}, completion: {total.get('completion_tokens', 0):,})")
    print(f"    - Generation:          {gen.get('total_tokens', 0):,}")
    print(f"    - Judge:               {judge.get('total_tokens', 0):,}")
    print(f"    - Optimizer:           {opt.get('total_tokens', 0):,}")
    print(f"  Tasks processed:        {timing.get('tasks_processed', 0)}")
    print(f"  Avg time per task:      {timing.get('avg_task_time_seconds', 0):.1f}s")
    print(f"  Total task time:        {timing.get('total_task_time_seconds', 0):.1f}s")
    print("=" * 65)

    print(f"\n{'=' * 65}\n")


def strip_responses_for_disk(result):
    """
    Return a deep copy of the result with generated responses removed.
    The response text is only ever shown in the terminal (print_formatted_results);
    it is never written to a saved results file.
    """
    redacted = json.loads(json.dumps(result))  # cheap deep copy
    for run in redacted.get("telemetry", {}).get("runs", []):
        for t in run.get("task_results", []):
            t.pop("response", None)
    return redacted


def load_config(config_path):
    """Load prompts, tasks, evaluation steps, and run settings from JSON."""
    if not os.path.exists(config_path):
        print(f"[!] Config file not found: {config_path}", file=sys.stderr)
        print("    Copy config.example.json to config.json and fill it in.", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)


def build_payload(config):
    """
    Build the pipeline payload from two sources:
      - .env: GENERATION_MODEL_NAME / EVAL_MODEL_NAME (LiteLLM model
        strings, e.g. "gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile",
        "gpt-4o", "claude-3-5-sonnet-20241022") and whatever API key(s)
        those models need (GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY,
        ANTHROPIC_API_KEY, etc.) -- LiteLLM reads the right key on its own
        based on the model string, so we never read or pass API keys
        ourselves anywhere in this codebase.
      - config.json: content settings (prompt, tasks, evaluation rubric,
        thresholds) -- things about WHAT is being optimized/tested.

    This is the ONLY place that touches os.environ for model selection --
    pipeline.py, evaluator.py, and optimizer.py all just receive plain
    strings/payload keys and never read os.environ themselves.
    """
    gen_model = os.environ.get("GENERATION_MODEL_NAME", "").strip()
    eval_model = os.environ.get("EVAL_MODEL_NAME", "").strip()

    if not gen_model or not eval_model:
        print(
            "[!] Set GENERATION_MODEL_NAME and EVAL_MODEL_NAME in your .env file.\n"
            "    Examples: 'gemini/gemini-2.0-flash', 'groq/llama-3.3-70b-versatile',\n"
            "              'gpt-4o', 'claude-3-5-sonnet-20241022'\n"
            "    Also make sure the matching API key env var is set (GEMINI_API_KEY,\n"
            "    GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY).",
            file=sys.stderr,
        )
        sys.exit(1)

    tasks = config.get("tasks", [])
    if not tasks:
        print("[!] config.json has no tasks. Add at least one task to tasks[].", file=sys.stderr)
        sys.exit(1)

    initial_prompt = config.get("initial_prompt", "")
    if not initial_prompt or "{user_query}" not in initial_prompt:
        print(
            "[!] config.json's initial_prompt must be set and must contain '{user_query}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    payload = dict(config)  # shallow copy so we don't mutate the loaded config
    payload["generation_model"] = gen_model
    payload["eval_model"] = eval_model

    return payload


def main():
    """
    Entry point used both by `python -m prompt_optimizer.cli` and by the
    installed `prompt-optimizer` terminal command (see setup.py's
    console_scripts entry point, which points here).
    """
    copy_readme_if_missing()

    parser = argparse.ArgumentParser(
        description="Run the LLM prompt optimizer using settings from config.json and .env."
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to the JSON config with prompts, tasks, and run settings (default: config.json)",
    )
    parser.add_argument(
        "-e", "--env",
        default=".env",
        help="Path to the .env file with API keys and model names (default: .env)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional path to also save the raw results as JSON.",
    )
    args = parser.parse_args()

    load_dotenv(args.env)
    config = load_config(args.config)
    payload = build_payload(config)

    print("=" * 65)
    print("         LLM PROMPT OPTIMIZER                                ")
    print("=" * 65)
    print(f"[+] Config loaded from: {args.config}")
    print(f"[+] API keys and model names loaded from: {args.env}")
    print(f"[+] Generation model: {payload['generation_model']}  |  "
          f"Eval model: {payload['eval_model']}")
    print(f"[+] Tasks: {len(payload['tasks'])}  |  Max loops: {payload.get('max_loops', 3)}")
    print("[+] Starting the dynamic optimization pipeline... Please wait.")
    print("=" * 65 + "\n")

    final_result = run_optimization_pipeline(payload)

    print_formatted_results(final_result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(strip_responses_for_disk(final_result), f, indent=2)
        print(f"[+] Full results saved to {args.output} (generated responses are terminal-only, not saved)")


if __name__ == "__main__":
    main()