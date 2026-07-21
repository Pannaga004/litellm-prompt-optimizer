import time
from litellm import completion
from .evaluator import evaluate_response
from .optimizer import build_next_prompt

EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# Used whenever a config doesn't supply its own "evaluation_steps" -- keeps
# the judge from running with an empty rubric on prompts that aren't the
# SRS use case this pipeline was originally built for.
DEFAULT_EVALUATION_STEPS = [
    "Check that the response directly and completely addresses what the user's prompt asked for.",
    "Check that the response is factually accurate and does not fabricate information, sources, or details.",
    "Check that the response follows any explicit formatting, structural, or length instructions given in the prompt.",
    "Check that the response is well-organized, internally consistent, and easy to follow.",
    "Check that the response stays on topic and does not include irrelevant, off-topic, or filler content.",
    "Check that the response is appropriately detailed for the request -- neither too shallow nor unnecessarily verbose.",
]


def _sum_usage(*usages):
    total = dict(EMPTY_USAGE)
    for usage in usages:
        for key in total:
            total[key] += (usage or {}).get(key, 0)
    return total


def _generate(model, contents):
    """
    Calls LiteLLM's completion() once and returns (response_text, usage_dict).
    `model` is any LiteLLM-supported model string, e.g. "gemini/gemini-2.0-flash",
    "groq/llama-3.3-70b-versatile", "gpt-4o", "claude-3-5-sonnet-20241022".
    LiteLLM resolves the right provider/API key from that string automatically.

    num_retries=3: LiteLLM retries transient failures (rate limits, timeouts,
    brief API hiccups) up to 3 times automatically -- this only happens
    because we pass num_retries explicitly; it is not LiteLLM's default.

    Raises if the model returns no usable text (e.g. blocked by safety filters).
    """
    response = completion(
        model=model,
        messages=[{"role": "user", "content": contents}],
        temperature=0,
        num_retries=3,
        max_tokens=4096,
    )

    text = response.choices[0].message.content
    if not text:
        raise RuntimeError(
            "Generation model returned no usable text (possibly blocked by safety filters)."
        )

    usage = getattr(response, "usage", None)
    tokens = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    } if usage is not None else dict(EMPTY_USAGE)
    return text, tokens


def _describe_deltas(previous_results, new_results):
    """
    Logs a per-task score delta between two runs so a net-positive average
    doesn't silently hide individual task regressions (or vice versa).
    Matches tasks by task_id.
    """
    previous_by_id = {r["task_id"]: r["score"] for r in previous_results}
    for r in new_results:
        old_score = previous_by_id.get(r["task_id"])
        if old_score is None:
            continue
        delta = r["score"] - old_score
        direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
        print(f"      Task {r['task_id']}: {old_score:.3f} -> {r['score']:.3f}  ({direction}, {delta:+.3f})")


def _print_summary(session_metrics, best_version_label, best_average_score,
                    total_gen_tokens, total_judge_tokens, total_optimizer_tokens,
                    total_tokens_all, total_task_time_seconds, avg_task_time_seconds,
                    total_tasks_processed):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Loops executed        : {session_metrics['loops_executed']}")
    print(f"Best version          : {best_version_label}  (score: {best_average_score:.3f})")
    print(f"Tasks processed       : {total_tasks_processed}")
    print(f"Total task time       : {total_task_time_seconds:.2f}s")
    print(f"Avg time per task     : {avg_task_time_seconds:.2f}s")
    print(f"Tokens (generation)   : {total_gen_tokens['total_tokens']}")
    print(f"Tokens (judge)        : {total_judge_tokens['total_tokens']}")
    print(f"Tokens (optimizer)    : {total_optimizer_tokens['total_tokens']}")
    print(f"Tokens (total)        : {total_tokens_all['total_tokens']}")
    print("=" * 70)


def run_optimization_pipeline(payload):
    """
    Runs the full generate -> evaluate -> optimize loop, regression-safe.

    Each loop:
      1. Generate a response per task using the current candidate prompt.
      2. Evaluate each response with the judge model (GEval; AnswerRelevancy
         shown alongside for visibility).
      3. Compare this candidate's average score against the BEST version seen
         so far. If it's at least as good, it becomes the new best. If it
         regressed, the best-so-far is kept and optimization continues from
         ITS weaknesses again, rather than building on a worse prompt.
      4. If the best score clears score_threshold (or max_loops is hit), stop.
      5. Otherwise, ask the optimizer for new instructions targeting the best
         version's recurring weaknesses, merge them into a bounded (<=10
         bullet) "## Additional Instructions" block, and try that as the next
         candidate (a new prompt VERSION: V0 = original, V1, V2, ...).

    Returns the BEST version found, not necessarily the last one generated --
    a regressed final candidate is recorded in prompt_versions for visibility
    but is never returned as the result.

    Generation and judge models are both any LiteLLM-supported model string,
    resolved entirely from payload["generation_model"] / payload["eval_model"]
    (which cli.py's build_payload() reads from .env). Nothing in this file
    ever touches os.environ directly.
    """
    # Was previously hardcoded to 0.85 -- now config-driven so it can
    # actually be tuned from config.json without touching code.
    score_threshold = payload.get("score_threshold", 0.85)
    max_loops = payload.get("max_loops", 3)
    tasks = payload.get("tasks", [])
    initial_prompt = payload.get("initial_prompt", "")
    optimizer_instructions = payload.get("optimizer_prompt", "")
    evaluation_steps = payload.get("evaluation_steps") or DEFAULT_EVALUATION_STEPS

    # Deterministic (non-LLM) section presence/order check -- see
    # evaluator.check_required_sections. Optional: if config.json doesn't
    # define required_sections, this check is simply skipped and scoring
    # is purely GEval, same as before.
    required_sections = payload.get("required_sections", [])
    section_penalty_per_missing = payload.get("section_penalty_per_missing", 0.15)
    section_penalty_for_order = payload.get("section_penalty_for_order", 0.05)

    # Final score = geval_weight*GEval + relevancy_weight*AnswerRelevancy,
    # then the section penalty is subtracted on top. GEval carries more
    # weight by default since it's the only metric that knows the
    # SRS-specific rubric; AnswerRelevancy is a generic relevance check.
    geval_weight = payload.get("geval_weight", 0.7)
    relevancy_weight = payload.get("relevancy_weight", 0.3)

    gen_model = payload.get("generation_model", "")
    eval_model = payload.get("eval_model", "")

    judge_max_response_chars = payload.get("eval_max_response_chars", 4000)

    session_metrics = {"loops_executed": 0, "runs": []}

    prompt_versions = [{"version": "V0", "prompt": initial_prompt}]
    version_counter = 0

    current_prompt = initial_prompt
    current_version_label = "V0"

    best_prompt = None
    best_version_label = None
    best_average_score = None
    best_loop_results = None

    total_gen_tokens = dict(EMPTY_USAGE)
    total_judge_tokens = dict(EMPTY_USAGE)
    total_optimizer_tokens = dict(EMPTY_USAGE)
    total_task_time_seconds = 0.0
    total_tasks_processed = 0

    loop = 0
    while loop < max_loops:
        loop += 1
        session_metrics["loops_executed"] = loop
        print(f"\n[Loop {loop}/{max_loops}] Testing {current_version_label} against {len(tasks)} task(s)...")

        loop_results = []
        task_scores = []

        for task in tasks:
            task_start = time.time()
            final_prompt = current_prompt.replace("{user_query}", task["query"])

            # 1. Generate
            generated_response, gen_tokens = _generate(
                model=gen_model,
                contents=final_prompt
            )
            total_gen_tokens = _sum_usage(total_gen_tokens, gen_tokens)

            # 2. Evaluate
            score, feedback, judge_tokens = evaluate_response(
                task=task["query"],
                response=generated_response,
                eval_model=eval_model,
                evaluation_steps=evaluation_steps,
                threshold=score_threshold,
                max_response_chars=judge_max_response_chars,
                required_sections=required_sections,
                geval_weight=geval_weight,
                relevancy_weight=relevancy_weight,
                section_penalty_per_missing=section_penalty_per_missing,
                section_penalty_for_order=section_penalty_for_order
            )
            total_judge_tokens = _sum_usage(total_judge_tokens, judge_tokens)

            task_elapsed = time.time() - task_start
            total_task_time_seconds += task_elapsed
            total_tasks_processed += 1

            task_scores.append(score)
            loop_results.append({
                "task_id": task["id"],
                "task_name": task["query"],
                "response": generated_response,
                "score": score,
                "feedback": feedback,
                "time_seconds": task_elapsed
            })
            print(f"    - Task {task['id']}: score={score:.2f}  (took {task_elapsed:.1f}s)")

            # Rate limiting cooldown for free tier key stability
            time.sleep(5)

        average_score = sum(task_scores) / len(task_scores) if task_scores else 0
        print(f"[Loop {loop}] {current_version_label} average score: {average_score:.3f} (threshold: {score_threshold})")

        session_metrics["runs"].append({
            "loop": loop,
            "prompt_version": current_version_label,
            "prompt_used": current_prompt,
            "average_score": average_score,
            "task_results": loop_results
        })

        # 3. Compare against best-so-far (regression-safe)
        if best_average_score is None or average_score >= best_average_score:
            if best_loop_results is not None:
                print(f"  [+] {current_version_label} improved on {best_version_label} -- keeping it.")
                print("      Per-task deltas:")
                _describe_deltas(best_loop_results, loop_results)
            best_prompt = current_prompt
            best_version_label = current_version_label
            best_average_score = average_score
            best_loop_results = loop_results
        else:
            print(f"  [!] {current_version_label} regressed vs {best_version_label} "
                  f"({average_score:.3f} < {best_average_score:.3f}). Continuing from {best_version_label}.")
            print("      Per-task deltas:")
            _describe_deltas(best_loop_results, loop_results)

        if best_average_score >= score_threshold:
            print(f"[+] Score threshold reached ({best_version_label}: {best_average_score:.3f}). Stopping.")
            break

        if loop == max_loops:
            print(f"[!] Maximum loops reached. Best version: {best_version_label} ({best_average_score:.3f}).")
            break

        # 4. Optimize FROM the best-known version's own weaknesses, not
        # necessarily the one just tested (which may have regressed).
        print(f"[*] Optimizing from {best_version_label}'s weaknesses...")
        optimizer_input = [
            (r["task_name"], r["response"], r["score"], r["feedback"]) for r in best_loop_results
        ]
        next_version_label = f"V{version_counter + 1}"
        new_prompt, optimizer_tokens = build_next_prompt(
            current_prompt=best_prompt,
            current_version=next_version_label,
            results=optimizer_input,
            eval_model=eval_model,
            score_threshold=score_threshold,
            optimizer_instructions=optimizer_instructions
        )
        total_optimizer_tokens = _sum_usage(total_optimizer_tokens, optimizer_tokens)

        if new_prompt == best_prompt:
            print(f"[*] Optimizer found no recurring weaknesses to fix. Stopping at {best_version_label}.")
            break

        version_counter += 1
        prompt_versions.append({"version": next_version_label, "prompt": new_prompt})
        print(f"[+] Prompt candidate created -> {next_version_label} (will be tested next loop)")
        print(f"  {'-' * 61}")
        for line in new_prompt.splitlines():
            print(f"  {line}")
        print(f"  {'-' * 61}")

        current_prompt = new_prompt
        current_version_label = next_version_label

    avg_task_time_seconds = (
        total_task_time_seconds / total_tasks_processed if total_tasks_processed else 0
    )
    total_tokens_all = _sum_usage(total_gen_tokens, total_judge_tokens, total_optimizer_tokens)

    session_metrics["token_usage"] = {
        "generation": total_gen_tokens,
        "judge": total_judge_tokens,
        "optimizer": total_optimizer_tokens,
        "total": total_tokens_all
    }
    session_metrics["timing"] = {
        "total_task_time_seconds": total_task_time_seconds,
        "tasks_processed": total_tasks_processed,
        "avg_task_time_seconds": avg_task_time_seconds
    }
    session_metrics["prompt_versions"] = prompt_versions
    session_metrics["best_version"] = best_version_label
    session_metrics["best_average_score"] = best_average_score

    _print_summary(
        session_metrics, best_version_label, best_average_score,
        total_gen_tokens, total_judge_tokens, total_optimizer_tokens,
        total_tokens_all, total_task_time_seconds, avg_task_time_seconds,
        total_tasks_processed
    )

    return {
        "original_prompt": initial_prompt,
        "final_optimized_prompt": best_prompt,
        "final_prompt_version": best_version_label,
        "final_average_score": best_average_score,
        "telemetry": session_metrics
    }