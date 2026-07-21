import os
from google import genai
from .evaluator import evaluate_response
from .optimizer import build_next_prompt
from .pipeline import DEFAULT_EVALUATION_STEPS


def run_benchmark(payload):
    """
    Model names are read from the GENERATION_MODEL_NAME and EVAL_MODEL_NAME
    environment variables (set in .env), not from the payload.
    """
    score_threshold = 0.85
    max_loops = payload.get("max_loops", 3)
    tasks = payload.get("tasks", [])
    prompt = payload.get("initial_prompt", "")
    optimizer_instructions = payload.get("optimizer_prompt", "")
    evaluation_steps = payload.get("evaluation_steps") or DEFAULT_EVALUATION_STEPS

    gen_model = os.environ["GENERATION_MODEL_NAME"]
    gen_api_key = payload.get("generation_api_key", "")
    judge_api_key = payload.get("eval_api_key", "")

    genai_client = genai.Client(api_key=gen_api_key)

    session_metrics = {"loops_executed": 0, "runs": []}

    for loop in range(1, max_loops + 1):
        session_metrics["loops_executed"] = loop
        loop_results = []
        task_scores = []

        for task in tasks:
            final_prompt = prompt.replace("{user_query}", task["query"])

            # Generate
            response = genai_client.models.generate_content(
                model=gen_model,
                contents=final_prompt
            )
            generated_response = response.text

            # Evaluate
            score, feedback, _ = evaluate_response(
                task=task["query"],
                response=generated_response,
                eval_api_key=judge_api_key,
                evaluation_steps=evaluation_steps,
                threshold=score_threshold
            )

            task_scores.append(score)
            loop_results.append({
                "task_name": task["query"],
                "score": score,
                "feedback": feedback
            })

        average_score = sum(task_scores) / len(task_scores) if task_scores else 0
        session_metrics["runs"].append({"loop": loop, "prompt_used": prompt, "average_score": average_score})

        if average_score >= score_threshold or loop == max_loops:
            break

        # Optimize
        optimizer_input = [(r["task_name"], "", r["score"], r["feedback"]) for r in loop_results]
        prompt, _ = build_next_prompt(
            current_prompt=prompt,
            current_version=f"V{loop}",
            results=optimizer_input,
            eval_api_key=judge_api_key,
            score_threshold=score_threshold,
            optimizer_instructions=optimizer_instructions
        )

    return {
        "original_prompt": payload.get("initial_prompt"),
        "final_optimized_prompt": prompt,
        "telemetry": session_metrics
    }