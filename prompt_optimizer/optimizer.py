from litellm import completion

EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
ADDITIONAL_INSTRUCTIONS_HEADER = "## Additional Instructions"


def _sum_usage(*usages):
    total = dict(EMPTY_USAGE)
    for usage in usages:
        for key in total:
            total[key] += (usage or {}).get(key, 0)
    return total


def _call_llm(model_name, prompt):
    """
    Both callers of this (rule generation, instruction consolidation) only
    need a short bullet-list response, never a long document. model_name is
    any LiteLLM-supported model string -- provider is resolved automatically.

    num_retries=3: LiteLLM retries transient failures (rate limits, timeouts)
    up to 3 times automatically -- only because we pass this explicitly.
    """
    response = completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        num_retries=3,
        max_tokens=4096,
    )
    usage = getattr(response, "usage", None)
    tokens_used = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    } if usage is not None else dict(EMPTY_USAGE)
    return response.choices[0].message.content.strip(), tokens_used


def generate_additional_rules(current_version, results, eval_model,
                               score_threshold, optimizer_instructions):
    """
    Analyses every scored task and asks for NEW prompt instructions that
    address recurring weaknesses. Results are split into "strengths" (score
    >= score_threshold) and "weaknesses" (score < score_threshold) -- the
    optimizer is told what's already working (so it doesn't undo it) and
    only asked to fix what's failing repeatedly. If nothing is failing, no
    LLM call is made at all.
    """
    if len(results) == 0:
        return "", dict(EMPTY_USAGE)

    weaknesses = [r for r in results if (r[2] or 0) < score_threshold]
    strengths = [r for r in results if (r[2] or 0) >= score_threshold]

    if len(weaknesses) == 0:
        return "", dict(EMPTY_USAGE)

    def _format(rows):
        block = ""
        for task_name, _, score, feedback in rows:
            block += f"\nTask: {task_name}\nScore: {score}\nFeedback: {feedback}\n"
        return block

    weaknesses_block = _format(weaknesses)
    strengths_block = _format(strengths) if strengths else "(none -- every task scored below threshold)"

    analysis_prompt = f"""
{optimizer_instructions}

Prompt Version: {current_version}

Tasks that scored WELL (>= {score_threshold}) -- treat these as things the
current prompt already gets right. Do NOT suggest changes that would
undermine whatever is working here:
{strengths_block}

Tasks that scored POORLY (< {score_threshold}) -- look for weaknesses here:
{weaknesses_block}
"""

    rules, tokens_used = _call_llm(eval_model, analysis_prompt)

    if rules.upper() == "NONE":
        return "", tokens_used

    return rules, tokens_used


def _split_additional_instructions(prompt_text):
    """
    Splits `prompt_text` into (base_prompt, existing_additional_rules).
    existing_additional_rules is "" if the prompt has no
    "## Additional Instructions" section yet.
    """
    marker_index = prompt_text.find(ADDITIONAL_INSTRUCTIONS_HEADER)
    if marker_index == -1:
        return prompt_text, ""
    base_prompt = prompt_text[:marker_index].rstrip()
    existing_rules = prompt_text[marker_index + len(ADDITIONAL_INSTRUCTIONS_HEADER):].strip()
    return base_prompt, existing_rules


def consolidate_instructions(existing_rules, new_rules, eval_model):
    """
    Merges `existing_rules` with `new_rules` into a single, compact,
    deduplicated bullet list capped at 10 bullets -- instead of naively
    concatenating them every optimization loop. If there's nothing to
    merge, no LLM call is made at all.
    """
    if not existing_rules:
        return new_rules, dict(EMPTY_USAGE)
    if not new_rules:
        return existing_rules, dict(EMPTY_USAGE)

    merge_prompt = f"""
You are a Prompt Optimization Expert whose job is to keep this instruction
set as SHORT as possible while losing no meaning.

Below are two sets of bullet-point instructions for the same prompt. Merge
them into ONE compact, deduplicated bullet list:

- Remove exact or near-duplicate instructions.
- Combine related or overlapping instructions into a single, more general
  bullet rather than listing them separately.
- If a new instruction is more specific than an existing one, keep only the
  more specific version.
- Keep every instruction short and generic.
- Cap the result at AT MOST 10 bullets total. If merging still leaves more
  than 10, drop the least impactful ones first.
- Preserve any instruction that is not clearly redundant.
- Do not explain anything, return ONLY the merged bullet list.

Existing Instructions:
{existing_rules}

New Instructions:
{new_rules}
"""
    merged, tokens_used = _call_llm(eval_model, merge_prompt)
    return merged, tokens_used


def build_next_prompt(current_prompt, current_version, results, eval_model,
                       score_threshold, optimizer_instructions):
    new_rules, gen_tokens = generate_additional_rules(
        current_version, results, eval_model, score_threshold, optimizer_instructions
    )

    if new_rules == "":
        return current_prompt, gen_tokens

    base_prompt, existing_rules = _split_additional_instructions(current_prompt)
    merged_rules, merge_tokens = consolidate_instructions(existing_rules, new_rules, eval_model)

    total_tokens = _sum_usage(gen_tokens, merge_tokens)

    next_prompt = base_prompt + "\n\n" + ADDITIONAL_INSTRUCTIONS_HEADER + "\n\n" + merged_rules

    print(f"    [Additional Instructions] previous: {len(existing_rules.splitlines())} lines / "
          f"{len(existing_rules.split())} words  ->  merged: {len(merged_rules.splitlines())} lines / "
          f"{len(merged_rules.split())} words")

    return next_prompt, total_tokens