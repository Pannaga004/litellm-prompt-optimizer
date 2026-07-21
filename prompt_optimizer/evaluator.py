import json
import re
from litellm import completion
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _extract_json_object(text: str) -> str:
    """
    Extract a single well-formed JSON object from `text`, tolerating:
      - markdown code fences (```json ... ``` or ``` ... ```)
      - leading prose/preamble before the JSON starts
      - trailing garbage after the JSON ends (e.g. a stray duplicated
        closing brace, or the model echoing part of a schema/example
        after finishing its real answer)
    """
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return text

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)
        return json.dumps(obj)
    except json.JSONDecodeError:
        return text

class LiteLLMJudge(DeepEvalBaseLLM):
    """
    DeepEval-compatible wrapper around LiteLLM. `model_name` can be any
    LiteLLM-supported model string, e.g.:
      "groq/llama-3.3-70b-versatile"
      "gemini/gemini-2.0-flash"
      "gpt-4o"
      "claude-3-5-sonnet-20241022"

    Retries: num_retries=3 is passed on every call, so LiteLLM automatically
    retries transient failures (rate limits, timeouts, brief API hiccups)
    up to 3 times before giving up -- this is NOT automatic by default in
    LiteLLM, it only happens because we pass this parameter explicitly.

    JSON handling: not every provider honors response_format the same way
    (Claude in particular often doesn't support it, or wraps JSON in
    markdown fences regardless). To handle this generically across all
    providers:
      1. Always add an explicit "respond with ONLY valid JSON" instruction
         to the prompt itself when schema validation is needed -- this
         works regardless of provider.
      2. Try response_format as a bonus hint; if the provider/model
         rejects that parameter, fall back to a plain call.
      3. Always strip markdown fences before parsing, since some models
         add them even when told not to.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.usage = dict(EMPTY_USAGE)

    def load_model(self):
        return self

    def reset_usage(self):
        self.usage = dict(EMPTY_USAGE)

    def get_usage(self):
        return dict(self.usage)

    def _track_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            self.usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

    def _call(self, prompt: str, json_mode: bool) -> str:
        final_prompt = prompt
        if json_mode:
            # Provider-agnostic nudge -- works even for providers (like Claude)
            # that don't support response_format at all.
            final_prompt = (
                prompt
                + "\n\nRespond with ONLY valid JSON. No markdown code fences, "
                  "no commentary, no preamble -- just the raw JSON object."
            )

        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0,
            num_retries=3,
            max_tokens=4096,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = completion(**kwargs)
        except Exception:
            if json_mode:
                # This model/provider doesn't support response_format
                # (e.g. Claude on some LiteLLM versions) -- retry without it,
                # relying on the prompt instruction + fence-stripping instead.
                kwargs.pop("response_format", None)
                response = completion(**kwargs)
            else:
                raise

        self._track_usage(response)
        text = response.choices[0].message.content
        return _extract_json_object(text) if json_mode else text

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            text = self._call(prompt, json_mode=True)
            return schema.model_validate_json(text)
        return self._call(prompt, json_mode=False)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


def _normalize_header_line(line):
    """
    Strips markdown/numbering/bullet decoration from a candidate header
    line so 'Database Design' matches whether it appears as
    '## Database Design', '**Database Design**', '3. Database Design',
    or 'Database Design:'.
    """
    line = line.strip()
    line = re.sub(r'^[#>\-\*\d\.\)\s]+', '', line)
    line = re.sub(r'[\*_`]+', '', line)
    line = line.strip().rstrip(':').strip()
    return line.lower()


def check_required_sections(response_text, required_sections):
    """
    Deterministic, code-level check for required-section presence and
    order -- runs regardless of what the LLM judge concludes, so a
    missing or misnamed section can't be talked past by a lenient judge.

    `required_sections` is a list of dicts:
      [{"name": "Database Tables", "synonyms": ["database tables",
        "database design", "data model"]}, ...]
    Synonym matching is fuzzy: a candidate header line counts as a match
    if it equals a synonym OR contains a synonym as a substring (so
    'Database Design (ERD)' still matches 'database design').

    Only short lines (<=70 chars) are treated as header candidates, to
    avoid false-matching body paragraphs that happen to mention a
    section name in passing.

    Returns:
      {
        "missing": [canonical names never found],
        "out_of_order": True/False,
        "found_sections_in_order": [names found, in the order required]
                                    if ordered, else None
      }
    """
    candidate_lines = []
    for line in response_text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) > 70:
            continue
        normalized = _normalize_header_line(stripped)
        if normalized:
            candidate_lines.append(normalized)

    positions = {}
    for section in required_sections:
        name = section["name"]
        synonyms = [s.lower() for s in section.get("synonyms", [name.lower()])]
        for idx, line in enumerate(candidate_lines):
            if any(line == syn or syn in line for syn in synonyms):
                positions[name] = idx
                break

    missing = [s["name"] for s in required_sections if s["name"] not in positions]
    found_sections = [s["name"] for s in required_sections if s["name"] in positions]
    found_positions = [positions[name] for name in found_sections]
    is_ordered = all(
        found_positions[i] < found_positions[i + 1] for i in range(len(found_positions) - 1)
    )

    return {
        "missing": missing,
        "out_of_order": not is_ordered,
        "found_sections_in_order": found_sections if is_ordered else None,
    }


def evaluate_response(task, response, eval_model, evaluation_steps, threshold,
                       max_response_chars=4000, required_sections=None,
                       section_penalty_per_missing=0.15,
                       section_penalty_for_order=0.05,
                       geval_weight=0.7, relevancy_weight=0.3):
    """
    Evaluates `response` with a LiteLLM-backed judge (any provider, chosen
    purely by the eval_model string).

    The final score is a WEIGHTED BLEND of two LLM-judged metrics:
      - GEval (domain-specific rubric from evaluation_steps) -- weighted
        `geval_weight`. This is the primary signal: it's the only metric
        that knows anything about SRS-specific rules (section presence,
        specificity to the named application, count caps, etc).
      - AnswerRelevancyMetric -- weighted `relevancy_weight`. Catches
        off-topic or fabricated statements that don't relate to the input
        query. It's a generic, domain-agnostic check, which is why it's
        weighted lower by default (0.3) rather than treated as equally
        authoritative as the custom rubric.
    geval_weight + relevancy_weight need not sum to exactly 1 -- they're
    applied as raw multipliers, so mismatched weights should be treated as
    a config mistake to fix, not a silent auto-normalization.

    On top of the blended score, if `required_sections` is provided, a
    DETERMINISTIC (non-LLM) check runs against the raw response text and
    subtracts a further penalty for each missing section and for
    out-of-order sections. This exists because an LLM judge can (and did,
    in testing) notice a missing/misnamed section, say so in its reasoning,
    and still hand out a high score anyway -- the deterministic check can't
    be talked out of penalizing it.

    max_response_chars caps how much of `response` is sent to the judge,
    since long generated documents can exceed a judge model's per-request
    token limit on their own.
    """
    judge_model = LiteLLMJudge(model_name=eval_model)
    judge_model.reset_usage()

    eval_response = response
    if max_response_chars and len(response) > max_response_chars:
        eval_response = (
            response[:max_response_chars]
            + "\n\n[...response truncated for evaluation due to judge model token limits...]"
        )

    test_case = LLMTestCase(input=task, actual_output=eval_response)

    geval_metric = GEval(
        name="Dynamic Evaluation",
        model=judge_model,
        evaluation_steps=evaluation_steps,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    )
    geval_metric.measure(test_case)

    # include_reason=False skips AnswerRelevancyMetric's 3rd unconditional
    # LLM call (statements -> verdicts -> reason). We keep statements+verdicts
    # (needed to compute the score itself, which now DOES feed the final
    # blended score) but drop the reason-writing call, since we only need
    # the number, not prose explaining it. This is a call-count reduction,
    # separate from and unrelated to retry-on-timeout behavior: retries on
    # transient failures (rate limits, timeouts) are still handled per-call
    # by num_retries=3 inside _call() above, for every remaining call this
    # metric makes.
    relevancy_metric = AnswerRelevancyMetric(
        threshold=threshold,
        model=judge_model,
        include_reason=False
    )
    relevancy_metric.measure(test_case)



    evaluation_tokens = judge_model.get_usage()

    raw_geval_score = geval_metric.score
    raw_relevancy_score = relevancy_metric.score
    blended_score = (geval_weight * raw_geval_score) + (relevancy_weight * raw_relevancy_score)
    final_score = blended_score
    section_feedback = ""

    if required_sections:
        section_result = check_required_sections(response, required_sections)
        num_missing = len(section_result["missing"])
        penalty = num_missing * section_penalty_per_missing
        if section_result["out_of_order"]:
            penalty += section_penalty_for_order

        if penalty > 0:
            final_score = max(0.0, blended_score - penalty)
            section_feedback = (
                f"\n\n[Deterministic Section Check -- code-level, not LLM-judged]\n"
                f"Missing sections: {section_result['missing'] or 'none'}\n"
                f"Out of order: {section_result['out_of_order']}\n"
                f"Penalty applied: -{penalty:.2f}  "
                f"(blended score: {blended_score:.2f} -> adjusted: {final_score:.2f})"
            )
            print(f"    [Section Check] missing={section_result['missing']}  "
                  f"out_of_order={section_result['out_of_order']}  penalty=-{penalty:.2f}  "
                  f"score {blended_score:.2f} -> {final_score:.2f}")

    print(f"    [GEval] score={raw_geval_score:.2f}  reason={geval_metric.reason}")
    print(f"    [Answer Relevancy] score={raw_relevancy_score:.2f}  "
          f"(reason generation skipped -- include_reason=False)")
    print(f"    [Blended] {geval_weight}*GEval + {relevancy_weight}*Relevancy = {blended_score:.2f}"
          + (f"  ->  after section penalty: {final_score:.2f}" if final_score != blended_score else ""))
    print(f"    [Judge tokens] prompt={evaluation_tokens['prompt_tokens']}  "
          f"completion={evaluation_tokens['completion_tokens']}  total={evaluation_tokens['total_tokens']}")

    combined_feedback = (
        f"GEval Score: {raw_geval_score}\nGEval Reason: {geval_metric.reason}\n\n"
        f"Answer Relevancy Score: {raw_relevancy_score}\n\n"
        f"Blended Score ({geval_weight}*GEval + {relevancy_weight}*Relevancy): {blended_score:.2f}"
        f"{section_feedback}"
    )

    return final_score, combined_feedback, evaluation_tokens