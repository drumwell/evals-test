import json
from pathlib import Path
from typing import List

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval

# In newer DeepEval docs: LLMTestCaseParams is in deepeval.metrics.
# Some versions expose it under deepeval.test_case. Try both gracefully.
try:
    from deepeval.metrics import LLMTestCaseParams  # preferred (matches docs)
except Exception:
    from deepeval.test_case import LLMTestCaseParams  # fallback

from openai import OpenAI
client = OpenAI()

GEN_MODEL   = "gpt-4o-mini"  # model that generates answers
JUDGE_MODEL = "gpt-4o-mini"  # model that judges correctness

DATA_PATH = "data/tiny_eval_dataset.jsonl"  # {"input":..., "ideal":...} per line

def gen(prompt: str) -> str:
    r = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content.strip()

def load_rows(path: str) -> List[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def main():
    rows = load_rows(DATA_PATH)

    test_cases: List[LLMTestCase] = []
    for r in rows:
        inp   = r["input"]
        ideal = r["ideal"]
        out   = gen(inp)
        test_cases.append(
            LLMTestCase(
                input=inp,
                expected_output=ideal,
                actual_output=out,
            )
        )

    # GEval-based Correctness (no RAG context required)
    correctness = GEval(
        name="Correctness",
        model=JUDGE_MODEL,
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        # keep it simple first; tighten later to match your domain
        evaluation_steps=[
            "Decide whether the actual output has the same meaning as the expected output.",
            "Ignore trivial differences in casing and punctuation.",
        ],
        threshold=0.5,      # >= threshold => pass
        # Newer versions may support these; older ones ignore/omit them:
        # strict=False,
        # async_mode=False,
    )

    evaluate(test_cases, [correctness])

if __name__ == "__main__":
    main()
