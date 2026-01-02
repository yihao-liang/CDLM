import re
from typing import Union

import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]

    # Determine actual number of samples per prompt
    num_samples = max(len(p) for p in predictions) if predictions else 1
    # Filter k values to only those <= num_samples
    valid_k = [kv for kv in k if kv <= num_samples]
    if not valid_k:
        valid_k = [min(k)]  # Use smallest k if none are valid

    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=valid_k
    )
    return res[0]


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return compute_.compute(
        references=references,
        predictions=predictions,
        k=[1],
        num_workers=48
    )[0]["pass@1"]


def extract_code_blocks(text: str) -> str:
    text = re.sub(r"\[DONE\]", "", text)
    text = re.sub(r"<\|eot_id\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    return text


def clean_response_string(r: str) -> str:
    cleaned_text = r if r.rfind("```python") == -1 else r[r.rfind("```python"):]
    cleaned_text = cleaned_text if cleaned_text.rfind("```") == -1 else cleaned_text[: cleaned_text.rfind("```")]
    cleaned_text = cleaned_text if cleaned_text.rfind("if __name__ == \"__main__\":") == -1 else cleaned_text[: cleaned_text.rfind("if __name__ == \"__main__\":")]
    return cleaned_text

    
def build_predictions(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """
    Build predictions from model responses.

    Handles two cases:
    - num_return_sequences=1: resp is [str, str, ...] -> each r is a string
    - num_return_sequences>1: resp is [[str, str, ...], ...] -> each r is a list of strings
    """
    result = []
    for resp, doc in zip(resps, docs):
        cleaned = []
        for r in resp:
            if isinstance(r, list):
                # num_return_sequences > 1: r is a list of strings
                cleaned.extend([clean_response_string(s) for s in r])
            else:
                # num_return_sequences = 1: r is a string
                cleaned.append(clean_response_string(r))
        result.append(cleaned)
    return result
