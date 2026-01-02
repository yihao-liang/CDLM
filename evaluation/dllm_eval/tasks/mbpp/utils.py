import re
from typing import Union

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def pass_at_k_metric(
    references: Union[str, list[str]],
    predictions: Union[str, list[list[str]]],
    k: int = 5
) -> float:
    """
    Compute pass@k metric for code generation

    Args:
        references: Test cases for each problem
        predictions: Generated code candidates (list of lists)
        k: Number of candidates to consider

    Returns:
        pass@k accuracy
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]

    result = pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[k],
    )[0]

    return result[f"pass@{k}"]


def pass_at_5(
    references: Union[str, list[str]],
    predictions: Union[str, list[list[str]]]
) -> float:
    """
    Compute pass@5 metric for code generation
    Fixed k=5 version for lm-eval-harness compatibility

    Args:
        references: Test cases for each problem
        predictions: Generated code candidates (list of lists, each with 5 candidates)
                    Can be raw text (will extract code blocks) or already extracted code

    Returns:
        pass@5 accuracy
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]

    # Extract code blocks from predictions if they contain markdown
    extracted_predictions = []
    for pred_list in predictions:
        extracted = [extract_code_blocks(p) if isinstance(p, str) else p for p in pred_list]
        extracted_predictions.append(extracted)

    # Determine actual number of samples per prompt
    num_samples = max(len(p) for p in extracted_predictions) if extracted_predictions else 1
    k_value = min(5, num_samples)  # Can't compute pass@k if k > num_samples

    result = pass_at_k.compute(
        references=references,
        predictions=extracted_predictions,
        k=[k_value],
    )[0]

    return result[f"pass@{k_value}"]


def extract_code_blocks(text: str) -> str:
    # Pattern to match ```...``` blocks
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    # (+ ```) as we add the opening "```python" to the gen_prefix
    matches = re.findall(pattern, r"```" + text, re.DOTALL)
    # if no matches, try to match ```...``` blocks (after removing the language)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = re.findall(pattern, text_without_lang, re.DOTALL)
    if not matches:
        # If no markdown code blocks found, return the original text
        # This handles cases where the model generates raw Python code without markdown formatting
        return text
    else:
        return matches[0]


def build_predictions(resps: list, docs: list[dict]) -> list[list[str]]:
    """
    Build predictions by extracting code blocks from responses.

    Args:
        resps: Generated responses, can be:
               - list[str]: single response per prompt
               - list[list[str]]: multiple responses per prompt
               - list[list[list[str]]]: nested structure from some generators
        docs: Document metadata (unused but required by filter interface)

    Returns:
        list[list[str]]: Extracted code blocks
    """
    results = []
    for resp in resps:
        # Handle nested list structure: [[gen1, gen2]] -> [gen1, gen2]
        if isinstance(resp, list) and len(resp) > 0 and isinstance(resp[0], list):
            resp = resp[0]

        # Handle single string response
        if isinstance(resp, str):
            resp = [resp]

        # Extract code blocks from each response
        extracted = []
        for r in resp:
            if isinstance(r, str):
                extracted.append(extract_code_blocks(r))
            elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], str):
                # Handle extra nesting: [['code']] -> 'code'
                extracted.append(extract_code_blocks(r[0]))
            else:
                extracted.append("")
        results.append(extracted)

    return results


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "prompt": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "prompt": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "prompt": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
