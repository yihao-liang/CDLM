import json
import os
import re
import sys
import glob
import ast
import traceback
from typing import Dict, List, Optional, Set, Tuple
import evaluate
import argparse

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def refine_text(text: str) -> str:
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > 100:
        lines = lines[:100]
    max_valid_lines = 0
    max_valid_snippet = ""
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i : j + 1])
            if syntax_check(current_snippet):
                valid_line_count = sum(1 for line in lines[i : j + 1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    max_valid_snippet = current_snippet
    return max_valid_snippet


def get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    pass
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]
    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)
    return visited


def get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    elif isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(node))


def sanitize(text: str, entrypoint: Optional[str] = None) -> str:
    text = refine_text(text)
    try:
        code = extract_longest_valid_code(text)
        if not code:
            return ""
        tree = ast.parse(code)
    except (SyntaxError, MemoryError):
        return ""
    definitions = {}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            name = node.name
            definitions[name] = ("class", node)
        elif isinstance(node, ast.FunctionDef):
            name = node.name
            definitions[name] = ("function", node)
        elif isinstance(node, ast.Assign):
            name = get_definition_name(node)
            if name:
                definitions[name] = ("variable", node)
    if entrypoint and entrypoint in definitions:
        name2deps = get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = get_function_dependency(entrypoint, name2deps)
    else:
        reachable = set(definitions.keys())
    sanitized_output = []
    for node in imports:
        sanitized_output.append(ast.unparse(node))
    for name, (_, node) in definitions.items():
        if name in reachable:
            sanitized_output.append(ast.unparse(node))
    return "\n".join(sanitized_output)


def extract_python_code(text: str) -> str:
    """Extract Python code from text.

    Handles multiple formats:
    1. ```python ... ``` markdown code blocks
    2. Direct code (starts with def/import/class)
    3. Code ending with [END] marker
    """
    # First try markdown code block
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ```\n...\n``` (without language specifier)
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Handle direct code output (no markdown wrapper)
    # Remove [END] marker and trailing special tokens
    text = re.sub(r'\[END\].*$', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|[^|]+\|>.*$', '', text, flags=re.DOTALL)  # Remove <|endoftext|> etc.
    text = text.strip()

    # Check if it looks like Python code (starts with def, import, class, or common patterns)
    if text and re.match(r'^(def\s|import\s|from\s|class\s|#|@|\w+\s*=)', text):
        return text

    return ""


def find_entry_point(code_block: str) -> Optional[str]:
    if not code_block:
        return None
    match = re.search(r"^\s*def\s+([a-zA-Z_]\w*)", code_block, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def evaluate_mbpp_results(directory, k_values=None):
    """
    Evaluate MBPP results with pass@k metrics.

    Args:
        directory: Path to evaluation results directory
        k_values: List of k values for pass@k (default: [1])
    """
    if k_values is None:
        k_values = [1]

    print("\n" + "="*50 + f"\nProcessing MBPP directory: {directory}\n" + "="*50)
    print(f"Computing pass@k for k={k_values}")

    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in directory '{directory}'.")
        return

    all_predictions, all_references = [], []
    agg_stats = {
        "processed": 0, "failed_extraction": 0, "failed_entry_point": 0
    }

    print(f"Found {len(jsonl_files)} files to process...")
    for file_path in jsonl_files:
        print(f"  -> Processing file: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    resps = item['resps'][0]  # List of responses
                    reference_tests = item['target']

                    # Handle both single response (string) and multiple responses (list)
                    if isinstance(resps, str):
                        raw_generations = [resps]
                    elif isinstance(resps, list):
                        # resps could be [[gen1, gen2, ...]] or [gen1]
                        if len(resps) > 0 and isinstance(resps[0], list):
                            raw_generations = resps[0]  # [[gen1, gen2]] -> [gen1, gen2]
                        else:
                            raw_generations = resps  # [gen1] or [gen1, gen2, ...]
                    else:
                        raw_generations = [resps]

                    # Process each candidate
                    candidates = []
                    valid_sample = False
                    for raw_gen in raw_generations:
                        if isinstance(raw_gen, str):
                            code_to_process = extract_python_code(raw_gen)
                            if code_to_process:
                                entry_point = find_entry_point(code_to_process)
                                if entry_point:
                                    sanitized_code = sanitize(code_to_process, entrypoint=entry_point)
                                    candidates.append(sanitized_code)
                                    valid_sample = True
                                else:
                                    candidates.append("")  # Empty for failed entry point
                            else:
                                candidates.append("")  # Empty for failed extraction
                        else:
                            candidates.append("")

                    # Always add to predictions (failed extractions become empty strings -> will fail tests)
                    all_predictions.append(candidates)
                    all_references.append(reference_tests)
                    agg_stats["processed"] += 1

                    # Track failure stats for reporting
                    if not valid_sample and len(raw_generations) > 0:
                        code = extract_python_code(raw_generations[0]) if isinstance(raw_generations[0], str) else ""
                        if not code:
                            agg_stats["failed_extraction"] += 1
                        else:
                            agg_stats["failed_entry_point"] += 1

                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"    Error processing file '{os.path.basename(file_path)}': {e}")
                    continue

    total_samples = agg_stats["processed"]

    if total_samples > 0:
        # Determine actual num_samples from predictions
        num_samples_per_prompt = max(len(p) for p in all_predictions) if all_predictions else 1
        print(f"\nDetected {num_samples_per_prompt} samples per prompt")

        # Filter k values to only those <= num_samples
        valid_k_values = [k for k in k_values if k <= num_samples_per_prompt]
        if not valid_k_values:
            valid_k_values = [1]

        print(f"Loading the code_eval evaluator and starting evaluation...")
        code_eval = evaluate.load("code_eval")
        # Use num_workers=4 to avoid multiprocessing issues in some environments
        pass_at_k_results, _ = code_eval.compute(
            references=all_references,
            predictions=all_predictions,
            k=valid_k_values,
            num_workers=4
        )

        # Calculate pass@k on valid extractions only
        valid_predictions = []
        valid_references = []
        for preds, refs in zip(all_predictions, all_references):
            # Check if at least one prediction is non-empty (valid extraction)
            if any(p.strip() for p in preds):
                valid_predictions.append(preds)
                valid_references.append(refs)

        valid_pass_at_k_results = {}
        if valid_predictions:
            valid_pass_at_k_results, _ = code_eval.compute(
                references=valid_references,
                predictions=valid_predictions,
                k=valid_k_values,
                num_workers=4
            )

        num_valid = len(valid_predictions)
        num_failed = total_samples - num_valid

        print("\n" + "-" * 80)
        print(f"Results for '{os.path.basename(directory)}'")
        print(f"  - Total samples:          {total_samples}")
        print(f"  - Valid extractions:      {num_valid}")
        print(f"  - Failed extractions:     {num_failed}")
        print(f"    - Failed code extract:  {agg_stats['failed_extraction']}")
        print(f"    - Failed entry point:   {agg_stats['failed_entry_point']}")
        print(f"  - Samples per prompt:     {num_samples_per_prompt}")
        print("-" * 40)
        print("  [All Samples]")
        for k in valid_k_values:
            score = pass_at_k_results.get(f"pass@{k}", 0.0)
            print(f"    - pass@{k}:              {score*100:.2f}%")
        print("-" * 40)
        print(f"  [Valid Extractions Only] (n={num_valid})")
        for k in valid_k_values:
            score = valid_pass_at_k_results.get(f"pass@{k}", 0.0)
            print(f"    - pass@{k}:              {score*100:.2f}%")
    else:
        print("No valid data processed. Cannot calculate results.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--res_path",
        type=str,
        required=True)
    parser.add_argument(
        "-k", "--k_values",
        type=str,
        default="1",
        help="Comma-separated k values for pass@k (e.g., '1,5' for pass@1 and pass@5)")
    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    evaluate_mbpp_results(directory=args.res_path, k_values=k_values)