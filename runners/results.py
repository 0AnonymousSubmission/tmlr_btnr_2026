# type: ignore
import os
import json
from typing import List


def get_result_filepath(output_dir: str, run_id: str) -> str:
    return os.path.join(output_dir, f"{run_id}.json")


def check_completed(output_dir: str, run_id: str) -> tuple:
    result_file = get_result_filepath(output_dir, run_id)
    if not os.path.exists(result_file):
        return False, False, False, None
    try:
        with open(result_file, "r") as f:
            result = json.load(f)
        return (
            True,
            result.get("success", False),
            result.get("singular", False),
            result.get("error"),
        )
    except:
        return False, False, False, None


def save_result(output_dir: str, run_id: str, result: dict):
    with open(get_result_filepath(output_dir, run_id), "w") as f:
        json.dump(result, f, indent=2)


def print_result(result: dict):
    if result.get("singular"):
        print("SINGULAR")
    elif result.get("success"):
        print(f"Test Q={result['test_quality']:.4f}")
    else:
        print(f"FAILED: {result.get('error', 'Unknown')[:50]}")


def summarize_results(results: List[dict]) -> dict:
    import numpy as np
    successful = [r for r in results if r.get("success") and not r.get("singular")]
    qualities = [r["test_quality"] for r in successful] if successful else []
    return {
        "total": len(results),
        "successful": len(successful),
        "singular": len(results) - len(successful),
        "mean": float(np.mean(qualities)) if qualities else 0.0,
        "std": float(np.std(qualities)) if qualities else 0.0,
    }
