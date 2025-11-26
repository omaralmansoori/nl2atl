"""
Evaluation Module

Provides evaluation helpers for NL-ATL datasets:
- Basic statistics computation
- Spot-checking examples with LLM critique
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from atl_syntax import validate_atl_string, extract_components
from nl2atl import critique_nl_atl_pair, get_llm_client, LLMClient


def load_dataset(dataset_path: str | Path) -> list[dict]:
    """
    Load a dataset from JSONL file.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        
    Returns:
        List of dataset entries as dictionaries
    """
    path = Path(dataset_path)
    entries = []

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    return entries


def compute_basic_stats(dataset_path: str | Path) -> dict:
    """
    Compute basic statistics for an NL-ATL dataset.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        Dictionary with statistics:
        - total_pairs: number of pairs
        - syntax_valid_count: number of syntactically valid ATL formulas
        - syntax_valid_fraction: fraction of valid formulas
        - operator_distribution: counts of G/F/X/U operators
        - coalition_size_distribution: distribution of coalition sizes
        - avg_nl_length: average NL sentence length in words
        - template_distribution: counts per template ID
        - critique_stats: statistics from LLM critiques (if available)
    """
    entries = load_dataset(dataset_path)

    if not entries:
        return {"error": "Empty dataset"}

    # Basic counts
    total = len(entries)
    valid_count = 0
    operator_counts = Counter()
    coalition_sizes = Counter()
    template_counts = Counter()
    nl_lengths = []
    critique_ok_count = 0
    critique_total = 0

    for entry in entries:
        atl = entry.get("atl_formula", "")
        nl = entry.get("nl_text", "")

        # Validate ATL
        is_valid, _ = validate_atl_string(atl)
        if is_valid:
            valid_count += 1

            # Extract operators
            try:
                components = extract_components(atl)
                for op in components.get("operators", []):
                    operator_counts[op] += 1
            except Exception:
                pass

        # Coalition size
        coalition = entry.get("coalition", "")
        if coalition:
            size = len(coalition.split(","))
            coalition_sizes[size] += 1

        # Template
        template_id = entry.get("template_id", "unknown")
        template_counts[template_id] += 1

        # NL length
        nl_lengths.append(len(nl.split()))

        # Critique stats
        if "critique_ok" in entry:
            critique_total += 1
            if entry["critique_ok"]:
                critique_ok_count += 1

    stats = {
        "total_pairs": total,
        "syntax_valid_count": valid_count,
        "syntax_valid_fraction": valid_count / total if total > 0 else 0,
        "operator_distribution": dict(operator_counts),
        "coalition_size_distribution": dict(coalition_sizes),
        "avg_nl_length": sum(nl_lengths) / len(nl_lengths) if nl_lengths else 0,
        "template_distribution": dict(template_counts),
    }

    if critique_total > 0:
        stats["critique_stats"] = {
            "total_critiqued": critique_total,
            "passed_critique": critique_ok_count,
            "pass_rate": critique_ok_count / critique_total,
        }

    return stats


def spot_check_examples(
    dataset_path: str | Path,
    k: int = 5,
    client: Optional[LLMClient] = None,
    provider: str = "openai",
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Randomly sample k examples and run LLM critique on each.
    
    Args:
        dataset_path: Path to the dataset file
        k: Number of examples to check
        client: Optional pre-configured LLM client
        provider: LLM provider to use
        seed: Random seed for reproducibility
        
    Returns:
        List of critique results for each sampled example
    """
    entries = load_dataset(dataset_path)

    if seed is not None:
        random.seed(seed)

    # Sample k examples
    samples = random.sample(entries, min(k, len(entries)))

    results = []

    for entry in samples:
        nl = entry.get("nl_text", "")
        atl = entry.get("atl_formula", "")

        result = {
            "nl_text": nl,
            "atl_formula": atl,
            "template_id": entry.get("template_id", ""),
        }

        # Run critique
        try:
            if client is None:
                client = get_llm_client(provider)

            critique = critique_nl_atl_pair(nl, atl, client=client)
            result["critique"] = critique
            result["status"] = "ok" if critique.get("ok") else "issues_found"

        except Exception as e:
            result["critique"] = None
            result["status"] = "error"
            result["error"] = str(e)

        results.append(result)

    return results


def print_spot_check_report(results: list[dict]) -> None:
    """
    Print a formatted report of spot-check results.
    
    Args:
        results: Results from spot_check_examples
    """
    print("=" * 70)
    print("SPOT CHECK REPORT")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\n--- Example {i} ---")
        print(f"NL: {result['nl_text']}")
        print(f"ATL: {result['atl_formula']}")

        if result["status"] == "error":
            print(f"Status: ERROR - {result.get('error', 'Unknown error')}")
        elif result["status"] == "ok":
            print("Status: ✓ PASS")
            critique = result.get("critique", {})
            if critique.get("explanation"):
                print(f"Explanation: {critique['explanation']}")
        else:
            print("Status: ✗ ISSUES FOUND")
            critique = result.get("critique", {})
            if critique.get("issues"):
                print(f"Issues: {critique['issues']}")
            if critique.get("suggested_fix"):
                print(f"Suggested fix: {critique['suggested_fix']}")

    # Summary
    print("\n" + "=" * 70)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] == "error")
    issues_count = sum(1 for r in results if r["status"] == "issues_found")

    print(f"SUMMARY: {len(results)} examples checked")
    print(f"  ✓ Passed: {ok_count}")
    print(f"  ✗ Issues: {issues_count}")
    print(f"  ⚠ Errors: {error_count}")
    print("=" * 70)


def print_stats_report(stats: dict) -> None:
    """
    Print a formatted statistics report.
    
    Args:
        stats: Statistics from compute_basic_stats
    """
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print(f"\nTotal pairs: {stats.get('total_pairs', 0)}")
    print(
        f"Syntax valid: {stats.get('syntax_valid_count', 0)} "
        f"({stats.get('syntax_valid_fraction', 0):.1%})"
    )
    print(f"Avg NL length: {stats.get('avg_nl_length', 0):.1f} words")

    print("\nOperator Distribution:")
    for op, count in sorted(stats.get("operator_distribution", {}).items()):
        print(f"  {op}: {count}")

    print("\nCoalition Size Distribution:")
    for size, count in sorted(stats.get("coalition_size_distribution", {}).items()):
        print(f"  Size {size}: {count}")

    print("\nTemplate Distribution:")
    for template, count in sorted(stats.get("template_distribution", {}).items()):
        print(f"  {template}: {count}")

    if "critique_stats" in stats:
        cs = stats["critique_stats"]
        print("\nCritique Statistics:")
        print(f"  Total critiqued: {cs['total_critiqued']}")
        print(f"  Passed: {cs['passed_critique']} ({cs['pass_rate']:.1%})")

    print("=" * 70)


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate NL-ATL datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Compute basic statistics")
    stats_parser.add_argument("dataset", help="Path to dataset file")
    stats_parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of formatted text"
    )

    # Spot-check command
    check_parser = subparsers.add_parser("spot-check", help="Spot-check random examples")
    check_parser.add_argument("dataset", help="Path to dataset file")
    check_parser.add_argument(
        "-k", type=int, default=5, help="Number of examples to check"
    )
    check_parser.add_argument("--provider", default="openai", help="LLM provider")
    check_parser.add_argument("--seed", type=int, help="Random seed")
    check_parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of formatted text"
    )

    args = parser.parse_args()

    if args.command == "stats":
        stats = compute_basic_stats(args.dataset)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print_stats_report(stats)

    elif args.command == "spot-check":
        try:
            results = spot_check_examples(
                args.dataset,
                k=args.k,
                provider=args.provider,
                seed=args.seed,
            )
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_spot_check_report(results)
        except ValueError as e:
            if "API key" in str(e):
                print(f"Error: {e}")
                print("Set the appropriate API key environment variable to use spot-check.")
            else:
                raise

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
