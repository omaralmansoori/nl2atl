#!/usr/bin/env python3
"""
Evaluation Module for NL-ATL Translation Quality Assessment.

This module provides comprehensive evaluation utilities for assessing the quality
of NL-ATL translation datasets. It includes statistical analysis, spot-checking
capabilities, and reporting functions.

Architecture Overview:
----------------------
The evaluation system operates in three layers:

1. Statistical Analysis Layer
   - compute_basic_stats(): Dataset-level statistics
   - compute_quality_metrics(): Per-example quality scoring
   - analyze_distribution(): Coverage and diversity metrics

2. Spot-Check Layer
   - spot_check_examples(): Random sample selection
   - evaluate_example(): Single example assessment
   - SpotCheckResult: Structured evaluation outcome

3. Reporting Layer
   - print_stats_report(): Human-readable statistics
   - print_spot_check_report(): Detailed sample review
   - export_evaluation(): JSON/YAML export

Example Usage:
--------------
    # Basic statistics on a dataset
    from evaluation import compute_basic_stats, print_stats_report
    
    with open("dataset.jsonl") as f:
        pairs = [json.loads(line) for line in f]
    
    stats = compute_basic_stats(pairs)
    print_stats_report(stats)
    
    # Spot-check random samples
    from evaluation import spot_check_examples, print_spot_check_report
    
    samples = spot_check_examples(pairs, n=5)
    print_spot_check_report(samples)

CLI Usage:
----------
    # Compute statistics
    python evaluation.py stats dataset.jsonl
    
    # Spot-check with specific count
    python evaluation.py spot-check dataset.jsonl --count 10
    
    # Full evaluation report
    python evaluation.py full-report dataset.jsonl --output report.json

Integration Points:
-------------------
- atl_syntax: For formula validation and structure analysis
- dataset_gen: Can evaluate generated datasets directly
- nl2atl: Can assess translation quality on test sets

Author: NL2ATL Project
Version: 1.0.0
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import click

# Local imports
from atl_syntax import (
    ATLFormula,
    Coalition,
    Not,
    And,
    Or,
    Implies,
    TemporalOp,
    parse_atl,
    is_valid as _is_valid,
    get_coalition_info,
)


def is_valid(formula: str) -> bool:
    """Check if an ATL formula is valid (returns bool)."""
    result = _is_valid(formula)
    return result.valid


def get_formula_depth(formula: ATLFormula) -> int:
    """Get the depth of a formula using its .depth() method."""
    return formula.depth()


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DatasetStats:
    """
    Comprehensive statistics for an NL-ATL dataset.
    
    This class captures all relevant metrics about a dataset including
    size, validity rates, coverage, and structural distribution.
    
    Attributes:
        total_pairs: Total number of NL-ATL pairs in dataset
        valid_atl_count: Number of pairs with valid ATL formulas
        invalid_atl_count: Number of pairs with invalid ATL formulas
        validity_rate: Proportion of valid ATL formulas (0.0 to 1.0)
        avg_nl_length: Average number of words in NL descriptions
        avg_atl_length: Average character length of ATL formulas
        nl_length_std: Standard deviation of NL word counts
        atl_length_std: Standard deviation of ATL character lengths
        depth_distribution: Distribution of formula depths {depth: count}
        operator_distribution: Distribution of temporal operators {op: count}
        coalition_size_distribution: Distribution of coalition sizes {size: count}
        unique_atoms: Set of unique atomic propositions across dataset
        template_distribution: Distribution of templates if available {template: count}
        source_distribution: Distribution of sources {source: count}
        timestamp: When statistics were computed
    
    Example:
        stats = compute_basic_stats(pairs)
        print(f"Validity: {stats.validity_rate:.1%}")
        print(f"Operators: {stats.operator_distribution}")
    """
    
    # Size metrics
    total_pairs: int = 0
    valid_atl_count: int = 0
    invalid_atl_count: int = 0
    validity_rate: float = 0.0
    
    # Length metrics
    avg_nl_length: float = 0.0
    avg_atl_length: float = 0.0
    nl_length_std: float = 0.0
    atl_length_std: float = 0.0
    min_nl_length: int = 0
    max_nl_length: int = 0
    min_atl_length: int = 0
    max_atl_length: int = 0
    
    # Structural distribution
    depth_distribution: dict[int, int] = field(default_factory=dict)
    operator_distribution: dict[str, int] = field(default_factory=dict)
    coalition_size_distribution: dict[int, int] = field(default_factory=dict)
    
    # Coverage metrics
    unique_atoms: set[str] = field(default_factory=set)
    unique_coalitions: set[str] = field(default_factory=set)
    
    # Source tracking
    template_distribution: dict[str, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert stats to dictionary for serialization.
        
        Returns:
            Dictionary representation of all statistics.
            
        Example:
            stats_dict = stats.to_dict()
            json.dump(stats_dict, open("stats.json", "w"))
        """
        return {
            "total_pairs": self.total_pairs,
            "valid_atl_count": self.valid_atl_count,
            "invalid_atl_count": self.invalid_atl_count,
            "validity_rate": self.validity_rate,
            "avg_nl_length": self.avg_nl_length,
            "avg_atl_length": self.avg_atl_length,
            "nl_length_std": self.nl_length_std,
            "atl_length_std": self.atl_length_std,
            "min_nl_length": self.min_nl_length,
            "max_nl_length": self.max_nl_length,
            "min_atl_length": self.min_atl_length,
            "max_atl_length": self.max_atl_length,
            "depth_distribution": self.depth_distribution,
            "operator_distribution": self.operator_distribution,
            "coalition_size_distribution": self.coalition_size_distribution,
            "unique_atoms_count": len(self.unique_atoms),
            "unique_atoms_sample": list(self.unique_atoms)[:20],
            "unique_coalitions_count": len(self.unique_coalitions),
            "unique_coalitions_sample": list(self.unique_coalitions)[:10],
            "template_distribution": self.template_distribution,
            "source_distribution": self.source_distribution,
            "timestamp": self.timestamp,
        }


@dataclass
class SpotCheckResult:
    """
    Result of spot-checking a single NL-ATL pair.
    
    Captures detailed information about an example for manual review
    or automated quality assessment.
    
    Attributes:
        nl: The natural language description
        atl: The ATL formula string
        is_valid: Whether the ATL formula parses correctly
        parsed_formula: The parsed ATLFormula object (if valid)
        formula_depth: Depth of the formula structure
        coalition_info: Information about coalitions in the formula
        operators_used: List of temporal operators in the formula
        source: Source/provenance of this example
        template_id: Template ID if generated from template
        notes: Any additional notes or observations
        quality_score: Optional quality score (0.0 to 1.0)
    
    Example:
        result = evaluate_example({"nl": "...", "atl": "..."})
        if result.is_valid:
            print(f"Depth: {result.formula_depth}")
    """
    
    nl: str
    atl: str
    is_valid: bool
    parsed_formula: ATLFormula | None = None
    formula_depth: int = 0
    coalition_info: list[dict[str, Any]] = field(default_factory=list)
    operators_used: list[str] = field(default_factory=list)
    source: str = "unknown"
    template_id: str | None = None
    notes: str = ""
    quality_score: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nl": self.nl,
            "atl": self.atl,
            "is_valid": self.is_valid,
            "formula_depth": self.formula_depth,
            "coalition_info": self.coalition_info,
            "operators_used": self.operators_used,
            "source": self.source,
            "template_id": self.template_id,
            "notes": self.notes,
            "quality_score": self.quality_score,
        }


# =============================================================================
# Statistical Analysis Functions
# =============================================================================


def compute_basic_stats(pairs: list[dict[str, Any]]) -> DatasetStats:
    """
    Compute comprehensive statistics for an NL-ATL dataset.
    
    This function analyzes a dataset of NL-ATL pairs and computes various
    statistical metrics including validity rates, length distributions,
    structural properties, and coverage metrics.
    
    Algorithm:
    1. Count valid/invalid ATL formulas
    2. Compute length statistics for NL and ATL
    3. Analyze formula structure (depth, operators, coalitions)
    4. Track coverage (atoms, coalitions, templates)
    5. Aggregate source information
    
    Args:
        pairs: List of dictionaries with 'nl' and 'atl' keys.
               Optional keys: 'source', 'template_id', 'metadata'
    
    Returns:
        DatasetStats object containing all computed statistics.
    
    Raises:
        ValueError: If pairs list is empty.
    
    Example:
        >>> pairs = [
        ...     {"nl": "Agent can reach goal", "atl": "<<agent>>F goal"},
        ...     {"nl": "System is always safe", "atl": "<<>>G safe"},
        ... ]
        >>> stats = compute_basic_stats(pairs)
        >>> print(f"Total: {stats.total_pairs}, Valid: {stats.validity_rate:.1%}")
        Total: 2, Valid: 100.0%
    
    Performance:
        O(n * m) where n is number of pairs and m is average formula size.
        Parsing is the main bottleneck for large datasets.
    """
    if not pairs:
        raise ValueError("Cannot compute statistics on empty dataset")
    
    stats = DatasetStats()
    stats.total_pairs = len(pairs)
    
    # Collect raw measurements
    nl_lengths: list[int] = []
    atl_lengths: list[int] = []
    valid_formulas: list[ATLFormula] = []
    
    for pair in pairs:
        nl = pair.get("nl", "")
        atl = pair.get("atl", "")
        
        # NL length (word count)
        nl_word_count = len(nl.split())
        nl_lengths.append(nl_word_count)
        
        # ATL length (character count)
        atl_lengths.append(len(atl))
        
        # Track source and template
        source = pair.get("source", "unknown")
        stats.source_distribution[source] = stats.source_distribution.get(source, 0) + 1
        
        template_id = pair.get("template_id")
        if template_id:
            stats.template_distribution[template_id] = (
                stats.template_distribution.get(template_id, 0) + 1
            )
        
        # Validate and analyze ATL
        try:
            if is_valid(atl):
                stats.valid_atl_count += 1
                formula = parse_atl(atl)
                if formula:
                    valid_formulas.append(formula)
                    
                    # Track atoms and coalitions
                    _collect_atoms(formula, stats.unique_atoms)
                    _collect_coalitions(formula, stats.unique_coalitions)
            else:
                stats.invalid_atl_count += 1
        except Exception:
            stats.invalid_atl_count += 1
    
    # Compute validity rate
    stats.validity_rate = stats.valid_atl_count / stats.total_pairs if stats.total_pairs > 0 else 0.0
    
    # Compute length statistics
    if nl_lengths:
        stats.avg_nl_length = statistics.mean(nl_lengths)
        stats.nl_length_std = statistics.stdev(nl_lengths) if len(nl_lengths) > 1 else 0.0
        stats.min_nl_length = min(nl_lengths)
        stats.max_nl_length = max(nl_lengths)
    
    if atl_lengths:
        stats.avg_atl_length = statistics.mean(atl_lengths)
        stats.atl_length_std = statistics.stdev(atl_lengths) if len(atl_lengths) > 1 else 0.0
        stats.min_atl_length = min(atl_lengths)
        stats.max_atl_length = max(atl_lengths)
    
    # Analyze formula structures
    for formula in valid_formulas:
        # Depth distribution
        depth = get_formula_depth(formula)
        stats.depth_distribution[depth] = stats.depth_distribution.get(depth, 0) + 1
        
        # Operator distribution
        operators = _collect_operators(formula)
        for op in operators:
            stats.operator_distribution[op] = stats.operator_distribution.get(op, 0) + 1
        
        # Coalition size distribution
        coalition_infos = get_coalition_info(formula)
        for info in coalition_infos:
            size = info.get("size", 0)
            stats.coalition_size_distribution[size] = (
                stats.coalition_size_distribution.get(size, 0) + 1
            )
    
    return stats


def _collect_atoms(formula: ATLFormula, atoms: set[str]) -> None:
    """
    Recursively collect all atomic propositions from a formula.
    
    Args:
        formula: The ATL formula to analyze
        atoms: Set to add discovered atoms to (modified in place)
    """
    from atl_syntax import Atom
    
    if isinstance(formula, Atom):
        atoms.add(formula.name)
    elif isinstance(formula, Not):
        _collect_atoms(formula.operand, atoms)
    elif isinstance(formula, (And, Or, Implies)):
        _collect_atoms(formula.left, atoms)
        _collect_atoms(formula.right, atoms)
    elif isinstance(formula, Coalition):
        _collect_atoms(formula.formula, atoms)
    elif isinstance(formula, TemporalOp):
        _collect_atoms(formula.operand, atoms)
        if formula.operand2:
            _collect_atoms(formula.operand2, atoms)


def _collect_coalitions(formula: ATLFormula, coalitions: set[str]) -> None:
    """
    Recursively collect all coalitions from a formula.
    
    Args:
        formula: The ATL formula to analyze
        coalitions: Set to add discovered coalitions to (modified in place)
    """
    if isinstance(formula, Coalition):
        coalition_str = ",".join(sorted(formula.agents))
        coalitions.add(coalition_str)
        _collect_coalitions(formula.formula, coalitions)
    elif isinstance(formula, Not):
        _collect_coalitions(formula.operand, coalitions)
    elif isinstance(formula, (And, Or, Implies)):
        _collect_coalitions(formula.left, coalitions)
        _collect_coalitions(formula.right, coalitions)
    elif isinstance(formula, TemporalOp):
        _collect_coalitions(formula.operand, coalitions)
        if formula.operand2:
            _collect_coalitions(formula.operand2, coalitions)


def _collect_operators(formula: ATLFormula) -> list[str]:
    """
    Recursively collect all operators from a formula.
    
    Returns:
        List of operator symbols found in the formula.
    """
    operators: list[str] = []
    
    if isinstance(formula, Not):
        operators.append("¬")
        operators.extend(_collect_operators(formula.operand))
    elif isinstance(formula, And):
        operators.append("∧")
        operators.extend(_collect_operators(formula.left))
        operators.extend(_collect_operators(formula.right))
    elif isinstance(formula, Or):
        operators.append("∨")
        operators.extend(_collect_operators(formula.left))
        operators.extend(_collect_operators(formula.right))
    elif isinstance(formula, Implies):
        operators.append("→")
        operators.extend(_collect_operators(formula.left))
        operators.extend(_collect_operators(formula.right))
    elif isinstance(formula, Coalition):
        operators.append("⟨⟨⟩⟩")
        operators.extend(_collect_operators(formula.formula))
    elif isinstance(formula, TemporalOp):
        operators.append(formula.op)
        operators.extend(_collect_operators(formula.operand))
        if formula.operand2:
            operators.extend(_collect_operators(formula.operand2))
    
    return operators


def compute_quality_metrics(
    pairs: list[dict[str, Any]],
    check_consistency: bool = True,
) -> dict[str, float]:
    """
    Compute quality metrics for a dataset.
    
    This provides higher-level quality indicators beyond basic statistics,
    including diversity measures and consistency checks.
    
    Args:
        pairs: List of NL-ATL pairs
        check_consistency: Whether to check NL-ATL consistency patterns
    
    Returns:
        Dictionary of quality metrics:
        - diversity_score: How diverse the dataset is (0.0 to 1.0)
        - validity_rate: Proportion of valid ATL formulas
        - coverage_score: How well operators/structures are covered
        - avg_formula_complexity: Average formula depth/operators
    
    Example:
        >>> metrics = compute_quality_metrics(pairs)
        >>> print(f"Diversity: {metrics['diversity_score']:.2f}")
    """
    if not pairs:
        return {
            "diversity_score": 0.0,
            "validity_rate": 0.0,
            "coverage_score": 0.0,
            "avg_formula_complexity": 0.0,
        }
    
    stats = compute_basic_stats(pairs)
    
    # Diversity: based on unique templates/sources and structural variety
    unique_templates = len(stats.template_distribution)
    unique_sources = len(stats.source_distribution)
    unique_depths = len(stats.depth_distribution)
    unique_operators = len(stats.operator_distribution)
    
    # Normalize diversity score
    max_diversity = 10  # Expected reasonable maximum
    raw_diversity = unique_templates + unique_sources + unique_depths + unique_operators
    diversity_score = min(1.0, raw_diversity / (max_diversity * 4))
    
    # Coverage: based on operators and coalition sizes
    expected_operators = {"X", "F", "G", "U", "∧", "∨", "¬", "→", "⟨⟨⟩⟩"}
    covered_operators = set(stats.operator_distribution.keys())
    coverage_score = len(covered_operators & expected_operators) / len(expected_operators)
    
    # Complexity: average depth weighted by operator count
    if stats.depth_distribution:
        avg_depth = sum(d * c for d, c in stats.depth_distribution.items()) / sum(stats.depth_distribution.values())
        avg_ops = sum(stats.operator_distribution.values()) / max(1, stats.valid_atl_count)
        avg_complexity = (avg_depth + avg_ops) / 2
    else:
        avg_complexity = 0.0
    
    return {
        "diversity_score": diversity_score,
        "validity_rate": stats.validity_rate,
        "coverage_score": coverage_score,
        "avg_formula_complexity": avg_complexity,
        "unique_atoms": len(stats.unique_atoms),
        "unique_coalitions": len(stats.unique_coalitions),
    }


# =============================================================================
# Spot-Check Functions
# =============================================================================


def spot_check_examples(
    pairs: list[dict[str, Any]],
    n: int = 5,
    seed: int | None = None,
    stratified: bool = False,
) -> list[SpotCheckResult]:
    """
    Select and evaluate random samples from the dataset for manual review.
    
    This function supports both random sampling and stratified sampling
    (by source or template) for comprehensive dataset review.
    
    Algorithm:
    1. If stratified, group by source/template and sample proportionally
    2. Otherwise, simple random sample
    3. Evaluate each selected example
    4. Return structured results
    
    Args:
        pairs: List of NL-ATL pairs to sample from
        n: Number of examples to select (default: 5)
        seed: Random seed for reproducibility (optional)
        stratified: Whether to stratify by source/template
    
    Returns:
        List of SpotCheckResult objects for each selected example.
    
    Example:
        >>> samples = spot_check_examples(pairs, n=10, seed=42)
        >>> for s in samples:
        ...     print(f"NL: {s.nl[:50]}...")
        ...     print(f"ATL: {s.atl}")
        ...     print(f"Valid: {s.is_valid}")
        ...     print()
    
    Note:
        Setting a seed ensures reproducible sampling for consistent evaluation.
    """
    if not pairs:
        return []
    
    if seed is not None:
        random.seed(seed)
    
    # Ensure we don't sample more than available
    n = min(n, len(pairs))
    
    if stratified:
        # Group by source
        by_source: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            source = pair.get("source", "unknown")
            by_source.setdefault(source, []).append(pair)
        
        # Proportional sampling
        selected: list[dict[str, Any]] = []
        remaining = n
        sources = list(by_source.keys())
        
        for i, source in enumerate(sources):
            source_pairs = by_source[source]
            # Last source gets remaining, others get proportional share
            if i == len(sources) - 1:
                count = remaining
            else:
                proportion = len(source_pairs) / len(pairs)
                count = max(1, int(n * proportion))
            count = min(count, len(source_pairs), remaining)
            selected.extend(random.sample(source_pairs, count))
            remaining -= count
            if remaining <= 0:
                break
    else:
        selected = random.sample(pairs, n)
    
    # Evaluate each selected example
    results = [evaluate_example(pair) for pair in selected]
    
    return results


def evaluate_example(pair: dict[str, Any]) -> SpotCheckResult:
    """
    Perform detailed evaluation of a single NL-ATL pair.
    
    This function parses the ATL formula (if valid), extracts structural
    information, and computes a basic quality score.
    
    Args:
        pair: Dictionary with 'nl' and 'atl' keys, plus optional metadata.
    
    Returns:
        SpotCheckResult with detailed evaluation information.
    
    Example:
        >>> result = evaluate_example({
        ...     "nl": "The robot can always reach the goal",
        ...     "atl": "<<robot>>G F goal",
        ...     "source": "template"
        ... })
        >>> print(f"Valid: {result.is_valid}, Depth: {result.formula_depth}")
    """
    nl = pair.get("nl", "")
    atl = pair.get("atl", "")
    source = pair.get("source", "unknown")
    template_id = pair.get("template_id")
    
    result = SpotCheckResult(
        nl=nl,
        atl=atl,
        is_valid=False,
        source=source,
        template_id=template_id,
    )
    
    # Try to parse and analyze
    try:
        if is_valid(atl):
            result.is_valid = True
            formula = parse_atl(atl)
            
            if formula:
                result.parsed_formula = formula
                result.formula_depth = get_formula_depth(formula)
                result.coalition_info = get_coalition_info(formula)
                result.operators_used = _collect_operators(formula)
                
                # Compute basic quality score
                result.quality_score = _compute_example_quality(nl, atl, formula)
        else:
            result.notes = "ATL formula failed validation"
    except Exception as e:
        result.notes = f"Error during evaluation: {e}"
    
    return result


def _compute_example_quality(nl: str, atl: str, formula: ATLFormula) -> float:
    """
    Compute a quality score for a single NL-ATL pair.
    
    Heuristic scoring based on:
    - NL completeness (word count)
    - ATL validity (already checked)
    - Structural reasonableness
    
    Returns:
        Quality score between 0.0 and 1.0
    """
    score = 0.0
    
    # NL quality (prefer 5-30 words)
    word_count = len(nl.split())
    if 5 <= word_count <= 30:
        score += 0.3
    elif word_count > 0:
        score += 0.1
    
    # ATL validity (already known to be valid)
    score += 0.3
    
    # Reasonable depth (prefer 1-4)
    depth = get_formula_depth(formula)
    if 1 <= depth <= 4:
        score += 0.2
    elif depth > 0:
        score += 0.1
    
    # Has coalition modality
    if isinstance(formula, Coalition):
        score += 0.2
    else:
        # Might be a sub-formula case
        score += 0.1
    
    return min(1.0, score)


# =============================================================================
# Reporting Functions
# =============================================================================


def print_stats_report(stats: DatasetStats, verbose: bool = False) -> None:
    """
    Print a human-readable statistics report to stdout.
    
    Args:
        stats: DatasetStats object to report on
        verbose: Whether to include detailed distributions
    
    Example:
        >>> stats = compute_basic_stats(pairs)
        >>> print_stats_report(stats, verbose=True)
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS REPORT")
    print("=" * 60)
    
    print(f"\n📊 Size Metrics:")
    print(f"   Total pairs:      {stats.total_pairs:,}")
    print(f"   Valid ATL:        {stats.valid_atl_count:,}")
    print(f"   Invalid ATL:      {stats.invalid_atl_count:,}")
    print(f"   Validity rate:    {stats.validity_rate:.1%}")
    
    print(f"\n📏 Length Statistics:")
    print(f"   Avg NL length:    {stats.avg_nl_length:.1f} words (±{stats.nl_length_std:.1f})")
    print(f"   NL range:         [{stats.min_nl_length}, {stats.max_nl_length}] words")
    print(f"   Avg ATL length:   {stats.avg_atl_length:.1f} chars (±{stats.atl_length_std:.1f})")
    print(f"   ATL range:        [{stats.min_atl_length}, {stats.max_atl_length}] chars")
    
    print(f"\n🔢 Coverage Metrics:")
    print(f"   Unique atoms:     {len(stats.unique_atoms)}")
    print(f"   Unique coalitions: {len(stats.unique_coalitions)}")
    
    if verbose:
        print(f"\n📈 Depth Distribution:")
        for depth, count in sorted(stats.depth_distribution.items()):
            bar = "█" * min(count, 50)
            print(f"   Depth {depth}: {count:4d} {bar}")
        
        print(f"\n🔧 Operator Distribution:")
        for op, count in sorted(stats.operator_distribution.items(), key=lambda x: -x[1]):
            bar = "█" * min(count // 10, 50)
            print(f"   {op:4s}: {count:4d} {bar}")
        
        print(f"\n👥 Coalition Size Distribution:")
        for size, count in sorted(stats.coalition_size_distribution.items()):
            bar = "█" * min(count // 5, 50)
            print(f"   Size {size}: {count:4d} {bar}")
        
        if stats.source_distribution:
            print(f"\n📁 Source Distribution:")
            for source, count in sorted(stats.source_distribution.items(), key=lambda x: -x[1]):
                pct = count / stats.total_pairs * 100
                print(f"   {source}: {count} ({pct:.1f}%)")
    
    print(f"\n⏰ Computed: {stats.timestamp}")
    print("=" * 60 + "\n")


def print_spot_check_report(results: list[SpotCheckResult]) -> None:
    """
    Print a detailed spot-check report for manual review.
    
    Args:
        results: List of SpotCheckResult objects from spot_check_examples()
    
    Example:
        >>> samples = spot_check_examples(pairs, n=5)
        >>> print_spot_check_report(samples)
    """
    print("\n" + "=" * 60)
    print("SPOT-CHECK REPORT")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    
    valid_count = sum(1 for r in results if r.is_valid)
    print(f"Valid: {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'─' * 60}")
        print(f"Example {i}")
        print(f"{'─' * 60}")
        print(f"NL:     {result.nl}")
        print(f"ATL:    {result.atl}")
        print(f"Valid:  {'✓' if result.is_valid else '✗'}")
        
        if result.is_valid:
            print(f"Depth:  {result.formula_depth}")
            print(f"Ops:    {', '.join(result.operators_used[:5])}")
            if result.coalition_info:
                coalitions = [c.get("agents", []) for c in result.coalition_info]
                print(f"Coalitions: {coalitions}")
            if result.quality_score is not None:
                print(f"Quality: {result.quality_score:.2f}")
        
        if result.notes:
            print(f"Notes:  {result.notes}")
        
        print(f"Source: {result.source}")
        if result.template_id:
            print(f"Template: {result.template_id}")
    
    print("\n" + "=" * 60 + "\n")


def export_evaluation(
    stats: DatasetStats,
    spot_checks: list[SpotCheckResult] | None = None,
    output_path: Path | str | None = None,
    format: str = "json",
) -> dict[str, Any] | str:
    """
    Export evaluation results to file or return as dict/string.
    
    Args:
        stats: Dataset statistics
        spot_checks: Optional spot-check results
        output_path: Optional file path to write to
        format: Output format ("json" or "yaml")
    
    Returns:
        If output_path is None, returns dict (json) or string (yaml).
        Otherwise, writes to file and returns the path.
    
    Example:
        >>> export_evaluation(stats, spot_checks, "report.json")
    """
    report = {
        "statistics": stats.to_dict(),
        "spot_checks": [r.to_dict() for r in (spot_checks or [])],
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "format_version": "1.0",
        },
    }
    
    if format == "yaml":
        try:
            import yaml
            content = yaml.dump(report, default_flow_style=False, allow_unicode=True)
        except ImportError:
            # Fall back to JSON
            content = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        content = json.dumps(report, indent=2, ensure_ascii=False)
    
    if output_path:
        path = Path(output_path)
        path.write_text(content, encoding="utf-8")
        return str(path)
    
    return report if format == "json" else content


# =============================================================================
# CLI Interface
# =============================================================================


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    NL-ATL Dataset Evaluation Tool.
    
    Provides commands for analyzing and spot-checking NL-ATL datasets.
    
    Examples:
    
        # Compute basic statistics
        evaluation.py stats dataset.jsonl
        
        # Spot-check with specific count
        evaluation.py spot-check dataset.jsonl --count 10
        
        # Full evaluation report
        evaluation.py full-report dataset.jsonl --output report.json
    """
    pass


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed distributions")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
def stats(dataset_path: str, verbose: bool, output: str | None):
    """
    Compute and display dataset statistics.
    
    DATASET_PATH: Path to JSONL file containing NL-ATL pairs.
    """
    # Load dataset
    pairs = _load_dataset(dataset_path)
    click.echo(f"Loaded {len(pairs)} pairs from {dataset_path}")
    
    # Compute stats
    dataset_stats = compute_basic_stats(pairs)
    
    # Print report
    print_stats_report(dataset_stats, verbose=verbose)
    
    # Optionally export
    if output:
        export_evaluation(dataset_stats, output_path=output)
        click.echo(f"Statistics exported to {output}")


@cli.command("spot-check")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--count", "-n", default=5, help="Number of examples to check")
@click.option("--seed", "-s", type=int, help="Random seed for reproducibility")
@click.option("--stratified", is_flag=True, help="Stratify by source")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
def spot_check(
    dataset_path: str,
    count: int,
    seed: int | None,
    stratified: bool,
    output: str | None,
):
    """
    Spot-check random samples from the dataset.
    
    DATASET_PATH: Path to JSONL file containing NL-ATL pairs.
    """
    # Load dataset
    pairs = _load_dataset(dataset_path)
    click.echo(f"Loaded {len(pairs)} pairs from {dataset_path}")
    
    # Spot-check
    results = spot_check_examples(pairs, n=count, seed=seed, stratified=stratified)
    
    # Print report
    print_spot_check_report(results)
    
    # Optionally export
    if output:
        export_evaluation(DatasetStats(), results, output_path=output)
        click.echo(f"Spot-check results exported to {output}")


@cli.command("full-report")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--spot-count", "-n", default=10, help="Number of spot-check examples")
@click.option("--seed", "-s", type=int, help="Random seed")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def full_report(
    dataset_path: str,
    spot_count: int,
    seed: int | None,
    output: str | None,
    verbose: bool,
):
    """
    Generate a comprehensive evaluation report.
    
    Includes both statistics and spot-check results.
    
    DATASET_PATH: Path to JSONL file containing NL-ATL pairs.
    """
    # Load dataset
    pairs = _load_dataset(dataset_path)
    click.echo(f"Loaded {len(pairs)} pairs from {dataset_path}")
    
    # Compute statistics
    dataset_stats = compute_basic_stats(pairs)
    print_stats_report(dataset_stats, verbose=verbose)
    
    # Quality metrics
    quality = compute_quality_metrics(pairs)
    click.echo("\n📈 Quality Metrics:")
    for key, value in quality.items():
        click.echo(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")
    
    # Spot-check
    results = spot_check_examples(pairs, n=spot_count, seed=seed)
    print_spot_check_report(results)
    
    # Export if requested
    if output:
        export_evaluation(dataset_stats, results, output_path=output)
        click.echo(f"\n📁 Full report exported to {output}")


@cli.command("quality")
@click.argument("dataset_path", type=click.Path(exists=True))
def quality(dataset_path: str):
    """
    Compute quality metrics for the dataset.
    
    DATASET_PATH: Path to JSONL file containing NL-ATL pairs.
    """
    pairs = _load_dataset(dataset_path)
    click.echo(f"Loaded {len(pairs)} pairs from {dataset_path}")
    
    metrics = compute_quality_metrics(pairs)
    
    click.echo("\n📈 Quality Metrics:")
    click.echo("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            click.echo(f"   {key:25s}: {value:.3f}")
        else:
            click.echo(f"   {key:25s}: {value}")


def _load_dataset(path: str) -> list[dict[str, Any]]:
    """Load dataset from JSONL or JSON file."""
    filepath = Path(path)
    
    if filepath.suffix == ".jsonl":
        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs
    elif filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle both list format and {"pairs": [...]} format
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "pairs" in data:
            return data["pairs"]
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


# =============================================================================
# Module Entry Point
# =============================================================================


if __name__ == "__main__":
    cli()
