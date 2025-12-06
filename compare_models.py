#!/usr/bin/env python3
"""
Model Comparison and Evaluation Script
======================================

This script tests the fine-tuned NL2ATL model against previously rejected
samples and compares its performance with the original model results.

Features:
- Load rejected samples from experiment data
- Re-translate using the fine-tuned model
- Re-verify translations using the verification pipeline
- Generate comparison statistics and reports
- Identify improvements and regressions

Usage Examples
--------------
# Run comparison on all rejected samples
python compare_models.py run --rejected data/experiment_20251201_200910_rejected.json

# Run on a subset with custom model
python compare_models.py run --rejected data/rejected.json --limit 50

# Generate comparison report from existing results
python compare_models.py report --results data/comparison_results.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from dotenv import load_dotenv

# Import local modules
from atl_syntax import is_valid, validate_atl_string

# Load environment variables
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:nl2atl-v1:CiGKGvnC"
BASE_MODEL = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = (
    "You are an expert in translating natural language requirements into "
    "Alternating-time Temporal Logic (ATL) formulas. Given a natural language "
    "statement describing agent capabilities and temporal properties, generate "
    "the corresponding ATL formula using proper syntax with coalition operators "
    "⟨⟨...⟩⟩, temporal operators (G, F, X, U), and logical operators (∧, ∨, →, ¬)."
)

VERIFIER_PROMPT = """You are an expert in Alternating-time Temporal Logic (ATL) verification.

Given a natural language requirement and its ATL formula translation, verify if the translation is correct.

Evaluation criteria:
1. **Agent Coalition**: Does the formula include the correct agents in the coalition ⟨⟨...⟩⟩?
2. **Temporal Operators**: Are the temporal operators (G, F, X, U) used correctly to capture the timing requirements?
3. **Logical Structure**: Does the formula correctly represent the logical relationships (implications, conjunctions, disjunctions)?
4. **Semantic Accuracy**: Does the formula capture the full meaning of the NL statement?

Respond with a JSON object:
{
    "verdict": "ACCEPT" or "REJECT",
    "confidence": 0.0-1.0,
    "issues": ["list of issues if rejected"],
    "notes": "brief explanation"
}
"""


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing model outputs for a single sample."""
    sample_id: str
    nl_statement: str
    domain: str
    agents: List[str]
    
    # Original (rejected) result
    original_atl: str
    original_generator: str
    original_rejection_reasons: List[str]
    
    # Fine-tuned model result
    finetuned_atl: str
    finetuned_syntax_valid: bool
    finetuned_syntax_errors: List[str]
    
    # Verification of fine-tuned result
    finetuned_verdict: Optional[str] = None
    finetuned_confidence: Optional[float] = None
    finetuned_issues: Optional[List[str]] = None
    finetuned_notes: Optional[str] = None
    
    # Comparison outcome
    improved: bool = False
    status: str = "pending"  # pending, improved, same, regressed, error
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    

# =============================================================================
# Model Interaction
# =============================================================================

def get_openai_client():
    """Get OpenAI client instance."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai package: pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return OpenAI(api_key=api_key)


def translate_with_model(
    client,
    nl_statement: str,
    model: str,
    domain: Optional[str] = None,
    agents: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Translate NL to ATL using specified model.
    
    Returns:
        Tuple of (atl_formula, usage_dict)
    """
    user_content = f"Translate the following natural language requirement to ATL:\n\n{nl_statement}"
    
    if domain:
        user_content += f"\n\nDomain: {domain}"
    
    if agents:
        user_content += f"\n\nAgents: {', '.join(agents)}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    
    atl_formula = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    
    return atl_formula, usage


def verify_translation(
    client,
    nl_statement: str,
    atl_formula: str,
    domain: Optional[str] = None,
    verifier_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Verify an NL-ATL translation using LLM.
    
    Returns:
        Dictionary with verdict, confidence, issues, notes
    """
    user_content = f"""Natural Language Requirement:
{nl_statement}

ATL Formula:
{atl_formula}
"""
    
    if domain:
        user_content += f"\nDomain: {domain}"
    
    messages = [
        {"role": "system", "content": VERIFIER_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    response = client.chat.completions.create(
        model=verifier_model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "verdict": result.get("verdict", "UNKNOWN"),
            "confidence": result.get("confidence", 0.0),
            "issues": result.get("issues", []),
            "notes": result.get("notes", ""),
        }
    except json.JSONDecodeError:
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "issues": ["Failed to parse verifier response"],
            "notes": response.choices[0].message.content,
        }


# =============================================================================
# Data Loading
# =============================================================================

def load_rejected_samples(filepath: Path) -> List[Dict[str, Any]]:
    """Load rejected samples from experiment file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    
    # Filter to only rejected samples (non-empty ones)
    rejected = []
    for sample in samples:
        if sample and sample.get('nl_statement') and sample.get('atl_formula'):
            rejected.append(sample)
    
    return rejected


def load_verified_samples(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """Load verified samples as a lookup dictionary by ID."""
    samples = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                samples[sample['id']] = sample
    return samples


# =============================================================================
# Comparison Logic
# =============================================================================

def compare_sample(
    client,
    sample: Dict[str, Any],
    finetuned_model: str,
    verify: bool = True,
    verifier_model: str = "gpt-4o-mini",
) -> ComparisonResult:
    """
    Compare fine-tuned model output against original rejected sample.
    """
    # Extract sample info
    sample_id = sample.get('id', 'unknown')
    nl_statement = sample['nl_statement']
    domain = sample.get('domain', '')
    agents = sample.get('agents', [])
    original_atl = sample.get('atl_formula', '')
    original_generator = sample.get('atl_generator', 'unknown')
    rejection_reasons = sample.get('rejection_reasons', [])
    
    # Translate with fine-tuned model
    finetuned_atl, _ = translate_with_model(
        client, nl_statement, finetuned_model, domain, agents
    )
    
    # Validate syntax
    syntax_valid, syntax_errors = validate_atl_string(finetuned_atl)
    
    result = ComparisonResult(
        sample_id=sample_id,
        nl_statement=nl_statement,
        domain=domain,
        agents=agents,
        original_atl=original_atl,
        original_generator=original_generator,
        original_rejection_reasons=rejection_reasons,
        finetuned_atl=finetuned_atl,
        finetuned_syntax_valid=syntax_valid,
        finetuned_syntax_errors=syntax_errors,
    )
    
    # Verify if requested and syntax is valid
    if verify and syntax_valid:
        verification = verify_translation(
            client, nl_statement, finetuned_atl, domain, verifier_model
        )
        result.finetuned_verdict = verification['verdict']
        result.finetuned_confidence = verification['confidence']
        result.finetuned_issues = verification['issues']
        result.finetuned_notes = verification['notes']
        
        # Determine improvement status
        if verification['verdict'] == 'ACCEPT':
            result.improved = True
            result.status = 'improved'
        else:
            result.status = 'same'
    elif not syntax_valid:
        result.status = 'syntax_error'
    else:
        result.status = 'pending'
    
    return result


def run_comparison(
    rejected_samples: List[Dict[str, Any]],
    finetuned_model: str = FINETUNED_MODEL,
    verify: bool = True,
    verifier_model: str = "gpt-4o-mini",
    limit: Optional[int] = None,
    progress_callback=None,
) -> List[ComparisonResult]:
    """
    Run comparison on all rejected samples.
    """
    client = get_openai_client()
    results = []
    
    samples_to_process = rejected_samples[:limit] if limit else rejected_samples
    total = len(samples_to_process)
    
    for i, sample in enumerate(samples_to_process):
        try:
            result = compare_sample(
                client, sample, finetuned_model, verify, verifier_model
            )
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, result)
                
        except Exception as e:
            # Create error result
            result = ComparisonResult(
                sample_id=sample.get('id', 'unknown'),
                nl_statement=sample.get('nl_statement', ''),
                domain=sample.get('domain', ''),
                agents=sample.get('agents', []),
                original_atl=sample.get('atl_formula', ''),
                original_generator=sample.get('atl_generator', 'unknown'),
                original_rejection_reasons=sample.get('rejection_reasons', []),
                finetuned_atl='',
                finetuned_syntax_valid=False,
                finetuned_syntax_errors=[str(e)],
                status='error',
            )
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, result)
        
        # Rate limiting
        time.sleep(0.5)
    
    return results


def generate_statistics(results: List[ComparisonResult]) -> Dict[str, Any]:
    """Generate summary statistics from comparison results."""
    total = len(results)
    
    if total == 0:
        return {"total": 0}
    
    improved = sum(1 for r in results if r.status == 'improved')
    same = sum(1 for r in results if r.status == 'same')
    syntax_errors = sum(1 for r in results if r.status == 'syntax_error')
    errors = sum(1 for r in results if r.status == 'error')
    pending = sum(1 for r in results if r.status == 'pending')
    
    syntax_valid = sum(1 for r in results if r.finetuned_syntax_valid)
    
    # Verdicts for verified samples
    accepted = sum(1 for r in results if r.finetuned_verdict == 'ACCEPT')
    rejected = sum(1 for r in results if r.finetuned_verdict == 'REJECT')
    
    # Average confidence for accepted
    accepted_confidences = [r.finetuned_confidence for r in results 
                           if r.finetuned_verdict == 'ACCEPT' and r.finetuned_confidence]
    avg_confidence = sum(accepted_confidences) / len(accepted_confidences) if accepted_confidences else 0
    
    return {
        "total": total,
        "improved": improved,
        "improvement_rate": improved / total * 100 if total > 0 else 0,
        "same": same,
        "syntax_errors": syntax_errors,
        "errors": errors,
        "pending": pending,
        "syntax_valid": syntax_valid,
        "syntax_valid_rate": syntax_valid / total * 100 if total > 0 else 0,
        "verification": {
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / (accepted + rejected) * 100 if (accepted + rejected) > 0 else 0,
            "avg_confidence": avg_confidence,
        },
    }


# =============================================================================
# CLI Commands
# =============================================================================

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Model Comparison and Evaluation CLI."""
    pass


@cli.command()
@click.option(
    '--rejected', '-r',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to rejected samples JSON file.',
)
@click.option(
    '--model', '-m',
    type=str,
    default=FINETUNED_MODEL,
    help=f'Fine-tuned model to test (default: {FINETUNED_MODEL}).',
)
@click.option(
    '--verifier', '-v',
    type=str,
    default='gpt-4o-mini',
    help='Model to use for verification.',
)
@click.option(
    '--limit', '-n',
    type=int,
    default=None,
    help='Limit number of samples to process.',
)
@click.option(
    '--no-verify',
    is_flag=True,
    help='Skip verification step (only check syntax).',
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    default=None,
    help='Output file for results (JSONL format).',
)
@click.option(
    '--report', '-R',
    type=click.Path(path_type=Path),
    default=None,
    help='Output file for summary report (JSON format).',
)
def run(
    rejected: Path,
    model: str,
    verifier: str,
    limit: Optional[int],
    no_verify: bool,
    output: Optional[Path],
    report: Optional[Path],
):
    """
    Run comparison test on rejected samples.
    
    This command:
    1. Loads previously rejected NL-ATL pairs
    2. Re-translates using the fine-tuned model
    3. Validates syntax and optionally re-verifies
    4. Generates comparison statistics
    """
    click.echo(f"\n{'='*60}")
    click.echo("NL2ATL Model Comparison Test")
    click.echo(f"{'='*60}\n")
    
    click.echo(f"Fine-tuned Model: {model}")
    click.echo(f"Verifier Model: {verifier}")
    click.echo(f"Verification: {'Disabled' if no_verify else 'Enabled'}")
    
    # Load samples
    click.echo(f"\nLoading rejected samples from: {rejected}")
    samples = load_rejected_samples(rejected)
    click.echo(f"Found {len(samples)} rejected samples")
    
    if limit:
        click.echo(f"Processing first {limit} samples")
    
    # Progress callback
    def progress(current, total, result):
        status_emoji = {
            'improved': '✅',
            'same': '➖',
            'syntax_error': '❌',
            'error': '⚠️',
            'pending': '⏳',
        }
        emoji = status_emoji.get(result.status, '?')
        click.echo(f"[{current}/{total}] {emoji} {result.sample_id}: {result.status}")
    
    # Run comparison
    click.echo(f"\n{'='*60}")
    click.echo("Running Comparison")
    click.echo(f"{'='*60}\n")
    
    results = run_comparison(
        samples,
        finetuned_model=model,
        verify=not no_verify,
        verifier_model=verifier,
        limit=limit,
        progress_callback=progress,
    )
    
    # Generate statistics
    stats = generate_statistics(results)
    
    # Display summary
    click.echo(f"\n{'='*60}")
    click.echo("Summary")
    click.echo(f"{'='*60}\n")
    
    click.echo(f"Total Samples: {stats['total']}")
    click.echo(f"Syntax Valid: {stats['syntax_valid']} ({stats['syntax_valid_rate']:.1f}%)")
    click.echo(f"\nVerification Results:")
    click.echo(f"  ✅ Improved (Accepted): {stats['improved']} ({stats['improvement_rate']:.1f}%)")
    click.echo(f"  ➖ Still Rejected: {stats['same']}")
    click.echo(f"  ❌ Syntax Errors: {stats['syntax_errors']}")
    click.echo(f"  ⚠️  Processing Errors: {stats['errors']}")
    
    if stats['verification']['accepted'] > 0:
        click.echo(f"\nAverage Confidence (Accepted): {stats['verification']['avg_confidence']:.2f}")
    
    # Save results
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
        click.echo(f"\nResults saved to: {output}")
    
    # Save report
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "verifier": verifier,
            "source_file": str(rejected),
            "statistics": stats,
            "sample_results": [asdict(r) for r in results],
        }
        with open(report, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        click.echo(f"Report saved to: {report}")


@cli.command()
@click.option(
    '--results', '-r',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to comparison results JSONL file.',
)
def report(results: Path):
    """Generate a detailed report from comparison results."""
    click.echo(f"\nLoading results from: {results}")
    
    # Load results
    comparison_results = []
    with open(results, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                # Convert dict back to dataclass-like structure
                comparison_results.append(data)
    
    stats = {
        "total": len(comparison_results),
        "improved": sum(1 for r in comparison_results if r.get('status') == 'improved'),
        "same": sum(1 for r in comparison_results if r.get('status') == 'same'),
        "syntax_errors": sum(1 for r in comparison_results if r.get('status') == 'syntax_error'),
        "errors": sum(1 for r in comparison_results if r.get('status') == 'error'),
    }
    
    stats["improvement_rate"] = stats["improved"] / stats["total"] * 100 if stats["total"] > 0 else 0
    
    click.echo(f"\n{'='*60}")
    click.echo("Comparison Report")
    click.echo(f"{'='*60}\n")
    
    click.echo(f"Total Samples: {stats['total']}")
    click.echo(f"✅ Improved: {stats['improved']} ({stats['improvement_rate']:.1f}%)")
    click.echo(f"➖ Same: {stats['same']}")
    click.echo(f"❌ Syntax Errors: {stats['syntax_errors']}")
    click.echo(f"⚠️  Errors: {stats['errors']}")
    
    # Show improved samples
    improved = [r for r in comparison_results if r.get('status') == 'improved']
    if improved:
        click.echo(f"\n{'='*60}")
        click.echo("Improved Samples (first 10)")
        click.echo(f"{'='*60}\n")
        
        for r in improved[:10]:
            click.echo(f"ID: {r['sample_id']}")
            click.echo(f"NL: {r['nl_statement'][:100]}...")
            click.echo(f"Original: {r['original_atl']}")
            click.echo(f"Improved: {r['finetuned_atl']}")
            click.echo(f"Confidence: {r.get('finetuned_confidence', 'N/A')}")
            click.echo()


@cli.command()
@click.argument('statement')
@click.option('--model', '-m', type=str, default=FINETUNED_MODEL, help='Model to use.')
@click.option('--domain', '-d', type=str, default=None, help='Domain context.')
@click.option('--agents', '-a', type=str, default=None, help='Comma-separated agents.')
@click.option('--verify/--no-verify', default=True, help='Verify the translation.')
def test(statement: str, model: str, domain: Optional[str], agents: Optional[str], verify: bool):
    """Test a single NL statement with the fine-tuned model."""
    client = get_openai_client()
    
    agents_list = [a.strip() for a in agents.split(',')] if agents else None
    
    click.echo(f"\nModel: {model}")
    click.echo(f"NL: {statement}")
    
    # Translate
    atl_formula, usage = translate_with_model(client, statement, model, domain, agents_list)
    
    # Validate syntax
    syntax_valid, syntax_errors = validate_atl_string(atl_formula)
    
    click.echo(f"\nATL: {atl_formula}")
    click.echo(f"Syntax Valid: {syntax_valid}")
    if syntax_errors:
        click.echo(f"Errors: {syntax_errors}")
    
    # Verify
    if verify and syntax_valid:
        click.echo("\nVerifying...")
        verification = verify_translation(client, statement, atl_formula, domain)
        click.echo(f"Verdict: {verification['verdict']}")
        click.echo(f"Confidence: {verification['confidence']}")
        if verification['issues']:
            click.echo(f"Issues: {verification['issues']}")
        click.echo(f"Notes: {verification['notes']}")


if __name__ == "__main__":
    cli()
