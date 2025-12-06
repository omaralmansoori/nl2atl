#!/usr/bin/env python3
"""
Full Model Comparison Pipeline
================================

Comprehensive comparison of NL2ATL translation models with manual review support.

This script:
1. Generates a new set of NL statements (configurable count, default 100)
2. Translates each statement using all three models:
   - OpenAI GPT-4o-mini (base)
   - OpenAI GPT-4o-mini (fine-tuned)
   - Claude 3.5 Sonnet
3. Tracks response time and token usage for each translation
4. Validates syntax for all translations
5. Saves results in a structured format for manual review
6. Generates comparison statistics and visualizations

Output Format
-------------
The script generates:
- `comparison_raw_TIMESTAMP.jsonl`: All translations with metadata
- `comparison_for_review_TIMESTAMP.json`: Formatted for manual review
- `comparison_stats_TIMESTAMP.json`: Statistical summary
- `comparison_report_TIMESTAMP.md`: Human-readable report

Usage Examples
--------------
# Generate 100 NL statements and compare all models
python full_model_comparison.py --count 100

# Use specific domains
python full_model_comparison.py --count 50 --domains robotics,medical

# Load existing NL statements instead of generating new ones
python full_model_comparison.py --nl-file data/nl_statements.json

# Run with verbose output
python full_model_comparison.py --count 100 --verbose
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from dotenv import load_dotenv

from atl_syntax import is_valid, validate_atl_string
from dataset_gen import DatasetGenerator, GenerationConfig, GenerationMode
from nl2atl import load_templates_config

# Load environment variables
load_dotenv()

# Load test domains
CONFIG_DIR = Path(__file__).parent / "config"

def load_test_domains() -> dict:
    """Load test domains configuration."""
    test_domains_path = CONFIG_DIR / "test_domains.json"
    if test_domains_path.exists():
        with open(test_domains_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"test_domains": {}}


# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    "openai_base": {
        "name": "GPT-4o-mini (Base)",
        "provider": "openai",
        "model_id": "gpt-4o-mini-2024-07-18",
    },
    "openai_finetuned": {
        "name": "GPT-4o-mini (Fine-tuned)",
        "provider": "openai",
        "model_id": "ft:gpt-4o-mini-2024-07-18:personal:nl2atl-v1:CiGKGvnC",
    },
    "claude": {
        "name": "Claude 3.5 Sonnet",
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
    }
}

SYSTEM_PROMPT = (
    "You are an expert in translating natural language requirements into "
    "Alternating-time Temporal Logic (ATL) formulas. Given a natural language "
    "statement describing agent capabilities and temporal properties, generate "
    "the corresponding ATL formula using proper syntax with coalition operators "
    "⟨⟨...⟩⟩, temporal operators (G, F, X, U), and logical operators (∧, ∨, →, ¬)."
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TranslationResult:
    """Result of translating a single NL statement with one model."""
    model_key: str
    model_name: str
    atl_formula: str
    response_time: float  # seconds
    tokens_used: int
    syntax_valid: bool
    syntax_errors: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ComparisonSample:
    """Complete comparison data for one NL statement across all models."""
    sample_id: str
    nl_statement: str
    domain: Optional[str] = None
    agents: List[str] = field(default_factory=list)
    
    # Translations from each model
    translations: Dict[str, TranslationResult] = field(default_factory=dict)
    
    # Manual review fields (to be filled in)
    manually_reviewed: bool = False
    openai_base_correct: Optional[bool] = None
    openai_finetuned_correct: Optional[bool] = None
    claude_correct: Optional[bool] = None
    notes: str = ""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ComparisonStats:
    """Statistical summary of the comparison."""
    total_samples: int
    models_compared: List[str]
    
    # Response time statistics (in seconds)
    avg_response_times: Dict[str, float] = field(default_factory=dict)
    median_response_times: Dict[str, float] = field(default_factory=dict)
    min_response_times: Dict[str, float] = field(default_factory=dict)
    max_response_times: Dict[str, float] = field(default_factory=dict)
    
    # Token usage statistics
    avg_tokens: Dict[str, float] = field(default_factory=dict)
    total_tokens: Dict[str, int] = field(default_factory=dict)
    
    # Syntax validity
    syntax_valid_count: Dict[str, int] = field(default_factory=dict)
    syntax_valid_rate: Dict[str, float] = field(default_factory=dict)
    
    # Error counts
    error_count: Dict[str, int] = field(default_factory=dict)
    
    # Generation timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# NL Statement Generation
# =============================================================================

def generate_nl_statements(
    count: int,
    domains: Optional[List[str]] = None,
    verbose: bool = False,
    use_test_domains: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate NL statements using the dataset generator.
    
    Args:
        count: Number of statements to generate
        domains: Optional list of domains to focus on
        verbose: Print progress
        use_test_domains: Use test domains instead of training domains
        
    Returns:
        List of dicts with 'nl', 'domain', 'agents', etc.
    """
    if verbose:
        click.echo(f"Generating {count} NL statements...")
    
    # If using test domains, specify them
    if use_test_domains:
        test_config = load_test_domains()
        test_domain_names = list(test_config.get('test_domains', {}).keys())
        
        if verbose and domains is None:
            # Use all test domains if not specified
            domains = test_domain_names
            click.echo(f"Using test domains (different from training): {domains}")
        elif domains is None:
            domains = test_domain_names
    
    config = GenerationConfig(
        num_examples=count,
        mode=GenerationMode.LLM,
        nl_temperature=0.9,
        atl_temperature=0.1,
        domains=domains,
        verbose=verbose
    )
    
    generator = DatasetGenerator(config)
    samples = generator.generate()
    
    # Convert to simpler format
    nl_statements = []
    for sample in samples:
        nl_statements.append({
            "nl": sample.nl,
            "domain": sample.domain,
            "agents": sample.agents,
            "sample_id": f"gen_{len(nl_statements):04d}"
        })
    
    if verbose:
        click.echo(f"Generated {len(nl_statements)} statements")
    
    return nl_statements


def load_nl_statements(filepath: str) -> List[Dict[str, Any]]:
    """
    Load NL statements from a JSON file.
    
    Expected format:
    [
        {"nl": "...", "domain": "...", "agents": [...], "sample_id": "..."},
        ...
    ]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        statements = json.load(f)
    
    # Ensure each has a sample_id
    for idx, stmt in enumerate(statements):
        if 'sample_id' not in stmt:
            stmt['sample_id'] = f"loaded_{idx:04d}"
    
    return statements


# =============================================================================
# Translation with Metrics
# =============================================================================

def translate_with_openai(
    nl_text: str,
    model_id: str,
    model_key: str
) -> TranslationResult:
    """Translate using OpenAI API with metrics tracking."""
    try:
        import openai
        
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate to ATL: {nl_text}"}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        end_time = time.time()
        
        atl_formula = response.choices[0].message.content.strip()
        
        # Clean up the response
        if ":" in atl_formula and len(atl_formula.split(":")) == 2:
            atl_formula = atl_formula.split(":", 1)[1].strip()
        if "```" in atl_formula:
            lines = atl_formula.split("```")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("atl"):
                    atl_formula = line
                    break
        
        # Validate syntax
        syntax_valid, syntax_errors = validate_atl_string(atl_formula)
        
        return TranslationResult(
            model_key=model_key,
            model_name=MODELS[model_key]["name"],
            atl_formula=atl_formula,
            response_time=end_time - start_time,
            tokens_used=response.usage.total_tokens,
            syntax_valid=syntax_valid,
            syntax_errors=syntax_errors,
            error=None
        )
        
    except Exception as e:
        return TranslationResult(
            model_key=model_key,
            model_name=MODELS[model_key]["name"],
            atl_formula="",
            response_time=0.0,
            tokens_used=0,
            syntax_valid=False,
            syntax_errors=[],
            error=str(e)
        )


def translate_with_anthropic(
    nl_text: str,
    model_id: str,
    model_key: str
) -> TranslationResult:
    """Translate using Anthropic API with metrics tracking."""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        start_time = time.time()
        
        message = client.messages.create(
            model=model_id,
            max_tokens=200,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Translate to ATL: {nl_text}"}
            ]
        )
        
        end_time = time.time()
        
        atl_formula = message.content[0].text.strip()
        
        # Clean up the response
        if ":" in atl_formula and len(atl_formula.split(":")) == 2:
            atl_formula = atl_formula.split(":", 1)[1].strip()
        if "```" in atl_formula:
            lines = atl_formula.split("```")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("atl"):
                    atl_formula = line
                    break
        
        # Validate syntax
        syntax_valid, syntax_errors = validate_atl_string(atl_formula)
        
        return TranslationResult(
            model_key=model_key,
            model_name=MODELS[model_key]["name"],
            atl_formula=atl_formula,
            response_time=end_time - start_time,
            tokens_used=message.usage.input_tokens + message.usage.output_tokens,
            syntax_valid=syntax_valid,
            syntax_errors=syntax_errors,
            error=None
        )
        
    except Exception as e:
        return TranslationResult(
            model_key=model_key,
            model_name=MODELS[model_key]["name"],
            atl_formula="",
            response_time=0.0,
            tokens_used=0,
            syntax_valid=False,
            syntax_errors=[],
            error=str(e)
        )


def translate_with_model(
    nl_text: str,
    model_key: str
) -> TranslationResult:
    """Translate with the specified model."""
    model_info = MODELS[model_key]
    
    if model_info["provider"] == "openai":
        return translate_with_openai(nl_text, model_info["model_id"], model_key)
    elif model_info["provider"] == "anthropic":
        return translate_with_anthropic(nl_text, model_info["model_id"], model_key)
    else:
        return TranslationResult(
            model_key=model_key,
            model_name=model_info["name"],
            atl_formula="",
            response_time=0.0,
            tokens_used=0,
            syntax_valid=False,
            syntax_errors=[],
            error=f"Unknown provider: {model_info['provider']}"
        )


# =============================================================================
# Comparison Pipeline
# =============================================================================

def run_comparison(
    nl_statements: List[Dict[str, Any]],
    verbose: bool = False
) -> List[ComparisonSample]:
    """
    Run full comparison across all models for all NL statements.
    
    Args:
        nl_statements: List of NL statement dicts
        verbose: Print progress
        
    Returns:
        List of ComparisonSample objects with all translations
    """
    results = []
    
    total = len(nl_statements)
    
    for idx, stmt in enumerate(nl_statements, 1):
        if verbose:
            click.echo(f"\nProcessing sample {idx}/{total}: {stmt['nl'][:60]}...")
        
        sample = ComparisonSample(
            sample_id=stmt.get('sample_id', f'sample_{idx:04d}'),
            nl_statement=stmt['nl'],
            domain=stmt.get('domain'),
            agents=stmt.get('agents', [])
        )
        
        # Translate with each model
        for model_key in MODELS.keys():
            if verbose:
                click.echo(f"  Translating with {MODELS[model_key]['name']}...")
            
            result = translate_with_model(stmt['nl'], model_key)
            sample.translations[model_key] = result
            
            if verbose:
                status = "✓" if result.syntax_valid else "✗"
                click.echo(
                    f"    {status} {result.response_time:.3f}s | "
                    f"{result.tokens_used} tokens | {result.atl_formula[:50]}"
                )
        
        results.append(sample)
    
    return results


# =============================================================================
# Statistics and Reporting
# =============================================================================

def calculate_statistics(samples: List[ComparisonSample]) -> ComparisonStats:
    """Calculate comprehensive statistics from comparison samples."""
    stats = ComparisonStats(
        total_samples=len(samples),
        models_compared=list(MODELS.keys())
    )
    
    # Collect metrics by model
    metrics_by_model: Dict[str, Dict[str, List[float]]] = {
        model_key: {
            "response_times": [],
            "tokens": [],
            "syntax_valid": [],
            "errors": []
        }
        for model_key in MODELS.keys()
    }
    
    for sample in samples:
        for model_key, translation in sample.translations.items():
            if translation.error is None:
                metrics_by_model[model_key]["response_times"].append(translation.response_time)
                metrics_by_model[model_key]["tokens"].append(translation.tokens_used)
                metrics_by_model[model_key]["syntax_valid"].append(1 if translation.syntax_valid else 0)
            else:
                metrics_by_model[model_key]["errors"].append(1)
    
    # Calculate statistics
    for model_key, metrics in metrics_by_model.items():
        # Response times
        if metrics["response_times"]:
            times = sorted(metrics["response_times"])
            stats.avg_response_times[model_key] = sum(times) / len(times)
            stats.median_response_times[model_key] = times[len(times) // 2]
            stats.min_response_times[model_key] = min(times)
            stats.max_response_times[model_key] = max(times)
        
        # Tokens
        if metrics["tokens"]:
            stats.avg_tokens[model_key] = sum(metrics["tokens"]) / len(metrics["tokens"])
            stats.total_tokens[model_key] = sum(metrics["tokens"])
        
        # Syntax validity
        if metrics["syntax_valid"]:
            valid_count = sum(metrics["syntax_valid"])
            stats.syntax_valid_count[model_key] = valid_count
            stats.syntax_valid_rate[model_key] = valid_count / len(metrics["syntax_valid"])
        
        # Errors
        stats.error_count[model_key] = len(metrics["errors"])
    
    return stats


def generate_markdown_report(
    stats: ComparisonStats,
    samples: List[ComparisonSample],
    output_path: Path
):
    """Generate a human-readable markdown report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# NL2ATL Model Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Samples:** {stats.total_samples}\n\n")
        
        # Response Time Comparison
        f.write("## Response Time Comparison\n\n")
        f.write("| Model | Avg (s) | Median (s) | Min (s) | Max (s) |\n")
        f.write("|-------|---------|------------|---------|----------|\n")
        
        for model_key in stats.models_compared:
            model_name = MODELS[model_key]["name"]
            avg = stats.avg_response_times.get(model_key, 0)
            median = stats.median_response_times.get(model_key, 0)
            min_time = stats.min_response_times.get(model_key, 0)
            max_time = stats.max_response_times.get(model_key, 0)
            f.write(f"| {model_name} | {avg:.3f} | {median:.3f} | {min_time:.3f} | {max_time:.3f} |\n")
        
        # Token Usage
        f.write("\n## Token Usage\n\n")
        f.write("| Model | Avg Tokens | Total Tokens |\n")
        f.write("|-------|------------|-------------|\n")
        
        for model_key in stats.models_compared:
            model_name = MODELS[model_key]["name"]
            avg = stats.avg_tokens.get(model_key, 0)
            total = stats.total_tokens.get(model_key, 0)
            f.write(f"| {model_name} | {avg:.1f} | {total} |\n")
        
        # Syntax Validity
        f.write("\n## Syntax Validity\n\n")
        f.write("| Model | Valid Count | Valid Rate |\n")
        f.write("|-------|-------------|------------|\n")
        
        for model_key in stats.models_compared:
            model_name = MODELS[model_key]["name"]
            count = stats.syntax_valid_count.get(model_key, 0)
            rate = stats.syntax_valid_rate.get(model_key, 0)
            f.write(f"| {model_name} | {count}/{stats.total_samples} | {rate*100:.1f}% |\n")
        
        # Error Summary
        f.write("\n## Error Summary\n\n")
        f.write("| Model | Error Count |\n")
        f.write("|-------|-------------|\n")
        
        for model_key in stats.models_compared:
            model_name = MODELS[model_key]["name"]
            errors = stats.error_count.get(model_key, 0)
            f.write(f"| {model_name} | {errors} |\n")
        
        # Sample Previews
        f.write("\n## Sample Previews (First 10)\n\n")
        
        for sample in samples[:10]:
            f.write(f"### {sample.sample_id}\n\n")
            f.write(f"**NL:** {sample.nl_statement}\n\n")
            
            for model_key, translation in sample.translations.items():
                model_name = MODELS[model_key]["name"]
                status = "✓" if translation.syntax_valid else "✗"
                f.write(f"- **{model_name}** ({translation.response_time:.3f}s): {status} `{translation.atl_formula}`\n")
            
            f.write("\n")
        
        f.write("\n---\n\n")
        f.write("*For manual review, see the corresponding JSON file.*\n")


def prepare_review_format(samples: List[ComparisonSample]) -> List[Dict[str, Any]]:
    """
    Prepare samples in a format optimized for manual review.
    
    Returns a list with clear structure for reviewers to mark correctness.
    """
    review_data = []
    
    for sample in samples:
        review_item = {
            "sample_id": sample.sample_id,
            "nl_statement": sample.nl_statement,
            "domain": sample.domain,
            "agents": sample.agents,
            "translations": {},
            "review": {
                "openai_base_correct": None,  # True/False/None
                "openai_finetuned_correct": None,
                "claude_correct": None,
                "notes": ""
            }
        }
        
        for model_key, translation in sample.translations.items():
            review_item["translations"][model_key] = {
                "atl_formula": translation.atl_formula,
                "response_time": round(translation.response_time, 3),
                "tokens_used": translation.tokens_used,
                "syntax_valid": translation.syntax_valid,
                "syntax_errors": translation.syntax_errors,
                "error": translation.error
            }
        
        review_data.append(review_item)
    
    return review_data


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    '--count',
    type=int,
    default=100,
    help='Number of NL statements to generate and compare'
)
@click.option(
    '--nl-file',
    type=click.Path(exists=True),
    help='Load NL statements from file instead of generating'
)
@click.option(
    '--domains',
    type=str,
    help='Comma-separated list of domains (e.g., healthcare,smart_home,autonomous_vehicles)'
)
@click.option(
    '--use-training-domains',
    is_flag=True,
    help='Use training domains instead of test domains (default: use test domains)'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='data',
    help='Output directory for results'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Print detailed progress'
)
def main(
    count: int,
    nl_file: Optional[str],
    domains: Optional[str],
    use_training_domains: bool,
    output_dir: str,
    verbose: bool
):
    """
    Run comprehensive model comparison for NL2ATL translation.
    
    This script generates NL statements, translates them with all models,
    and produces detailed comparison reports with metrics.
    
    By default, uses TEST domains (different from training) to evaluate
    generalization. Use --use-training-domains to test on familiar domains.
    """
    click.echo("=" * 60)
    click.echo("NL2ATL Full Model Comparison")
    click.echo("=" * 60)
    
    # Check API keys
    if not os.environ.get("OPENAI_API_KEY"):
        click.echo("Error: OPENAI_API_KEY not set", err=True)
        sys.exit(1)
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo("Warning: ANTHROPIC_API_KEY not set. Claude will be skipped.")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Get NL statements
    if nl_file:
        click.echo(f"\nLoading NL statements from {nl_file}...")
        nl_statements = load_nl_statements(nl_file)
    else:
        domain_list = domains.split(',') if domains else None
        use_test = not use_training_domains
        
        if use_test:
            click.echo("\n📝 Using TEST domains (different from training data)")
            click.echo("   This tests model generalization to unseen domains.")
            test_config = load_test_domains()
            test_domain_names = list(test_config.get('test_domains', {}).keys())
            click.echo(f"   Test domains: {', '.join(test_domain_names)}")
        else:
            click.echo("\n📝 Using TRAINING domains")
            click.echo("   This tests model performance on familiar domains.")
        
        nl_statements = generate_nl_statements(count, domain_list, verbose, use_test_domains=use_test)
    
    click.echo(f"Total NL statements: {len(nl_statements)}")
    
    # Save NL statements
    nl_output = output_path / f"comparison_nl_{timestamp}.json"
    with open(nl_output, 'w', encoding='utf-8') as f:
        json.dump(nl_statements, f, indent=2, ensure_ascii=False)
    click.echo(f"Saved NL statements to: {nl_output}")
    
    # Step 2: Run comparison
    click.echo("\nRunning translations with all models...")
    comparison_samples = run_comparison(nl_statements, verbose)
    
    # Step 3: Calculate statistics
    click.echo("\nCalculating statistics...")
    stats = calculate_statistics(comparison_samples)
    
    # Step 4: Save results
    click.echo("\nSaving results...")
    
    # Raw JSONL format
    raw_output = output_path / f"comparison_raw_{timestamp}.jsonl"
    with open(raw_output, 'w', encoding='utf-8') as f:
        for sample in comparison_samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')
    click.echo(f"Saved raw results to: {raw_output}")
    
    # Review format
    review_data = prepare_review_format(comparison_samples)
    review_output = output_path / f"comparison_for_review_{timestamp}.json"
    with open(review_output, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
    click.echo(f"Saved review format to: {review_output}")
    
    # Statistics
    stats_output = output_path / f"comparison_stats_{timestamp}.json"
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, indent=2, ensure_ascii=False)
    click.echo(f"Saved statistics to: {stats_output}")
    
    # Markdown report
    report_output = output_path / f"comparison_report_{timestamp}.md"
    generate_markdown_report(stats, comparison_samples, report_output)
    click.echo(f"Saved report to: {report_output}")
    
    # Step 5: Display summary
    click.echo("\n" + "=" * 60)
    click.echo("COMPARISON SUMMARY")
    click.echo("=" * 60)
    
    click.echo(f"\nTotal Samples: {stats.total_samples}")
    
    click.echo("\nAverage Response Times:")
    for model_key in stats.models_compared:
        model_name = MODELS[model_key]["name"]
        avg_time = stats.avg_response_times.get(model_key, 0)
        click.echo(f"  {model_name}: {avg_time:.3f}s")
    
    click.echo("\nSyntax Validity Rates:")
    for model_key in stats.models_compared:
        model_name = MODELS[model_key]["name"]
        rate = stats.syntax_valid_rate.get(model_key, 0)
        count = stats.syntax_valid_count.get(model_key, 0)
        click.echo(f"  {model_name}: {count}/{stats.total_samples} ({rate*100:.1f}%)")
    
    click.echo("\n" + "=" * 60)
    click.echo(f"\n✓ All results saved to {output_path}")
    click.echo(f"\n→ For manual review, open: {review_output}")
    click.echo(f"→ For report, open: {report_output}")


if __name__ == "__main__":
    main()
