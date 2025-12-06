#!/usr/bin/env python3
"""
Demo Script - Quick Model Comparison
=====================================

A simple demo showing how to use the comparison tools programmatically.
"""

from full_model_comparison import (
    generate_nl_statements,
    run_comparison,
    calculate_statistics,
    prepare_review_format,
    generate_markdown_report
)
from pathlib import Path
import json
from datetime import datetime


def demo_quick_comparison(num_samples: int = 5):
    """Run a quick comparison with just 5 samples for demonstration."""
    
    print("=" * 60)
    print("NL2ATL Model Comparison Demo")
    print("=" * 60)
    print(f"\nGenerating {num_samples} NL statements...")
    
    # Generate NL statements
    nl_statements = generate_nl_statements(num_samples, verbose=True)
    
    print(f"\nRunning comparison with all models...")
    print("(This will take about {num_samples * 2} seconds)\n")
    
    # Run comparison
    comparison_samples = run_comparison(nl_statements, verbose=True)
    
    # Calculate statistics
    stats = calculate_statistics(comparison_samples)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nSamples processed: {stats.total_samples}")
    
    print("\nAverage Response Times:")
    for model in stats.models_compared:
        avg_time = stats.avg_response_times.get(model, 0)
        print(f"  {model}: {avg_time:.3f}s")
    
    print("\nSyntax Validity:")
    for model in stats.models_compared:
        count = stats.syntax_valid_count.get(model, 0)
        rate = stats.syntax_valid_rate.get(model, 0)
        print(f"  {model}: {count}/{stats.total_samples} ({rate*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("\nSample Translations:")
    print("-" * 60)
    
    # Show first sample in detail
    if comparison_samples:
        sample = comparison_samples[0]
        print(f"\nNL: {sample.nl_statement}")
        print(f"Domain: {sample.domain}")
        print("\nTranslations:")
        
        for model_key, translation in sample.translations.items():
            status = "✓" if translation.syntax_valid else "✗"
            print(f"\n  {model_key}:")
            print(f"    ATL: {translation.atl_formula}")
            print(f"    Time: {translation.response_time:.3f}s")
            print(f"    Valid: {status}")
    
    print("\n" + "=" * 60)
    print("\n✅ Demo complete!")
    print("\nTo run a full comparison with 100 samples:")
    print("  python full_model_comparison.py --count 100 --verbose")
    print("\nTo launch the interactive GUI:")
    print("  python gui_tester.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    
    # Check for API keys
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    # Run demo
    try:
        demo_quick_comparison(num_samples=5)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
