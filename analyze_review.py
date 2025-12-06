#!/usr/bin/env python3
"""
Analysis Script for Manual Review Results
==========================================

Analyzes the manually reviewed comparison data and generates statistics.

Usage:
    python analyze_review.py data/comparison_for_review_20251206_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import click


def calculate_accuracy(review_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy statistics from manual review data."""
    
    stats = {
        'openai_base': {'correct': 0, 'incorrect': 0, 'not_reviewed': 0},
        'openai_finetuned': {'correct': 0, 'incorrect': 0, 'not_reviewed': 0},
        'claude': {'correct': 0, 'incorrect': 0, 'not_reviewed': 0}
    }
    
    # Also track response times and syntax validity
    metrics = {
        'openai_base': {'response_times': [], 'syntax_valid': 0},
        'openai_finetuned': {'response_times': [], 'syntax_valid': 0},
        'claude': {'response_times': [], 'syntax_valid': 0}
    }
    
    for sample in review_data:
        review = sample.get('review', {})
        translations = sample.get('translations', {})
        
        # Count manual review results
        for model_key in ['openai_base', 'openai_finetuned', 'claude']:
            review_key = f'{model_key}_correct'
            correctness = review.get(review_key)
            
            if correctness is True:
                stats[model_key]['correct'] += 1
            elif correctness is False:
                stats[model_key]['incorrect'] += 1
            else:
                stats[model_key]['not_reviewed'] += 1
            
            # Collect metrics
            if model_key in translations:
                trans = translations[model_key]
                metrics[model_key]['response_times'].append(trans.get('response_time', 0))
                if trans.get('syntax_valid', False):
                    metrics[model_key]['syntax_valid'] += 1
    
    # Calculate percentages
    results = {}
    for model_key in stats.keys():
        total_reviewed = stats[model_key]['correct'] + stats[model_key]['incorrect']
        
        if total_reviewed > 0:
            accuracy = stats[model_key]['correct'] / total_reviewed
        else:
            accuracy = None
        
        avg_response_time = (
            sum(metrics[model_key]['response_times']) / len(metrics[model_key]['response_times'])
            if metrics[model_key]['response_times'] else 0
        )
        
        results[model_key] = {
            'correct': stats[model_key]['correct'],
            'incorrect': stats[model_key]['incorrect'],
            'not_reviewed': stats[model_key]['not_reviewed'],
            'total_reviewed': total_reviewed,
            'accuracy': accuracy,
            'syntax_valid_count': metrics[model_key]['syntax_valid'],
            'avg_response_time': avg_response_time
        }
    
    return results


def print_analysis(results: Dict[str, Any], total_samples: int):
    """Print formatted analysis results."""
    
    model_names = {
        'openai_base': 'GPT-4o-mini (Base)',
        'openai_finetuned': 'GPT-4o-mini (Fine-tuned)',
        'claude': 'Claude 3.5 Sonnet'
    }
    
    print("\n" + "=" * 70)
    print("MANUAL REVIEW ANALYSIS")
    print("=" * 70)
    print(f"\nTotal Samples: {total_samples}")
    
    print("\n" + "-" * 70)
    print("ACCURACY (Based on Manual Review)")
    print("-" * 70)
    print(f"{'Model':<30} {'Correct':<12} {'Incorrect':<12} {'Accuracy':<12}")
    print("-" * 70)
    
    for model_key, model_name in model_names.items():
        res = results[model_key]
        
        if res['accuracy'] is not None:
            accuracy_str = f"{res['accuracy']*100:.1f}%"
        else:
            accuracy_str = "N/A"
        
        correct_str = f"{res['correct']}/{res['total_reviewed']}"
        
        print(f"{model_name:<30} {correct_str:<12} {res['incorrect']:<12} {accuracy_str:<12}")
    
    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)
    print(f"{'Model':<30} {'Avg Response Time':<20} {'Syntax Valid':<15}")
    print("-" * 70)
    
    for model_key, model_name in model_names.items():
        res = results[model_key]
        time_str = f"{res['avg_response_time']:.3f}s"
        syntax_str = f"{res['syntax_valid_count']}/{total_samples}"
        
        print(f"{model_name:<30} {time_str:<20} {syntax_str:<15}")
    
    print("\n" + "-" * 70)
    print("REVIEW PROGRESS")
    print("-" * 70)
    print(f"{'Model':<30} {'Reviewed':<15} {'Not Reviewed':<15} {'Progress':<12}")
    print("-" * 70)
    
    for model_key, model_name in model_names.items():
        res = results[model_key]
        reviewed_str = f"{res['total_reviewed']}/{total_samples}"
        progress = res['total_reviewed'] / total_samples if total_samples > 0 else 0
        progress_str = f"{progress*100:.1f}%"
        
        print(f"{model_name:<30} {reviewed_str:<15} {res['not_reviewed']:<15} {progress_str:<12}")
    
    print("\n" + "=" * 70)
    
    # Summary insights
    print("\nKEY INSIGHTS:")
    print("-" * 70)
    
    # Find best accuracy
    best_accuracy = None
    best_model = None
    for model_key, res in results.items():
        if res['accuracy'] is not None:
            if best_accuracy is None or res['accuracy'] > best_accuracy:
                best_accuracy = res['accuracy']
                best_model = model_key
    
    if best_model:
        print(f"• Highest accuracy: {model_names[best_model]} ({best_accuracy*100:.1f}%)")
    
    # Find fastest
    fastest_time = min(res['avg_response_time'] for res in results.values())
    fastest_model = [k for k, res in results.items() if res['avg_response_time'] == fastest_time][0]
    print(f"• Fastest model: {model_names[fastest_model]} ({fastest_time:.3f}s avg)")
    
    # Syntax validity
    for model_key, res in results.items():
        syntax_rate = res['syntax_valid_count'] / total_samples if total_samples > 0 else 0
        if syntax_rate < 1.0:
            invalid_count = total_samples - res['syntax_valid_count']
            print(f"• {model_names[model_key]}: {invalid_count} syntax errors ({(1-syntax_rate)*100:.1f}%)")
    
    print("\n" + "=" * 70 + "\n")


@click.command()
@click.argument('review_file', type=click.Path(exists=True))
@click.option('--export', type=click.Path(), help='Export analysis results to JSON')
def main(review_file: str, export: str):
    """
    Analyze manual review results and display statistics.
    
    REVIEW_FILE: Path to the comparison_for_review_*.json file
    """
    
    # Load review data
    with open(review_file, 'r', encoding='utf-8') as f:
        review_data = json.load(f)
    
    total_samples = len(review_data)
    
    # Calculate statistics
    results = calculate_accuracy(review_data)
    
    # Print analysis
    print_analysis(results, total_samples)
    
    # Export if requested
    if export:
        export_data = {
            'source_file': review_file,
            'total_samples': total_samples,
            'results': results
        }
        
        with open(export, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Analysis exported to: {export}\n")


if __name__ == '__main__':
    main()
