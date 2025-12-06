#!/bin/bash
# Quick Start Script for Model Comparison Experiment
# This script automates the setup and execution of a full model comparison

set -e

echo "=========================================="
echo "NL2ATL Model Comparison Experiment"
echo "=========================================="
echo ""

# Check environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Please set it with: export OPENAI_API_KEY='sk-...'"
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: ANTHROPIC_API_KEY not set"
    echo "Claude will be skipped in the comparison"
    echo ""
fi

# Parse arguments
COUNT=${1:-100}
OUTPUT_DIR="data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Configuration:"
echo "  - Samples to generate: $COUNT"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Timestamp: $TIMESTAMP"
echo ""

# Confirm before running
read -p "Proceed with comparison? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1: Running full model comparison..."
echo "This will take several minutes..."
echo ""

python full_model_comparison.py \
    --count $COUNT \
    --output-dir $OUTPUT_DIR \
    --verbose

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo ""

# Find the latest files
LATEST_REVIEW=$(ls -t $OUTPUT_DIR/comparison_for_review_*.json 2>/dev/null | head -1)
LATEST_REPORT=$(ls -t $OUTPUT_DIR/comparison_report_*.md 2>/dev/null | head -1)
LATEST_STATS=$(ls -t $OUTPUT_DIR/comparison_stats_*.json 2>/dev/null | head -1)

echo "Generated files:"
if [ -n "$LATEST_REVIEW" ]; then
    echo "  - Review file: $LATEST_REVIEW"
fi
if [ -n "$LATEST_REPORT" ]; then
    echo "  - Report: $LATEST_REPORT"
fi
if [ -n "$LATEST_STATS" ]; then
    echo "  - Statistics: $LATEST_STATS"
fi
echo ""

# Display quick stats
if [ -n "$LATEST_STATS" ]; then
    echo "Quick Statistics:"
    echo ""
    python -c "
import json
import sys

with open('$LATEST_STATS') as f:
    stats = json.load(f)

print(f\"Total Samples: {stats['total_samples']}\")
print()
print('Average Response Times:')
for model, time in stats['avg_response_times'].items():
    print(f\"  {model}: {time:.3f}s\")
print()
print('Syntax Validity Rates:')
for model, rate in stats['syntax_valid_rate'].items():
    count = stats['syntax_valid_count'][model]
    total = stats['total_samples']
    print(f\"  {model}: {count}/{total} ({rate*100:.1f}%)\")
" 2>/dev/null || echo "Could not parse statistics"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Review the report:"
if [ -n "$LATEST_REPORT" ]; then
    echo "   cat $LATEST_REPORT"
fi
echo ""
echo "2. Perform manual review:"
if [ -n "$LATEST_REVIEW" ]; then
    echo "   code $LATEST_REVIEW"
    echo "   (or use your preferred editor)"
fi
echo ""
echo "3. For each sample, mark translations as correct/incorrect:"
echo "   - Set 'openai_base_correct' to true/false"
echo "   - Set 'openai_finetuned_correct' to true/false"
echo "   - Set 'claude_correct' to true/false"
echo "   - Add notes if needed"
echo ""
echo "4. Save the file and analyze results"
echo ""
echo "=========================================="
