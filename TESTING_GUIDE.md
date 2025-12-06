# Model Testing and Comparison Tools

This directory contains tools for testing and comparing different NL2ATL translation models.

## Tools Overview

### 1. GUI Tester (`gui_tester.py`)

An interactive graphical interface for real-time model testing and comparison.

**Features:**
- Test fine-tuned model against standard LLMs (OpenAI GPT-4o-mini, Claude)
- Real-time response time tracking
- Side-by-side comparison of translations
- Syntax validation with error highlighting
- Export test results to JSON
- Test history tracking

**Usage:**
```bash
python gui_tester.py
```

**Requirements:**
- Python's tkinter (usually included with Python)
- `OPENAI_API_KEY` environment variable
- `ANTHROPIC_API_KEY` environment variable (optional, for Claude)

**Interface:**
1. Enter a natural language requirement in the input box
2. Click "Translate with All Models" to see translations from all three models
3. Review the ATL formulas, response times, token usage, and syntax validity
4. Export results for later analysis

---

### 2. Full Model Comparison Script (`full_model_comparison.py`)

Comprehensive automated comparison pipeline for systematic evaluation.

**Features:**
- Generate new NL statements or load from file
- Translate with all models (OpenAI base, fine-tuned, Claude)
- Track response times and token usage
- Validate syntax automatically
- Generate formatted output for manual review
- Create statistical reports and visualizations

**Usage:**

Generate 100 new statements and compare:
```bash
python full_model_comparison.py --count 100 --verbose
```

Use existing NL statements:
```bash
python full_model_comparison.py --nl-file data/my_statements.json
```

Specify domains:
```bash
python full_model_comparison.py --count 50 --domains robotics,medical,network
```

**Output Files:**

The script generates several files in the `data/` directory:

1. `comparison_nl_TIMESTAMP.json` - The NL statements used
2. `comparison_raw_TIMESTAMP.jsonl` - Complete raw data (all translations, metrics)
3. `comparison_for_review_TIMESTAMP.json` - Formatted for manual review
4. `comparison_stats_TIMESTAMP.json` - Statistical summary
5. `comparison_report_TIMESTAMP.md` - Human-readable markdown report

**Manual Review Format:**

The `comparison_for_review_*.json` file is structured for easy manual review:

```json
[
  {
    "sample_id": "gen_0001",
    "nl_statement": "Agents 1 and 2 can ensure the system never crashes",
    "domain": "software",
    "agents": ["1", "2"],
    "translations": {
      "openai_base": {
        "atl_formula": "⟨⟨1,2⟩⟩ G ¬crash",
        "response_time": 0.542,
        "tokens_used": 85,
        "syntax_valid": true,
        "syntax_errors": []
      },
      "openai_finetuned": {
        "atl_formula": "⟨⟨1,2⟩⟩ G ¬crash",
        "response_time": 0.321,
        "tokens_used": 78,
        "syntax_valid": true,
        "syntax_errors": []
      },
      "claude": {
        "atl_formula": "⟨⟨1,2⟩⟩ G (¬crash)",
        "response_time": 0.687,
        "tokens_used": 92,
        "syntax_valid": true,
        "syntax_errors": []
      }
    },
    "review": {
      "openai_base_correct": null,      // ← Fill in: true/false
      "openai_finetuned_correct": null, // ← Fill in: true/false
      "claude_correct": null,           // ← Fill in: true/false
      "notes": ""                       // ← Add any notes
    }
  }
]
```

**Performing Manual Review:**

1. Open the `comparison_for_review_*.json` file
2. For each sample, evaluate whether each model's translation is semantically correct
3. Set `openai_base_correct`, `openai_finetuned_correct`, and `claude_correct` to:
   - `true` if the translation correctly captures the NL meaning
   - `false` if the translation is incorrect or incomplete
   - Leave as `null` if unsure or skipping
4. Add any observations in the `notes` field
5. Save the file for later analysis

---

## NL2ATL Module Enhancement

The `nl2atl.py` module now includes a new function for metrics tracking:

### `translate_nl_to_atl_with_metrics()`

Extended version of `translate_nl_to_atl()` that tracks performance metrics.

**Returns:**
```python
{
    "formulas": ["⟨⟨1,2⟩⟩ G safe"],  # List of valid ATL formulas
    "response_time": 0.542,            # Time taken for LLM call (seconds)
    "tokens_used": 85,                 # Total tokens used
    "candidates_found": 1,             # Number of candidates extracted
    "valid_count": 1                   # Number of syntactically valid formulas
}
```

**Usage:**
```python
from nl2atl import translate_nl_to_atl_with_metrics

result = translate_nl_to_atl_with_metrics(
    "Agents 1 and 2 can ensure safety",
    config={"provider": "openai", "temperature": 0.1}
)

print(f"Formulas: {result['formulas']}")
print(f"Response time: {result['response_time']:.3f}s")
print(f"Tokens used: {result['tokens_used']}")
```

---

## Model Configuration

The tools are configured to use the following models:

| Key | Model | Provider | Model ID |
|-----|-------|----------|----------|
| `openai_base` | GPT-4o-mini (Base) | OpenAI | gpt-4o-mini-2024-07-18 |
| `openai_finetuned` | GPT-4o-mini (Fine-tuned) | OpenAI | ft:gpt-4o-mini-2024-07-18:personal:nl2atl-v1:CiGKGvnC |
| `claude` | Claude 3.5 Sonnet | Anthropic | claude-3-5-sonnet-20241022 |

To update the fine-tuned model ID, edit the configuration in:
- `gui_tester.py` (MODELS dict)
- `full_model_comparison.py` (MODELS dict)

---

## Environment Setup

### Required Environment Variables

```bash
# OpenAI API key (required for OpenAI models)
export OPENAI_API_KEY="sk-..."

# Anthropic API key (required for Claude)
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Example Workflow

### Complete Comparison Study

1. **Generate NL statements and run comparison:**
   ```bash
   python full_model_comparison.py --count 100 --verbose
   ```

2. **Review the generated report:**
   ```bash
   cat data/comparison_report_*.md
   ```

3. **Perform manual review:**
   - Open `data/comparison_for_review_*.json`
   - Mark each translation as correct/incorrect
   - Save the file

4. **Analyze results:**
   ```python
   import json
   
   # Load reviewed data
   with open('data/comparison_for_review_20251206_*.json') as f:
       data = json.load(f)
   
   # Count correct translations per model
   stats = {
       'openai_base': {'correct': 0, 'total': 0},
       'openai_finetuned': {'correct': 0, 'total': 0},
       'claude': {'correct': 0, 'total': 0}
   }
   
   for sample in data:
       review = sample['review']
       if review['openai_base_correct'] is not None:
           stats['openai_base']['total'] += 1
           if review['openai_base_correct']:
               stats['openai_base']['correct'] += 1
       # ... repeat for other models
   
   # Calculate accuracy
   for model, counts in stats.items():
       if counts['total'] > 0:
           accuracy = counts['correct'] / counts['total']
           print(f"{model}: {accuracy:.1%} ({counts['correct']}/{counts['total']})")
   ```

### Interactive Testing

1. **Launch the GUI:**
   ```bash
   python gui_tester.py
   ```

2. **Test various requirements:**
   - Click "Load Example" for sample requirements
   - Or enter your own NL statements
   - Click "Translate with All Models"
   - Compare response times and outputs

3. **Export results:**
   - Click "Export Results" to save test history
   - Use the exported JSON for further analysis

---

## Metrics Tracked

### Performance Metrics
- **Response Time**: Time taken for LLM API call (seconds)
- **Token Usage**: Total tokens consumed (input + output)

### Quality Metrics
- **Syntax Validity**: Whether the ATL formula parses correctly
- **Syntax Errors**: List of validation errors (if any)

### Manual Review Metrics
- **Semantic Correctness**: Human judgment of translation accuracy
- **Notes**: Qualitative observations

---

## Tips for Manual Review

1. **Focus on semantic equivalence**, not exact formula matching
   - Different valid formulas may express the same requirement
   - e.g., `⟨⟨1⟩⟩ G safe` ≡ `⟨⟨1⟩⟩ G (safe)` (parentheses don't change meaning)

2. **Check for common issues:**
   - Wrong agents in coalition
   - Incorrect temporal operator (G vs F, X vs G)
   - Missing or extra logical operators
   - Misinterpreted implications or conditions

3. **Be consistent in your criteria:**
   - Define what counts as "correct" before starting
   - Consider partial credit vs binary correct/incorrect
   - Document edge cases in notes

4. **Batch similar domains together:**
   - Reviewing similar statements together helps maintain consistency
   - Sort by domain if needed before reviewing

---

## Troubleshooting

### GUI won't launch
- Ensure tkinter is installed (usually included with Python)
- On macOS: `brew install python-tk`
- On Ubuntu: `sudo apt-get install python3-tk`

### API errors
- Verify API keys are set correctly
- Check API rate limits
- Ensure sufficient API credits

### Slow performance
- Reduce batch size for comparison script
- Use `--verbose` to monitor progress
- Consider running overnight for large comparisons (100+ samples)

### Import errors
- Run: `pip install -r requirements.txt`
- Ensure you're in the correct virtual environment

---

## Future Enhancements

Potential improvements for these tools:

- [ ] Add statistical significance testing
- [ ] Visualizations (charts, graphs) for comparison results
- [ ] Support for additional LLM providers
- [ ] Batch import of NL statements from CSV
- [ ] Inter-annotator agreement calculations
- [ ] Automated semantic equivalence checking
- [ ] Cost tracking for API usage

---

## Contributing

When extending these tools:

1. Maintain the existing output format for compatibility
2. Add metrics to the `TranslationResult` dataclass
3. Update the review format if new fields are added
4. Document new features in this README
5. Test with multiple providers before committing

---

## License

Part of the NL2ATL project. See main README for license information.
