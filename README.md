# NL2ATL: Natural Language to Alternating-time Temporal Logic

A Python toolkit for translating natural language requirements into ATL (Alternating-time Temporal Logic) formulas using LLM-based translation and synthetic dataset generation.

## Overview

This project provides:
- **ATL Syntax Module** (`atl_syntax.py`): Data structures, parsing, and validation for ATL formulas
- **NL→ATL Translation** (`nl2atl.py`): LLM-based translation from natural language to ATL
- **Dataset Generation** (`dataset_gen.py`): Synthetic NL-ATL pair generation pipeline
- **Evaluation Helpers** (`evaluation.py`): Quality statistics and spot-checking utilities

## Installation

### Requirements
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/nl2atl.git
cd nl2atl

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Set up your LLM API credentials:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

## Quick Start

### Single Translation

Translate a natural language requirement to ATL:

```python
from nl2atl import translate_nl_to_atl

nl_text = "Agents 1 and 2 can ensure the system never crashes."
formulas = translate_nl_to_atl(nl_text)
print(formulas)  # ['⟨⟨1,2⟩⟩ G ¬crash']
```

### Command Line Translation

```bash
python -m nl2atl "The robot can eventually reach the goal."
```

### Generate a Dataset

Generate a small dataset of NL-ATL pairs:

```bash
# Generate 100 examples without LLM (template-based only)
python -m dataset_gen --num-examples 100 --no-paraphrase --no-crosscheck --out data/train.jsonl

# Generate with LLM paraphrasing (requires API key)
python -m dataset_gen --num-examples 500 --out data/train.jsonl --verbose
```

### Evaluate a Dataset

```bash
# Compute statistics
python -m evaluation stats data/train.jsonl

# Spot-check 5 random examples
python -m evaluation spot-check data/train.jsonl -k 5
```

## Project Structure

```
nl2atl/
├── atl_syntax.py        # ATL data structures, parser, validation
├── nl2atl.py            # LLM client, prompt building, translation
├── dataset_gen.py       # Dataset generation pipeline
├── evaluation.py        # Evaluation and quality checking
├── config/
│   ├── atl_fragment.yaml    # ATL fragment configuration
│   └── templates_atl.json   # Seed templates and few-shot examples
├── requirements.txt
└── README.md
```

## ATL Syntax Reference

### Strategic Modality
- `⟨⟨A⟩⟩ φ` : Coalition A has a strategy to ensure φ
- A is a comma-separated list of agent identifiers (e.g., `⟨⟨1,2⟩⟩` or `⟨⟨robot⟩⟩`)

### Temporal Operators
- `X φ` : φ holds in the next state (neXt)
- `F φ` : φ eventually holds (Future)
- `G φ` : φ always holds (Globally)
- `φ U ψ` : φ holds until ψ becomes true (Until)

### Boolean Connectives
- `¬φ` : negation (NOT)
- `φ ∧ ψ` : conjunction (AND)
- `φ ∨ ψ` : disjunction (OR)
- `φ → ψ` : implication (IF-THEN)

### Examples

| Natural Language | ATL Formula |
|-----------------|-------------|
| Agents 1 and 2 can ensure the system never crashes | `⟨⟨1,2⟩⟩ G ¬crash` |
| The robot can eventually reach the goal | `⟨⟨robot⟩⟩ F goal_reached` |
| The controller can keep temperature stable until alarm | `⟨⟨controller⟩⟩ (temp_stable U alarm)` |
| Whenever a request is made, the server can respond | `⟨⟨server⟩⟩ G (request → F response)` |

## Configuration

### ATL Fragment (`config/atl_fragment.yaml`)

Define allowed operators and constraints:

```yaml
temporal_operators:
  - X
  - F
  - G
  - U

constraints:
  max_nesting_depth: 4
  max_coalition_size: 5
```

### Templates (`config/templates_atl.json`)

Define seed templates for dataset generation:

```json
{
  "templates": [
    {
      "id": "safety_basic",
      "atl_template": "⟨⟨{coalition}⟩⟩ G ¬{p}",
      "nl_template": "The coalition {coalition_nl} can ensure that {p_nl} never happens."
    }
  ],
  "few_shot_examples": [
    {
      "nl": "Agents 1 and 2 can ensure the system never crashes.",
      "atl": "⟨⟨1,2⟩⟩ G ¬crash"
    }
  ]
}
```

## API Reference

### `atl_syntax.py`

```python
from atl_syntax import parse_atl, normalize_atl, is_valid, validate_atl_string

# Parse ATL formula
formula = parse_atl("⟨⟨1,2⟩⟩ G ¬crash")

# Normalize to canonical form
normalized = normalize_atl(formula)

# Validate formula
result = is_valid(formula, max_depth=4, max_coalition_size=5)
print(result.valid, result.errors)

# Quick validation
is_ok, errors = validate_atl_string("⟨⟨1⟩⟩ F goal")
```

### `nl2atl.py`

```python
from nl2atl import (
    translate_nl_to_atl,
    critique_nl_atl_pair,
    get_llm_client,
    build_translation_prompt,
)

# Translate NL to ATL
formulas = translate_nl_to_atl("The robot can reach the goal.")

# Critique a pair
result = critique_nl_atl_pair(
    nl_text="The robot can reach the goal.",
    atl_text="⟨⟨robot⟩⟩ F goal"
)
print(result)  # {"ok": True, "issues": [], ...}

# Use custom LLM client
client = get_llm_client("openai", model="gpt-4o")
formulas = translate_nl_to_atl("...", client=client)
```

### `dataset_gen.py`

```python
from dataset_gen import DatasetGenerator, GenerationConfig, save_dataset

config = GenerationConfig(
    num_examples=100,
    use_llm_paraphrasing=True,
    use_cross_checking=True,
    llm_provider="openai",
)

generator = DatasetGenerator(config)
pairs = generator.generate()

save_dataset(pairs, Path("data/train.jsonl"), format="jsonl")
```

### `evaluation.py`

```python
from evaluation import compute_basic_stats, spot_check_examples

# Get statistics
stats = compute_basic_stats("data/train.jsonl")
print(stats["syntax_valid_fraction"])

# Spot check
results = spot_check_examples("data/train.jsonl", k=5)
```

## Testing Without API Keys

For testing without LLM API access:

```python
from nl2atl import get_llm_client, translate_nl_to_atl

# Use mock client
mock_client = get_llm_client("mock", responses=["⟨⟨1⟩⟩ G ¬error"])
formulas = translate_nl_to_atl("Agent 1 prevents errors.", client=mock_client)
```

```bash
# Generate dataset without LLM calls
python -m dataset_gen --no-paraphrase --no-crosscheck --out data/test.jsonl
```

## References

This work is inspired by:
- [NL2TL](https://yongchao98.github.io/MIT-realm-NL2TL/) - NL to Temporal Logic translation
- [VLTL-style pipelines](https://arxiv.org/html/2507.00877v1) - Verification-aware approaches
- [Related temporal logic work](https://arxiv.org/abs/2305.07766) - LLM-based logic translation

## License

MIT License - see LICENSE file for details.
