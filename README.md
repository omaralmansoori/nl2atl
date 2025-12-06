# Verification-Aware NL2ATL

A Python toolkit for **Natural Language to Alternating-time Temporal Logic (ATL)** translation with cross-model verification and dataset generation.

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Pre-Run Checklist](#-pre-run-checklist)
- [Project Structure](#-project-structure)
- [Core Modules](#-core-modules)
- [Verification Pipeline](#-verification-pipeline)
- [Configuration](#-configuration)
- [CLI Reference](#-cli-reference)
- [Next Steps](#-next-steps)
- [Requirements Fulfillment](#-requirements-fulfillment)
- [Development](#-development)
- [License](#-license)

---

## 🎯 Overview

This project provides a complete, verification-aware pipeline for:

1. **ATL Formula Handling**: Parse, validate, normalize, and manipulate ATL formulas
2. **NL→ATL Translation**: Convert natural language descriptions to ATL using LLMs
3. **Cross-Model Verification**: Use different LLM providers to verify translations
4. **Dataset Generation**: Create synthetic NL-ATL pairs with quality assurance
5. **Sample Unification**: Parse and normalize samples from multiple sources
6. **Metrics & Monitoring**: Track pipeline performance and quality

### Supported ATL Fragment

```
φ ::= p                     # Atomic proposition
    | ¬φ                    # Negation
    | φ ∧ φ                 # Conjunction
    | φ ∨ φ                 # Disjunction
    | φ → φ                 # Implication
    | ⟨⟨Γ⟩⟩ X φ             # Next (coalition Γ)
    | ⟨⟨Γ⟩⟩ F φ             # Eventually (coalition Γ)
    | ⟨⟨Γ⟩⟩ G φ             # Always (coalition Γ)
    | ⟨⟨Γ⟩⟩ φ U φ           # Until (coalition Γ)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NL2ATL Verification Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Samples    │    │  Templates   │    │    LLM       │          │
│  │  (3 files)   │    │  (JSON)      │    │  Paraphrase  │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌────────────────────────────────────────────────────────┐        │
│  │              Sample Store (Unified Schema)              │        │
│  │   - Parse diverse formats                               │        │
│  │   - Normalize ATL syntax (ASCII ↔ Unicode)             │        │
│  │   - Extract agents, operators, atoms                    │        │
│  │   - Classify domains                                    │        │
│  └────────────────────────┬───────────────────────────────┘        │
│                           │                                         │
│                           ▼                                         │
│  ┌────────────────────────────────────────────────────────┐        │
│  │            Verification Pipeline (3-Stage)              │        │
│  │                                                          │        │
│  │  Stage 1: Syntactic Validation                          │        │
│  │    └─ Regex-based ATL pattern matching                  │        │
│  │                                                          │        │
│  │  Stage 2: Semantic Verification (OpenAI)                │        │
│  │    └─ High-temp NL generation, low-temp ATL translation │        │
│  │                                                          │        │
│  │  Stage 3: Cross-Check (Anthropic)                       │        │
│  │    └─ Different model family for verification           │        │
│  └────────────────────────┬───────────────────────────────┘        │
│                           │                                         │
│                           ▼                                         │
│  ┌────────────────────────────────────────────────────────┐        │
│  │                   Output & Reports                       │        │
│  │   - JSONL dataset (unified schema)                      │        │
│  │   - JSON verification reports                           │        │
│  │   - Metrics & monitoring                                │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/omaralmansoori/nl2atl.git
cd nl2atl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
# Parse an ATL formula
from atl_syntax import parse_atl, is_valid

formula = parse_atl("<<agent>>F goal")
print(formula.pretty_print())  # ⟨⟨agent⟩⟩ F goal

# Translate NL to ATL
from nl2atl import translate_nl_to_atl

results = translate_nl_to_atl(
    "The robot can eventually reach the charging station",
    config={"provider": "openai"}
)
print(results)  # ['⟨⟨robot⟩⟩ F charging_station']

# Generate a dataset
from dataset_gen import DatasetGenerator, GenerationConfig

config = GenerationConfig(num_examples=100, use_llm_paraphrasing=False)
generator = DatasetGenerator(config)
pairs = generator.generate()
```

### 🆕 Model Testing & Comparison

**NEW**: Interactive GUI and comprehensive comparison tools for evaluating models.

```bash
# Interactive GUI for real-time testing
python gui_tester.py

# Full model comparison experiment (100 samples)
python full_model_comparison.py --count 100 --verbose

# Or use the quick-start script
./run_experiment.sh 100

# Analyze manual review results
python analyze_review.py data/comparison_for_review_*.json
```

**📖 See [QUICKSTART.md](QUICKSTART.md) for detailed testing instructions.**
**📚 See [TESTING_GUIDE.md](TESTING_GUIDE.md) for complete documentation.**

---

## ✅ Pre-Run Checklist

Before running the verification pipeline with real LLM APIs, complete these checks:

### 1. Environment Setup

```bash
# Verify Python version (3.10+)
python --version

# Check virtual environment is active
which python  # Should point to .venv

# Verify dependencies
pip list | grep -E "openai|anthropic|pyparsing|pyyaml"
```

### 2. API Keys Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit and add your keys
nano .env  # or your preferred editor

# Required keys:
# - OPENAI_API_KEY: For generation (GPT-4o-mini)
# - ANTHROPIC_API_KEY: For cross-verification (Claude)
```

### 3. Validate Configuration Files

```bash
# Check config files exist
ls config/
# Should show: atl_fragment.yaml, pipeline_config.yaml, templates_atl.json

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/pipeline_config.yaml'))"
python -c "import yaml; yaml.safe_load(open('config/atl_fragment.yaml'))"

# Validate JSON syntax
python -c "import json; json.load(open('config/templates_atl.json'))"
```

### 4. Test Mock Mode First

```bash
# Run verification pipeline in mock mode (no API calls)
python verification_pipeline.py --mock

# Generate dataset in mock mode
python dataset_gen.py --num-examples 5 --provider mock --verbose
```

### 5. Check Sample Files

```bash
# Verify sample files exist
ls samples/
# Should show: sample 1.txt, sample 2.txt, sample 3.txt

# Parse and unify samples
python sample_store.py
# Should report "Loaded N samples from M files"
```

### 6. Pre-flight API Test

```bash
# Test OpenAI connection (requires OPENAI_API_KEY)
python -c "
from nl2atl import get_llm_client
client = get_llm_client('openai')
response = client.generate('Say hello', max_tokens=10)
print('OpenAI OK:', response.text)
"

# Test Anthropic connection (requires ANTHROPIC_API_KEY)
python -c "
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model='claude-sonnet-4-20250514',
    max_tokens=10,
    messages=[{'role': 'user', 'content': 'Say hello'}]
)
print('Anthropic OK:', response.content[0].text)
"
```

---

## 📁 Project Structure

```
nl2atl/
├── atl_syntax.py           # ATL parsing, validation, normalization
├── nl2atl.py               # LLM-based NL→ATL translation
├── dataset_gen.py          # Synthetic dataset generation
├── evaluation.py           # Quality metrics and evaluation
├── sample_store.py         # Sample unification and storage
├── verification_pipeline.py # Cross-model verification
├── metrics.py              # Pipeline monitoring
├── config/
│   ├── atl_fragment.yaml   # ATL fragment constraints
│   ├── pipeline_config.yaml # Verification settings
│   └── templates_atl.json  # NL-ATL templates
├── samples/
│   ├── sample 1.txt        # Team sample files
│   ├── sample 2.txt
│   └── sample 3.txt
├── data/
│   ├── unified_samples.json # Parsed samples
│   └── nl_atl_dataset.jsonl # Generated dataset
├── reports/
│   └── verification_*.json  # Verification reports
├── .env.example            # Environment template
├── pyproject.toml          # Project metadata
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 📦 Core Modules

### ATL Syntax (`atl_syntax.py`)

Handles ATL formula representation, parsing, and validation.

```python
from atl_syntax import parse_atl, is_valid, normalize_atl, extract_components

# Parse formula
formula = parse_atl("<<1,2>> G safe")
print(formula.pretty_print())  # ⟨⟨1,2⟩⟩ G safe
print(formula.to_ascii())      # <<1,2>> G safe

# Validate
result = is_valid("<<robot>> F goal")
print(result.valid)    # True
print(result.errors)   # []

# Extract structure
components = extract_components("<<a,b>> G (safe & ready)")
print(components["agents"])     # {'a', 'b'}
print(components["operators"])  # ['G']
print(components["atoms"])      # {'safe', 'ready'}
```

### NL2ATL Translation (`nl2atl.py`)

LLM-powered translation with cross-checking.

```python
from nl2atl import translate_nl_to_atl, critique_nl_atl_pair, paraphrase_nl

# Translate
formulas = translate_nl_to_atl("The robot can always ensure safety")
print(formulas)  # ['⟨⟨robot⟩⟩ G safe']

# Critique a translation
result = critique_nl_atl_pair(
    "The agent can reach the goal",
    "<<agent>> F goal"
)
print(result["ok"])  # True

# Generate paraphrases
paras = paraphrase_nl("The robot can reach the goal", num_paraphrases=3)
```

### Sample Store (`sample_store.py`)

Unified parsing and storage of samples from multiple sources.

```python
from sample_store import SampleStore

store = SampleStore()

# Load from sample files
store.load_from_files(["samples/sample 1.txt", "samples/sample 2.txt"])
print(f"Loaded {len(store.samples)} samples")

# Access unified format
for sample in store.samples[:3]:
    print(f"NL: {sample.nl_statement}")
    print(f"ATL: {sample.atl_formula}")
    print(f"Domain: {sample.domain}")
    print()

# Save unified store
store.save("data/unified_samples.json")
```

### Verification Pipeline (`verification_pipeline.py`)

Three-stage cross-model verification.

```python
from verification_pipeline import VerificationPipeline

# Initialize with config
pipeline = VerificationPipeline("config/pipeline_config.yaml")

# Run verification
results = pipeline.verify_all()

# Generate report
report = pipeline.generate_report()
print(f"Passed: {report['summary']['passed']}")
print(f"Failed: {report['summary']['failed']}")
```

---

## 🔍 Verification Pipeline

The pipeline implements three-stage verification with different LLM providers:

### Stage 1: Syntactic Validation
- Regex-based ATL pattern matching
- No API calls required
- Catches malformed formulas early

### Stage 2: Semantic Verification (OpenAI)
- **Two-temperature approach**:
  - High temperature (0.8) for creative NL paraphrasing
  - Low temperature (0.2) for precise ATL translation
- Validates NL-ATL semantic alignment

### Stage 3: Cross-Model Check (Anthropic)
- Uses Claude (different model family)
- Provides independent verification
- Catches model-specific biases

### Running the Pipeline

```bash
# Mock mode (testing)
python verification_pipeline.py --mock

# Full verification (requires API keys)
python verification_pipeline.py

# With custom config
python verification_pipeline.py --config config/custom_config.yaml
```

---

## ⚙️ Configuration

### Pipeline Configuration (`config/pipeline_config.yaml`)

```yaml
verification:
  stages:
    syntactic: true
    semantic: true
    cross_check: true
  
  providers:
    generation: openai
    verification: anthropic
  
  models:
    openai: gpt-4o-mini
    anthropic: claude-sonnet-4-20250514
  
  temperature:
    nl_generation: 0.8
    atl_translation: 0.2

samples:
  source_files:
    - samples/sample 1.txt
    - samples/sample 2.txt
    - samples/sample 3.txt

output:
  reports_dir: reports
  unified_store: data/unified_samples.json
```

### Environment Variables (`.env`)

```bash
# Required for real verification
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Model configuration
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Temperature settings
NL_GENERATION_TEMPERATURE=0.8
ATL_TRANSLATION_TEMPERATURE=0.2
```

---

## 📚 CLI Reference

### Dataset Generation

```bash
# Generate with templates only (fast, no API)
python dataset_gen.py --num-examples 100 --no-paraphrase --no-crosscheck

# Generate with LLM paraphrasing
python dataset_gen.py --num-examples 500 --provider openai --verbose

# Generate with full pipeline
python dataset_gen.py --num-examples 200 --paraphrases 3 --provider openai
```

### Verification Pipeline

```bash
# Mock mode
python verification_pipeline.py --mock

# Full verification
python verification_pipeline.py

# With specific config
python verification_pipeline.py --config custom.yaml
```

### Evaluation

```bash
# Compute statistics
python evaluation.py stats data/nl_atl_dataset.jsonl --verbose

# Spot-check samples
python evaluation.py spot-check data/nl_atl_dataset.jsonl --count 10

# Full report
python evaluation.py full-report data/nl_atl_dataset.jsonl -o report.json
```

### Sample Store

```bash
# Parse and unify samples
python sample_store.py

# Output shows: "Loaded N samples from M files"
```

---

## 🎯 Next Steps

After completing the pre-run checklist:

### 1. Run Full Verification (with APIs)

```bash
# Set up environment
source .venv/bin/activate
export $(cat .env | xargs)

# Run verification
python verification_pipeline.py
```

### 2. Analyze Results

```bash
# Check verification reports
ls reports/verification_*.json

# View latest report
cat reports/verification_$(ls -t reports/ | head -1)

# Generate evaluation
python evaluation.py full-report data/unified_samples.json -o analysis.json
```

### 3. Generate Production Dataset

```bash
# Generate large dataset with quality checks
python dataset_gen.py \
  --num-examples 1000 \
  --provider openai \
  --paraphrases 3 \
  --verbose \
  --out data/production_dataset.jsonl
```

### 4. Iterate on Failures

```python
# Load report and analyze failures
import json

with open("reports/verification_latest.json") as f:
    report = json.load(f)

# Find failed samples
failures = [r for r in report["results"] if r["status"] == "failed"]
for f in failures[:5]:
    print(f"NL: {f['nl_statement']}")
    print(f"ATL: {f['atl_formula']}")
    print(f"Issues: {f['issues']}")
    print()
```

---

## ✅ Requirements Fulfillment

This implementation addresses all original requirements:

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| **Unify sample syntax** | `sample_store.py` parses 89 samples from 3 files into unified `ATLSample` schema | ✅ |
| **Store for further analysis** | `data/unified_samples.json` with domain classification, components extraction | ✅ |
| **Cross-model verification** | OpenAI for generation, Anthropic for cross-check in `verification_pipeline.py` | ✅ |
| **Two-stage temperature** | High temp (0.8) for NL, low temp (0.2) for ATL in config | ✅ |
| **Syntactic evaluation** | Regex-based `SyntaxValidator` (bypasses pyparsing recursion) | ✅ |
| **LLM semantic verification** | `SemanticVerifier` with GPT-4o-mini | ✅ |
| **Before/after comparison** | Verification reports show original vs verified status | ✅ |
| **Easily configurable** | YAML configs in `config/`, `.env` for API keys | ✅ |
| **Easy to monitor** | `metrics.py` module, JSON reports in `reports/` | ✅ |
| **Consistent output format** | Unified `ATLSample` schema across all modules | ✅ |

### Key Design Decisions

1. **Bypassed pyparsing recursion**: The ATL parser had infinite recursion issues; replaced with regex-based validation for the pipeline while keeping the full parser for non-recursive use cases.

2. **Unified schema**: Combined `NLATLPair` (dataset_gen) and `ATLSample` (sample_store) into a single schema that supports both parsed samples and generated samples.

3. **Mock mode**: All pipelines support `--mock` flag for testing without API calls.

4. **Provider abstraction**: LLM clients are abstracted via `LLMClient` interface, supporting OpenAI, Azure, Anthropic, and mock implementations.

---

## 🛠️ Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black .

# Type checking
mypy .
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_atl_syntax.py -v
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📞 Support

- **Repository**: [GitHub](https://github.com/omaralmansoori/nl2atl)
- **Issues**: [GitHub Issues](https://github.com/omaralmansoori/nl2atl/issues)
