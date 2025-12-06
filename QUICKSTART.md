# Quick Start: Model Comparison Experiment

## What You Now Have

Three new powerful tools for testing and comparing your fine-tuned NL2ATL model:

### 1. 🖥️ Interactive GUI (`gui_tester.py`)
- Real-time testing with visual interface
- Side-by-side comparison of all three models
- Response time tracking
- Export test results

### 2. 🔬 Full Comparison Script (`full_model_comparison.py`)
- Generate 100 NL statements automatically
- Translate with all models (OpenAI base, fine-tuned, Claude)
- Track performance metrics (response time, tokens)
- Create structured output for manual review

### 3. 📊 Review Analysis (`analyze_review.py`)
- Analyze manual review results
- Calculate accuracy statistics
- Compare model performance

---

## Quick Start (3 Steps)

### Step 1: Set up environment variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a `.env` file with these keys.

### Step 2: Run the comparison experiment

**Option A: Use the automated script**
```bash
./run_experiment.sh 100
```

**Option B: Run manually**
```bash
python full_model_comparison.py --count 100 --verbose
```

This will generate 100 NL statements and translate each with all three models, saving results to the `data/` directory.

### Step 3: Perform manual review

1. Open the generated review file:
   ```bash
   code data/comparison_for_review_*.json
   ```

2. For each sample, fill in the review fields:
   ```json
   "review": {
       "openai_base_correct": true,      // ← Set to true/false
       "openai_finetuned_correct": true, // ← Set to true/false
       "claude_correct": false,          // ← Set to true/false
       "notes": "Claude used wrong operator"
   }
   ```

3. Save the file

4. Analyze results:
   ```bash
   python analyze_review.py data/comparison_for_review_*.json
   ```

---

## Interactive Testing (Alternative)

For quick, interactive testing:

```bash
python gui_tester.py
```

Features:
- Enter NL requirements manually or load examples
- See real-time translations from all models
- Compare response times and syntax validity
- Export test history

---

## Output Files Explained

After running the comparison:

| File | Purpose |
|------|---------|
| `comparison_nl_*.json` | Generated NL statements |
| `comparison_raw_*.jsonl` | Complete raw data with all details |
| `comparison_for_review_*.json` | **Main file for manual review** |
| `comparison_stats_*.json` | Statistical summary |
| `comparison_report_*.md` | Human-readable report |

---

## Models Being Compared

| Model | Purpose |
|-------|---------|
| GPT-4o-mini (Base) | Baseline performance |
| GPT-4o-mini (Fine-tuned) | Your fine-tuned model |
| Claude 3.5 Sonnet | Alternative state-of-the-art |

---

## Metrics Tracked

### Automatic Metrics
- ✅ **Response Time**: How fast each model responds
- ✅ **Token Usage**: API cost estimation
- ✅ **Syntax Validity**: Whether ATL formula parses correctly

### Manual Metrics (You provide)
- ✅ **Semantic Correctness**: Does it mean the right thing?
- ✅ **Notes**: Qualitative observations

---

## Example Analysis Output

After reviewing and running `analyze_review.py`:

```
======================================================================
MANUAL REVIEW ANALYSIS
======================================================================

Total Samples: 100

----------------------------------------------------------------------
ACCURACY (Based on Manual Review)
----------------------------------------------------------------------
Model                          Correct      Incorrect    Accuracy    
----------------------------------------------------------------------
GPT-4o-mini (Base)            82/100       18           82.0%       
GPT-4o-mini (Fine-tuned)      95/100       5            95.0%       
Claude 3.5 Sonnet             88/100       12           88.0%       

----------------------------------------------------------------------
PERFORMANCE METRICS
----------------------------------------------------------------------
Model                          Avg Response Time    Syntax Valid   
----------------------------------------------------------------------
GPT-4o-mini (Base)            0.542s               98/100         
GPT-4o-mini (Fine-tuned)      0.321s               100/100        
Claude 3.5 Sonnet             0.687s               97/100         

KEY INSIGHTS:
----------------------------------------------------------------------
• Highest accuracy: GPT-4o-mini (Fine-tuned) (95.0%)
• Fastest model: GPT-4o-mini (Fine-tuned) (0.321s avg)
```

---

## Tips for Success

### For Manual Review:
1. **Be consistent** - Define "correct" before you start
2. **Take breaks** - Reviewing 100 samples takes focus
3. **Note patterns** - Document common error types
4. **Consider context** - Some domains may be harder than others

### For Best Results:
1. **Run multiple trials** - Statistical significance matters
2. **Vary domains** - Test across different application areas
3. **Check edge cases** - Complex nested formulas, negations, etc.
4. **Document everything** - Use the notes field liberally

---

## Troubleshooting

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"API key not set"**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**GUI won't launch (macOS)**
```bash
brew install python-tk
```

**GUI won't launch (Linux)**
```bash
sudo apt-get install python3-tk
```

---

## Next Steps

After completing your experiment:

1. **Calculate statistical significance**
   - Use paired t-tests for response time
   - Use McNemar's test for accuracy differences

2. **Create visualizations**
   - Plot accuracy by domain
   - Show response time distributions
   - Visualize error patterns

3. **Write up results**
   - Use the generated markdown report as a starting point
   - Include sample translations (good and bad)
   - Discuss implications

---

## Need Help?

- 📖 **Full documentation**: See `TESTING_GUIDE.md`
- 🐛 **Issues**: Check error messages carefully
- 💡 **Ideas**: The tools are extensible - add what you need!

---

**Happy Testing! 🚀**
