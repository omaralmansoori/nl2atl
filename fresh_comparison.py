#!/usr/bin/env python3
"""
Fresh Model Comparison Script
==============================

Generates 100 NL statements (50 from OpenAI, 50 from Claude Haiku)
Translates each with 3 models (OpenAI base, fine-tuned, Claude Haiku)
Saves all results to JSON with clear progress tracking.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    "openai_base": "gpt-4o-mini-2024-07-18",
    "openai_finetuned": "ft:gpt-4o-mini-2024-07-18:personal:nl2atl-v1:CiGKGvnC",
    "claude": "claude-3-haiku-20240307"
}

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
NL_FILE = OUTPUT_DIR / f"nl_statements_{TIMESTAMP}.json"
RESULTS_FILE = OUTPUT_DIR / f"comparison_results_{TIMESTAMP}.json"

NL_GEN_PROMPT = """Generate a natural language requirement that describes what some agents can guarantee or achieve in a temporal logic setting. 

The requirement should:
- Mention specific agent(s) or coalition
- Express a temporal property (always, eventually, until, next, etc.)
- Be clear and unambiguous
- Use realistic domain vocabulary

Example domains: healthcare, smart home, autonomous vehicles, supply chain, financial systems, energy grid, air traffic control, manufacturing

Generate ONE requirement only, no explanations."""

TRANSLATION_PROMPT = """You are an expert in translating natural language requirements into Alternating-time Temporal Logic (ATL) formulas. Given a natural language statement describing agent capabilities and temporal properties, generate the corresponding ATL formula using proper syntax with coalition operators ⟨⟨...⟩⟩, temporal operators (G, F, X, U), and logical operators (∧, ∨, →, ¬).

Translate to ATL: {nl_text}

Respond with ONLY the ATL formula, nothing else."""


# =============================================================================
# NL Generation Functions
# =============================================================================

def generate_nl_with_openai() -> str:
    """Generate NL statement using OpenAI."""
    import openai
    
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": NL_GEN_PROMPT}],
        max_tokens=500,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()


def generate_nl_with_anthropic() -> str:
    """Generate NL statement using Anthropic Claude."""
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        temperature=0.7,
        messages=[{"role": "user", "content": NL_GEN_PROMPT}]
    )
    
    return message.content[0].text.strip()


# =============================================================================
# Translation Functions
# =============================================================================

def translate_with_openai(nl_text: str, model_id: str) -> dict:
    """Translate using OpenAI."""
    import openai
    
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an expert in ATL (Alternating-time Temporal Logic) translation."},
                {"role": "user", "content": TRANSLATION_PROMPT.format(nl_text=nl_text)}
            ],
            temperature=0.1,
            max_tokens=200
        )
        response_time = time.time() - start_time
        
        atl_formula = response.choices[0].message.content.strip()
        
        # Clean up formula
        if ":" in atl_formula:
            atl_formula = atl_formula.split(":", 1)[1].strip()
        if "```" in atl_formula:
            atl_formula = atl_formula.split("```")[1].strip()
        
        return {
            "atl_formula": atl_formula,
            "response_time": response_time,
            "tokens_used": response.usage.total_tokens,
            "error": None
        }
    except Exception as e:
        return {
            "atl_formula": "",
            "response_time": 0.0,
            "tokens_used": 0,
            "error": str(e)
        }


def translate_with_anthropic(nl_text: str, model_id: str) -> dict:
    """Translate using Anthropic Claude."""
    import anthropic
    
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        start_time = time.time()
        message = client.messages.create(
            model=model_id,
            max_tokens=200,
            temperature=0.1,
            system="You are an expert in ATL (Alternating-time Temporal Logic) translation.",
            messages=[{"role": "user", "content": TRANSLATION_PROMPT.format(nl_text=nl_text)}]
        )
        response_time = time.time() - start_time
        
        atl_formula = message.content[0].text.strip()
        
        # Clean up formula
        if ":" in atl_formula:
            atl_formula = atl_formula.split(":", 1)[1].strip()
        if "```" in atl_formula:
            atl_formula = atl_formula.split("```")[1].strip()
        
        return {
            "atl_formula": atl_formula,
            "response_time": response_time,
            "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
            "error": None
        }
    except Exception as e:
        return {
            "atl_formula": "",
            "response_time": 0.0,
            "tokens_used": 0,
            "error": str(e)
        }


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("NL2ATL Model Comparison - Fresh Start")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  NL Statements: {NL_FILE}")
    print(f"  Results:       {RESULTS_FILE}")
    print()
    
    # Step 1: Generate NL statements
    print("STEP 1: Generating 100 NL Statements (50 OpenAI + 50 Claude)")
    print("-" * 70)
    
    nl_statements = []
    
    # Generate 50 from OpenAI
    for i in range(1, 51):
        try:
            nl = generate_nl_with_openai()
            nl_statements.append({
                "sample_id": f"sample_{i:03d}",
                "nl": nl,
                "generator": "openai"
            })
            print(f"[{i}/100] OpenAI    ✓ {nl[:60]}...")
        except Exception as e:
            print(f"[{i}/100] OpenAI    ✗ Error: {e}")
            nl_statements.append({
                "sample_id": f"sample_{i:03d}",
                "nl": "",
                "generator": "openai",
                "error": str(e)
            })
        
        # Save incrementally
        with open(NL_FILE, 'w') as f:
            json.dump(nl_statements, f, indent=2)
    
    # Generate 50 from Anthropic
    for i in range(51, 101):
        try:
            nl = generate_nl_with_anthropic()
            nl_statements.append({
                "sample_id": f"sample_{i:03d}",
                "nl": nl,
                "generator": "anthropic"
            })
            print(f"[{i}/100] Anthropic ✓ {nl[:60]}...")
        except Exception as e:
            print(f"[{i}/100] Anthropic ✗ Error: {e}")
            nl_statements.append({
                "sample_id": f"sample_{i:03d}",
                "nl": "",
                "generator": "anthropic",
                "error": str(e)
            })
        
        # Save incrementally
        with open(NL_FILE, 'w') as f:
            json.dump(nl_statements, f, indent=2)
    
    print(f"\n✓ NL generation complete! Saved to {NL_FILE}\n")
    
    # Step 2: Translate with all models
    print("STEP 2: Translating with 3 Models (300 total translations)")
    print("-" * 70)
    
    results = []
    
    for idx, statement in enumerate(nl_statements, 1):
        if "error" in statement or not statement.get("nl"):
            print(f"[{idx}/100] Skipping (no NL statement)")
            continue
        
        nl_text = statement["nl"]
        sample_id = statement["sample_id"]
        
        print(f"\n[{idx}/100] Translating: {nl_text[:50]}...")
        
        translations = {}
        
        # Translate with OpenAI base
        print(f"  → OpenAI Base...", end=" ", flush=True)
        result = translate_with_openai(nl_text, MODELS["openai_base"])
        translations["openai_base"] = result
        if result["error"]:
            print(f"✗ {result['error'][:40]}")
        else:
            print(f"✓ ({result['response_time']:.2f}s)")
        
        # Translate with fine-tuned
        print(f"  → OpenAI Fine-tuned...", end=" ", flush=True)
        result = translate_with_openai(nl_text, MODELS["openai_finetuned"])
        translations["openai_finetuned"] = result
        if result["error"]:
            print(f"✗ {result['error'][:40]}")
        else:
            print(f"✓ ({result['response_time']:.2f}s)")
        
        # Translate with Claude
        print(f"  → Claude Haiku...", end=" ", flush=True)
        result = translate_with_anthropic(nl_text, MODELS["claude"])
        translations["claude"] = result
        if result["error"]:
            print(f"✗ {result['error'][:40]}")
        else:
            print(f"✓ ({result['response_time']:.2f}s)")
        
        # Save result
        results.append({
            "sample_id": sample_id,
            "nl_statement": nl_text,
            "nl_generator": statement["generator"],
            "translations": translations,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save incrementally
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"✓ All translations complete!")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"\nNext steps:")
    print(f"  1. Review the results manually")
    print(f"  2. Mark each translation as correct/incorrect")
    print(f"  3. Run analyze_review.py to calculate accuracy metrics")
    print()


if __name__ == "__main__":
    main()
