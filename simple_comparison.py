#!/usr/bin/env python3
"""
Simple Model Comparison Script
===============================

Generates 100 NL statements (50 from OpenAI, 50 from Anthropic)
Translates each with 3 models (OpenAI base, fine-tuned, Claude)
Saves all results to JSON.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    "openai_base": "gpt-4o-mini-2024-07-18",
    "openai_finetuned": "ft:gpt-4o-mini-2024-07-18:personal:nl2atl-v1:CiGKGvnC",
    "claude": "claude-3-haiku-20240307"  # Using Haiku (Sonnet not available with current API key)
}

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
# NL Generation
# =============================================================================

def generate_nl_with_openai() -> str:
    """Generate one NL statement using OpenAI."""
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": NL_GEN_PROMPT}],
        temperature=0.9,
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()


def generate_nl_with_anthropic() -> str:
    """Generate one NL statement using Anthropic."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",  # Using Haiku (Sonnet not available)
        max_tokens=500,
        temperature=0.7,
        messages=[{"role": "user", "content": NL_GEN_PROMPT}]
    )
    
    return message.content[0].text.strip()


def generate_nl_statements(count: int) -> List[Dict[str, Any]]:
    """
    Generate NL statements 50/50 from OpenAI and Anthropic.
    
    Returns list of dicts with 'nl' and 'generator' fields.
    """
    statements = []
    openai_count = count // 2
    anthropic_count = count - openai_count
    
    print(f"\nGenerating {count} NL statements:")
    print(f"  - {openai_count} from OpenAI")
    print(f"  - {anthropic_count} from Anthropic")
    print()
    
    # Generate from OpenAI
    for i in range(openai_count):
        try:
            print(f"[{i+1}/{count}] Generating with OpenAI...", end=" ")
            nl = generate_nl_with_openai()
            statements.append({
                "sample_id": f"sample_{i+1:03d}",
                "nl": nl,
                "generator": "openai"
            })
            print(f"✓ {nl[:60]}...")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Generate from Anthropic
    for i in range(anthropic_count):
        idx = openai_count + i
        try:
            print(f"[{idx+1}/{count}] Generating with Anthropic...", end=" ")
            nl = generate_nl_with_anthropic()
            statements.append({
                "sample_id": f"sample_{idx+1:03d}",
                "nl": nl,
                "generator": "anthropic"
            })
            print(f"✓ {nl[:60]}...")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return statements


# =============================================================================
# Translation
# =============================================================================

def translate_with_openai(nl_text: str, model_id: str) -> Dict[str, Any]:
    """Translate using OpenAI with metrics."""
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": TRANSLATION_PROMPT.format(nl_text=nl_text)}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        atl_formula = response.choices[0].message.content.strip()
        
        # Clean up formula
        for prefix in ["ATL:", "Formula:", "Translation:"]:
            if atl_formula.startswith(prefix):
                atl_formula = atl_formula[len(prefix):].strip()
        
        return {
            "atl_formula": atl_formula,
            "response_time": time.time() - start_time,
            "tokens_used": response.usage.total_tokens,
            "error": None
        }
    except Exception as e:
        return {
            "atl_formula": "",
            "response_time": time.time() - start_time,
            "tokens_used": 0,
            "error": str(e)
        }


def translate_with_anthropic(nl_text: str, model_id: str) -> Dict[str, Any]:
    """Translate using Anthropic with metrics."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    start_time = time.time()
    
    try:
        message = client.messages.create(
            model=model_id,
            max_tokens=200,
            temperature=0.1,
            messages=[
                {"role": "user", "content": TRANSLATION_PROMPT.format(nl_text=nl_text)}
            ]
        )
        
        atl_formula = message.content[0].text.strip()
        
        # Clean up formula
        for prefix in ["ATL:", "Formula:", "Translation:"]:
            if atl_formula.startswith(prefix):
                atl_formula = atl_formula[len(prefix):].strip()
        
        return {
            "atl_formula": atl_formula,
            "response_time": time.time() - start_time,
            "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
            "error": None
        }
    except Exception as e:
        return {
            "atl_formula": "",
            "response_time": time.time() - start_time,
            "tokens_used": 0,
            "error": str(e)
        }


def translate_all_models(nl_text: str) -> Dict[str, Any]:
    """Translate one NL statement with all 3 models."""
    results = {}
    
    # OpenAI base
    print("    OpenAI base...", end=" ")
    results["openai_base"] = translate_with_openai(nl_text, MODELS["openai_base"])
    print(f"✓ {results['openai_base']['response_time']:.2f}s")
    
    # OpenAI fine-tuned
    print("    OpenAI fine-tuned...", end=" ")
    results["openai_finetuned"] = translate_with_openai(nl_text, MODELS["openai_finetuned"])
    print(f"✓ {results['openai_finetuned']['response_time']:.2f}s")
    
    # Claude
    print("    Claude...", end=" ")
    results["claude"] = translate_with_anthropic(nl_text, MODELS["claude"])
    print(f"✓ {results['claude']['response_time']:.2f}s")
    
    return results


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("SIMPLE NL2ATL MODEL COMPARISON")
    print("=" * 70)
    
    # Check API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Generate NL statements
    print("\n" + "=" * 70)
    print("STEP 1: Generate NL Statements")
    print("=" * 70)
    
    nl_statements = generate_nl_statements(100)
    print(f"\n✓ Generated {len(nl_statements)} NL statements")
    
    # Save NL statements
    nl_file = f"data/nl_statements_{timestamp}.json"
    Path("data").mkdir(exist_ok=True)
    with open(nl_file, 'w') as f:
        json.dump(nl_statements, f, indent=2)
    print(f"✓ Saved to {nl_file}")
    
    # Step 2: Translate with all models
    print("\n" + "=" * 70)
    print("STEP 2: Translate with All Models")
    print("=" * 70)
    
    results = []
    
    for i, stmt in enumerate(nl_statements, 1):
        print(f"\n[{i}/{len(nl_statements)}] {stmt['nl'][:60]}...")
        
        translations = translate_all_models(stmt['nl'])
        
        result = {
            "sample_id": stmt["sample_id"],
            "nl_statement": stmt["nl"],
            "nl_generator": stmt["generator"],
            "translations": translations,
            "timestamp": datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Save incrementally
        results_file = f"data/comparison_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Calculate statistics
    stats = {
        "openai_base": {"total_time": 0, "total_tokens": 0, "count": 0},
        "openai_finetuned": {"total_time": 0, "total_tokens": 0, "count": 0},
        "claude": {"total_time": 0, "total_tokens": 0, "count": 0}
    }
    
    for result in results:
        for model_key, trans in result["translations"].items():
            if not trans["error"]:
                stats[model_key]["total_time"] += trans["response_time"]
                stats[model_key]["total_tokens"] += trans["tokens_used"]
                stats[model_key]["count"] += 1
    
    print(f"\nTotal samples: {len(results)}")
    print("\nAverage Response Times:")
    for model_key, s in stats.items():
        if s["count"] > 0:
            avg_time = s["total_time"] / s["count"]
            avg_tokens = s["total_tokens"] / s["count"]
            print(f"  {model_key}: {avg_time:.3f}s (avg {avg_tokens:.0f} tokens)")
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
