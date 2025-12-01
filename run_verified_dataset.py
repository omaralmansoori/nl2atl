#!/usr/bin/env python3
"""
Cross-Verified NL-ATL Dataset Generator with Incremental Saving.

Generates NL-ATL pairs across 4 provider combinations with:
- Incremental saving after each sample (no data loss on interrupt)
- Signal handling for graceful shutdown
- Real-time progress and critique data
"""

import os
import sys
import json
import re
import signal
import atexit
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Load environment
from dotenv import load_dotenv
load_dotenv()

# LLM clients
import openai
import anthropic

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLES_PER_CONFIG = 5  # 5 samples per combo = 20 total for testing

DOMAINS = [
    "autonomous_systems",
    "distributed_systems",
    "safety_critical",
    "robotics",
    "access_control",
    "industrial_control",
    "networking",
    "transaction_processing",
]

PATTERNS = ["safety", "liveness", "response", "until", "fairness"]

CONFIGS = [
    {"name": "OpenAI→Anthropic", "generator": "openai", "verifier": "anthropic"},
    {"name": "Anthropic→OpenAI", "generator": "anthropic", "verifier": "openai"},
    {"name": "Anthropic→Anthropic", "generator": "anthropic", "verifier": "anthropic"},
    {"name": "OpenAI→OpenAI", "generator": "openai", "verifier": "openai"},
]

# ============================================================================
# GLOBAL STATE FOR INCREMENTAL SAVING
# ============================================================================

@dataclass
class RunState:
    """Tracks run state for incremental saving."""
    run_id: str = ""
    start_time: str = ""
    samples: list = field(default_factory=list)
    config_stats: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    interrupted: bool = False
    
    def save(self):
        """Save current state to disk."""
        if not self.run_id:
            return None, None
        
        # Save samples in JSONL format (one per line)
        samples_file = Path(f"data/verified_{SAMPLES_PER_CONFIG * 4}_{self.run_id}.jsonl")
        samples_file.parent.mkdir(exist_ok=True)
        with open(samples_file, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")
        
        # Save unified JSON format (like unified_samples.json)
        unified_file = Path(f"data/verified_unified_{self.run_id}.json")
        unified_data = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "samples": self.samples,
            "stats": self._compute_unified_stats(),
        }
        with open(unified_file, "w") as f:
            json.dump(unified_data, f, indent=2)
        
        # Save full report with metadata
        report = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "interrupted": self.interrupted,
            "config": {
                "samples_per_config": SAMPLES_PER_CONFIG,
                "total_expected": SAMPLES_PER_CONFIG * 4,
            },
            "stats": self._compute_stats(),
            "errors": self.errors,
        }
        
        report_file = Path(f"reports/verified_dataset_{self.run_id}.json")
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        return samples_file, report_file
    
    def _compute_stats(self):
        """Compute statistics from samples (for report)."""
        stats = {
            "total": len(self.samples),
            "verified": 0,
            "rejected": 0,
            "needs_review": 0,
            "by_config": defaultdict(lambda: {"total": 0, "verified": 0}),
            "by_domain": defaultdict(lambda: {"total": 0, "verified": 0}),
            "issues": defaultdict(int),
        }
        
        for s in self.samples:
            config_name = s.get("config_name", "unknown")
            domain = s.get("domain", "unknown")
            status = s.get("verification_status", "unknown")
            
            stats["by_config"][config_name]["total"] += 1
            stats["by_domain"][domain]["total"] += 1
            
            if status == "verified":
                stats["verified"] += 1
                stats["by_config"][config_name]["verified"] += 1
                stats["by_domain"][domain]["verified"] += 1
            elif status == "rejected":
                stats["rejected"] += 1
                if s.get("rejection_reasons"):
                    for reason in s["rejection_reasons"][:2]:  # Top 2 reasons
                        stats["issues"][reason[:80]] += 1
            else:
                stats["needs_review"] += 1
        
        # Convert defaultdicts to regular dicts for JSON
        stats["by_config"] = dict(stats["by_config"])
        stats["by_domain"] = dict(stats["by_domain"])
        stats["issues"] = dict(stats["issues"])
        
        return stats
    
    def _compute_unified_stats(self):
        """Compute statistics matching unified_samples.json format."""
        all_agents = []
        all_atoms = []
        all_operators = []
        domain_counts = Counter()
        config_counts = Counter()
        verified_count = 0
        syntax_valid_count = 0
        complexity_scores = []
        
        for s in self.samples:
            domain_counts[s.get("domain", "unknown")] += 1
            config_counts[s.get("config_name", "unknown")] += 1
            
            if s.get("verification_status") == "verified":
                verified_count += 1
            if not s.get("syntax_errors"):
                syntax_valid_count += 1
            
            all_agents.extend(s.get("agents", []))
            all_atoms.extend(s.get("atoms", []))
            all_operators.extend(s.get("operators", []))
            
            # Simple complexity = number of operators + agents
            complexity = len(s.get("operators", [])) + len(s.get("agents", []))
            complexity_scores.append(complexity)
        
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        
        return {
            "total_samples": len(self.samples),
            "unique_domains": len(domain_counts),
            "domain_distribution": dict(domain_counts),
            "config_distribution": dict(config_counts),
            "verified_count": verified_count,
            "syntax_valid_count": syntax_valid_count,
            "unique_agents": len(set(all_agents)),
            "unique_atoms": len(set(all_atoms)),
            "avg_complexity": round(avg_complexity, 2),
            "operator_distribution": dict(Counter(all_operators)),
        }


# Global state
STATE = RunState()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\n\n⚠️  INTERRUPT RECEIVED - Saving current progress...")
    STATE.interrupted = True
    samples_file, report_file = STATE.save()
    if samples_file:
        print(f"✅ Saved {len(STATE.samples)} samples to: {samples_file}")
        print(f"✅ Report saved to: {report_file}")
    print("\n📊 Partial Results:")
    print_summary()
    sys.exit(0)


def print_summary():
    """Print current summary."""
    stats = STATE._compute_stats()
    total = stats["total"]
    if total == 0:
        print("  No samples generated yet.")
        return
    
    verified = stats["verified"]
    rejected = stats["rejected"]
    needs_review = stats["needs_review"]
    
    print(f"  Total: {total} | ✅ Verified: {verified} | ❌ Rejected: {rejected} | ⚠️ Review: {needs_review}")
    
    if stats["issues"]:
        print("\n  Top Issues:")
        for issue, count in list(stats["issues"].items())[:5]:
            print(f"    [{count}x] {issue}...")


# Register handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(lambda: STATE.save() if STATE.samples else None)


# ============================================================================
# LLM CLIENTS
# ============================================================================

class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
    
    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()


class AnthropicClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-20250514"
    
    def generate(self, prompt: str, system: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system if system else "You are an expert in formal verification and temporal logic.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


def get_client(provider: str):
    if provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# ATL VALIDATION AND PARSING
# ============================================================================

def extract_components(formula: str) -> dict:
    """Extract agents, operators, and atoms from ATL formula."""
    components = {
        "agents": [],
        "operators": [],
        "atoms": [],
    }
    
    # Extract agents from <<...>>
    coalition_match = re.search(r'<<([^>]+)>>', formula)
    if coalition_match:
        agents_str = coalition_match.group(1)
        components["agents"] = [a.strip() for a in agents_str.split(',')]
    
    # Extract temporal operators
    for op in ['G', 'F', 'X', 'U', 'W', 'R']:
        if op in formula:
            components["operators"].append(op)
    
    # Extract logical operators
    if '&' in formula:
        components["operators"].append('&')
    if '|' in formula:
        components["operators"].append('|')
    if '->' in formula:
        components["operators"].append('->')
    if '!' in formula:
        components["operators"].append('!')
    
    # Extract atoms (identifiers that are not operators or keywords)
    # Simple heuristic: lowercase words/underscored identifiers
    atom_pattern = r'\b[a-z][a-z0-9_]*\b'
    potential_atoms = re.findall(atom_pattern, formula)
    # Filter out known operators and keywords
    reserved = {'true', 'false', 'and', 'or', 'not'}
    components["atoms"] = [a for a in potential_atoms if a not in reserved]
    
    return components

def formula_to_unicode(formula: str) -> str:
    """Convert ATL formula to Unicode representation."""
    unicode_formula = formula
    unicode_formula = unicode_formula.replace('<<', '⟨⟨')
    unicode_formula = unicode_formula.replace('>>', '⟩⟩')
    unicode_formula = unicode_formula.replace('->', '→')
    unicode_formula = unicode_formula.replace('!', '¬')
    unicode_formula = unicode_formula.replace('&', '∧')
    unicode_formula = unicode_formula.replace('|', '∨')
    return unicode_formula

def generate_sample_id(nl: str, atl: str) -> str:
    """Generate a unique 12-character hex ID for a sample."""
    content = f"{nl}|{atl}"
    hash_obj = hashlib.md5(content.encode())
    return hash_obj.hexdigest()[:12]

def validate_atl_syntax(formula: str) -> tuple[bool, list[str]]:
    """Validate ATL formula syntax using regex patterns."""
    errors = []
    
    # Clean formula
    formula = formula.strip()
    if not formula:
        return False, ["Empty formula"]
    
    # Check for coalition operators
    coalition_pattern = r'<<[^>]+>>'
    if not re.search(coalition_pattern, formula):
        errors.append("No coalition operator <<agents>> found")
    
    # Check for temporal operators
    temporal_ops = ['F', 'G', 'X', 'U', 'W', 'R']
    has_temporal = any(op in formula for op in temporal_ops)
    if not has_temporal:
        errors.append("No temporal operators found")
    
    # Check bracket balance
    if formula.count('(') != formula.count(')'):
        errors.append("Unbalanced parentheses")
    if formula.count('[') != formula.count(']'):
        errors.append("Unbalanced brackets")
    
    # Check angle brackets (exclude -> implication arrows)
    formula_no_arrows = formula.replace('->', '').replace('<-', '')
    if formula_no_arrows.count('<') != formula_no_arrows.count('>'):
        errors.append("Unbalanced angle brackets")
    
    return len(errors) == 0, errors


# ============================================================================
# GENERATION AND VERIFICATION
# ============================================================================

def generate_nl(client, domain: str, pattern: str) -> Optional[str]:
    """Generate a natural language statement."""
    prompt = f"""Generate a natural language statement describing a multi-agent system requirement.

Domain: {domain}
Pattern type: {pattern}

Requirements for the statement:
1. Involve exactly 2 agents with clear, simple roles (e.g., Robot, Controller, Sensor, Monitor)
2. Express a clear temporal property using simple terms:
   - "always" for invariants
   - "eventually" for liveness  
   - "if X then eventually Y" for response
   - "X until Y" for until properties
3. Keep the requirement SIMPLE and directly translatable to ATL
4. Do NOT include specific time constraints (no "within 5 seconds")
5. Use clear cause-effect relationships

Good example: "The Robot must always ensure that if an obstacle is detected, it eventually stops moving."
Bad example: "The Robot must notify within 5 seconds while coordinating with 3 other agents..."

Respond with ONLY the natural language statement, no explanations."""

    system = "You are an expert in multi-agent systems. Generate simple, clear requirements."
    return client.generate(prompt, system)


def translate_to_atl(client, nl_statement: str, domain: str) -> Optional[str]:
    """Translate NL to ATL formula."""
    prompt = f"""Translate this natural language statement to an ATL formula.

Statement: {nl_statement}

ATL Syntax Rules:
- Coalition operator: <<agent1, agent2>> means "agents can cooperate to ensure"
- G(φ) = "always φ" (globally)
- F(φ) = "eventually φ" (finally)  
- X(φ) = "next state φ"
- φ U ψ = "φ until ψ"
- Logical: & (and), | (or), -> (implies), ! (not)
- Propositions: lowercase_with_underscores

Pattern examples:
- Safety (always avoid bad): <<agents>> G(!bad_state)
- Liveness (eventually good): <<agents>> F(good_state)
- Response (if P then eventually Q): <<agents>> G(trigger -> F(response))
- Until (P holds until Q): <<agents>> (condition U goal)

IMPORTANT:
- Use ONLY the agents mentioned in the statement
- Keep the formula simple and direct
- Match the temporal pattern to the statement's intent

Respond with ONLY the ATL formula, nothing else."""

    system = "You are an ATL formula expert. Output only valid ATL syntax."
    return client.generate(prompt, system)


def verify_pair(client, nl: str, atl: str, domain: str) -> dict:
    """Verify NL-ATL pair for semantic correctness."""
    prompt = f"""Verify if this ATL formula correctly captures the natural language requirement.

Natural Language: {nl}

ATL Formula: {atl}

Verification Checklist:
1. Are the correct agents in the coalition <<...>>?
2. Does the temporal operator match the requirement?
   - "always" -> G
   - "eventually" -> F
   - "until" -> U
   - "if...then eventually" -> G(... -> F(...))
3. Are the propositions reasonable representations of the concepts?

Scoring:
- ACCEPT: Formula correctly captures the core meaning (minor naming differences OK)
- REJECT: Formula has wrong temporal structure or missing key concepts
- NEEDS_REVIEW: Partial match, some aspects unclear

Be lenient on proposition names. Focus on structural correctness.

Respond in JSON:
{{"verdict": "ACCEPT" or "REJECT" or "NEEDS_REVIEW", "confidence": 0.0-1.0, "issues": [], "explanation": "brief"}}"""

    system = "You verify ATL formulas. Be reasonable - accept formulas that capture the core meaning."
    
    try:
        response = client.generate(prompt, system)
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"verdict": "NEEDS_REVIEW", "confidence": 0.5, "issues": ["Could not parse response"], "explanation": response[:200]}
    except Exception as e:
        return {"verdict": "NEEDS_REVIEW", "confidence": 0.0, "issues": [str(e)], "explanation": "Verification failed"}


def generate_sample(generator_client, verifier_client, domain: str, pattern: str, config_name: str) -> Optional[dict]:
    """Generate and verify a single sample."""
    # Initialize with metadata (will be transformed later)
    metadata = {
        "config_name": config_name,
        "generator": type(generator_client).__name__,
        "verifier": type(verifier_client).__name__,
        "domain": domain,
        "pattern": pattern,
    }
    
    # Generate NL
    try:
        nl = generate_nl(generator_client, domain, pattern)
        if not nl:
            return {**metadata, "error": "NL generation returned empty"}
        print("NL✓ ", end="", flush=True)
    except Exception as e:
        print(f"❌ NL failed: {e}")
        return {**metadata, "error": f"NL generation error: {e}"}
    
    # Translate to ATL
    try:
        atl = translate_to_atl(generator_client, nl, domain)
        if not atl:
            return {**metadata, "nl_statement": nl, "error": "ATL translation returned empty"}
        
        # Validate syntax
        valid, syntax_errors = validate_atl_syntax(atl)
        if not valid:
            sample_id = generate_sample_id(nl, atl)
            components = extract_components(atl)
            print(f"❌ Invalid syntax: {syntax_errors}")
            return {
                "id": sample_id,
                "nl_statement": nl,
                "atl_formula": atl,
                "atl_unicode": formula_to_unicode(atl),
                "domain": domain,
                "source_file": f"verified_{config_name}",
                "agents": components["agents"],
                "operators": components["operators"],
                "atoms": components["atoms"],
                "verification_notes": syntax_errors,
                "verification_status": "rejected",
                "rejection_reasons": syntax_errors,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata,
            }
        
        print("ATL✓ ", end="", flush=True)
    except Exception as e:
        print(f"❌ ATL failed: {e}")
        return {**metadata, "nl_statement": nl, "error": f"ATL translation error: {e}"}
    
    # Extract components
    components = extract_components(atl)
    sample_id = generate_sample_id(nl, atl)
    
    # Verify with cross-model
    verification_notes = []
    verification_status = "verified"
    rejection_reasons = []
    
    try:
        verification = verify_pair(verifier_client, nl, atl, domain)
        
        verdict = verification.get("verdict", "NEEDS_REVIEW").upper()
        confidence = verification.get("confidence", 0.5)
        explanation = verification.get("explanation", "")
        
        verification_notes.append(f"Verifier: {metadata['verifier']}")
        verification_notes.append(f"Verdict: {verdict} (confidence: {confidence})")
        if explanation:
            verification_notes.append(f"Explanation: {explanation}")
        
        if verdict == "ACCEPT":
            verification_status = "verified"
            print("✅")
        elif verdict == "REJECT":
            verification_status = "rejected"
            rejection_reasons = verification.get("issues", [])
            print("❌")
        else:
            verification_status = "needs_review"
            print("⚠️")
    except Exception as e:
        verification_status = "needs_review"
        verification_notes.append(f"Verification error: {e}")
        print(f"⚠️ Verify error: {e}")
    
    # Build sample in unified format
    sample = {
        "id": sample_id,
        "nl_statement": nl,
        "atl_formula": atl,
        "atl_unicode": formula_to_unicode(atl),
        "domain": domain,
        "source_file": f"verified_{config_name}",
        "agents": components["agents"],
        "operators": components["operators"],
        "atoms": components["atoms"],
        "verification_notes": verification_notes,
        "verification_status": verification_status,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata,
    }
    
    if rejection_reasons:
        sample["rejection_reasons"] = rejection_reasons
    
    return sample


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Initialize state
    STATE.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE.start_time = datetime.now().isoformat()
    
    print("=" * 70)
    print("🚀 CROSS-VERIFIED NL-ATL DATASET GENERATOR")
    print("=" * 70)
    print(f"Run ID: {STATE.run_id}")
    print(f"Samples per config: {SAMPLES_PER_CONFIG}")
    print(f"Total expected: {SAMPLES_PER_CONFIG * len(CONFIGS)}")
    print(f"Configs: {len(CONFIGS)}")
    print()
    print("⚠️  Press Ctrl+C anytime to save progress and exit")
    print("=" * 70)
    print()
    
    # Initialize clients
    clients = {}
    for provider in ["openai", "anthropic"]:
        try:
            clients[provider] = get_client(provider)
            print(f"✅ {provider} client initialized")
        except Exception as e:
            print(f"❌ {provider} client failed: {e}")
            return
    print()
    
    # Run each configuration
    for config_idx, config in enumerate(CONFIGS, 1):
        config_name = config["name"]
        generator = clients[config["generator"]]
        verifier = clients[config["verifier"]]
        
        # Rotate domains for diversity
        config_domains = DOMAINS[config_idx-1::len(CONFIGS)][:4]  # 4 domains per config
        if len(config_domains) < 4:
            config_domains = DOMAINS[:4]
        
        print(f"[{config_idx}/{len(CONFIGS)}] 🚀 {config_name}")
        print(f"   Generator: {config['generator']}")
        print(f"   Verifier: {config['verifier']}")
        print(f"   Domains: {', '.join(config_domains)}")
        
        sample_count = 0
        attempt = 0
        max_attempts = SAMPLES_PER_CONFIG * 3  # Allow retries
        
        while sample_count < SAMPLES_PER_CONFIG and attempt < max_attempts:
            domain = config_domains[attempt % len(config_domains)]
            pattern = PATTERNS[attempt % len(PATTERNS)]
            
            print(f"   [{sample_count + 1}/{SAMPLES_PER_CONFIG}] {domain}/{pattern}... ", end="", flush=True)
            
            sample = generate_sample(generator, verifier, domain, pattern, config_name)
            
            if sample and "error" not in sample:
                STATE.samples.append(sample)
                sample_count += 1
                # Save after each successful sample
                STATE.save()
            elif sample:
                STATE.errors.append(sample)
            
            attempt += 1
        
        print(f"   ✅ Completed {config_name}: {sample_count} samples\n")
    
    # Final save and report
    samples_file, report_file = STATE.save()
    
    print("=" * 70)
    print("🎯 FINAL REPORT")
    print("=" * 70)
    print_summary()
    print()
    if samples_file:
        print(f"📁 Samples: {samples_file}")
        print(f"📁 Report: {report_file}")
    print()
    print(f"🎉 Done! {len(STATE.samples)} samples generated.")


if __name__ == "__main__":
    main()
