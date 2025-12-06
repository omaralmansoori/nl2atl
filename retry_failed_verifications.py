#!/usr/bin/env python3
"""
Retry Failed Verifications Script

This script:
1. Loads the rejected samples from a previous experiment run
2. Identifies samples that failed due to API errors (credit issues, timeouts, etc.)
3. Retries verification on those samples
4. Merges results back into verified/rejected files
5. Updates statistics
"""

import os
import sys
import json
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import time

from dotenv import load_dotenv
load_dotenv()

# Also try loading from environment directly if .env doesn't exist
import os
if not os.getenv("OPENAI_API_KEY"):
    # Try loading from shell environment
    pass

import openai
import anthropic

# ============================================================================
# CONFIGURATION
# ============================================================================

# The run ID to retry (can be overridden via command line)
DEFAULT_RUN_ID = "20251201_200910"

# Patterns that indicate a retryable API error (not a legitimate rejection)
# These are exact phrases that appear in API error messages
RETRYABLE_ERROR_PATTERNS = [
    "credit balance is too low",
    "rate_limit_exceeded",
    "rate limit exceeded",
    "connection timed out",
    "Request timed out",
    "Error code: 429",
    "Error code: 500",
    "Error code: 502",
    "Error code: 503",
    "Error code: 504",
    "Error code: 400",
    "overloaded_error",
    "invalid_request_error",
    "anthropic api",
    "openai api",
    "APIConnectionError",
    "APITimeoutError",
]

# ============================================================================
# LLM CLIENTS
# ============================================================================

class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        self.name = "openai"
    
    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()


class AnthropicClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-20250514"
        self.name = "anthropic"
    
    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=temperature,
            system=system if system else "You are an expert in formal verification and temporal logic.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


# ============================================================================
# VERIFICATION LOGIC
# ============================================================================

def verify_translation(client, nl: str, atl: str, domain: str) -> dict:
    """Verify a single NL-ATL pair."""
    
    system_prompt = """You are an expert ATL formula verifier. Your task is to check if an ATL formula 
correctly captures the meaning of a natural language requirement.

Verification criteria:
1. Agents: Are the correct agents in the coalition?
2. Temporal pattern: Does the formula use the right temporal operators?
3. Propositions: Do the proposition names reasonably represent the concepts?
4. Semantics: Does the formula capture the core meaning of the requirement?

Be lenient on minor naming differences. Focus on structural correctness.
A formula is CORRECT if it captures the essential meaning, even if not perfect."""

    user_prompt = f"""Verify if this ATL formula correctly captures the natural language requirement:

Natural Language: "{nl}"

ATL Formula: {atl}

Domain: {domain}

Analyze:
1. Does the coalition contain appropriate agents?
2. Does the temporal structure match the requirement?
3. Do the propositions represent the key concepts?

Respond with ONLY valid JSON (no markdown):
{{"verdict": "ACCEPT" or "REJECT", "confidence": 0.0-1.0, "issues": ["list", "of", "issues"], "explanation": "brief explanation"}}"""

    try:
        response = client.generate(user_prompt, system_prompt, temperature=0.2)
        
        # Extract JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        
        return {"verdict": "REJECT", "confidence": 0.0, "issues": ["Could not parse response"], "explanation": response[:200]}
    except Exception as e:
        return {"verdict": "ERROR", "confidence": 0.0, "issues": [str(e)], "explanation": "Verification failed"}


def is_retryable_error(sample: dict) -> bool:
    """Check if a sample failed due to a retryable API error."""
    rejection_reasons = sample.get("rejection_reasons", [])
    verification_notes = sample.get("verification_notes", [])
    
    all_text = " ".join(str(r) for r in rejection_reasons + verification_notes).lower()
    
    for pattern in RETRYABLE_ERROR_PATTERNS:
        if pattern.lower() in all_text:
            return True
    
    return False


def load_experiment_data(run_id: str) -> Tuple[dict, dict, List[dict]]:
    """Load verified and rejected data from an experiment run."""
    
    verified_path = Path(f"data/experiment_{run_id}_verified.json")
    rejected_path = Path(f"data/experiment_{run_id}_rejected.json")
    
    if not verified_path.exists():
        print(f"❌ Verified file not found: {verified_path}")
        sys.exit(1)
    
    if not rejected_path.exists():
        print(f"❌ Rejected file not found: {rejected_path}")
        sys.exit(1)
    
    with open(verified_path) as f:
        verified_data = json.load(f)
    
    with open(rejected_path) as f:
        rejected_data = json.load(f)
    
    # Find retryable samples
    retryable = []
    for sample in rejected_data["samples"]:
        if is_retryable_error(sample):
            retryable.append(sample)
    
    return verified_data, rejected_data, retryable


def save_merged_results(run_id: str, verified_data: dict, rejected_data: dict, 
                         newly_verified: List[dict], still_rejected: List[dict],
                         retry_stats: dict):
    """Save merged results with updated statistics."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update verified data
    verified_data["samples"].extend(newly_verified)
    verified_data["total_verified"] = len(verified_data["samples"])
    verified_data["generated_at"] = datetime.now().isoformat()
    verified_data["retry_info"] = {
        "retry_timestamp": timestamp,
        "samples_retried": retry_stats["total_retried"],
        "newly_verified": len(newly_verified),
    }
    
    # Compute updated stats
    verified_data["stats"] = compute_stats(verified_data["samples"])
    
    # Update rejected data - remove retryable samples and add still_rejected
    non_retryable = [s for s in rejected_data["samples"] if not is_retryable_error(s)]
    rejected_data["samples"] = non_retryable + still_rejected
    rejected_data["total_rejected"] = len(rejected_data["samples"])
    rejected_data["generated_at"] = datetime.now().isoformat()
    rejected_data["retry_info"] = {
        "retry_timestamp": timestamp,
        "samples_retried": retry_stats["total_retried"],
        "still_rejected_after_retry": len(still_rejected),
    }
    
    # Save verified
    verified_path = Path(f"data/experiment_{run_id}_verified.json")
    with open(verified_path, "w") as f:
        json.dump(verified_data, f, indent=2)
    
    # Also update JSONL
    jsonl_path = Path(f"data/experiment_{run_id}_verified.jsonl")
    with open(jsonl_path, "w") as f:
        for sample in verified_data["samples"]:
            f.write(json.dumps(sample) + "\n")
    
    # Save rejected
    rejected_path = Path(f"data/experiment_{run_id}_rejected.json")
    with open(rejected_path, "w") as f:
        json.dump(rejected_data, f, indent=2)
    
    # Save retry report
    report = {
        "run_id": run_id,
        "retry_timestamp": timestamp,
        "stats": retry_stats,
        "files_updated": [str(verified_path), str(rejected_path), str(jsonl_path)],
    }
    
    report_path = Path(f"reports/retry_{run_id}_{timestamp}.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return verified_path, rejected_path, report_path


def compute_stats(samples: List[dict]) -> dict:
    """Compute statistics for samples."""
    domain_counts = Counter()
    operator_counts = Counter()
    pattern_counts = Counter()
    generator_counts = Counter()
    verifier_counts = Counter()
    
    for s in samples:
        domain_counts[s.get("domain", "unknown")] += 1
        pattern_counts[s.get("pattern", "unknown")] += 1
        generator_counts[s.get("atl_generator", "unknown")] += 1
        verifier_counts[s.get("verifier", "unknown")] += 1
        for op in s.get("operators", []):
            operator_counts[op] += 1
    
    return {
        "by_domain": dict(domain_counts),
        "by_pattern": dict(pattern_counts),
        "by_operator": dict(operator_counts),
        "by_generator": dict(generator_counts),
        "by_verifier": dict(verifier_counts),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Get run ID from command line or use default
    run_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_ID
    
    print("=" * 70)
    print("🔄 RETRY FAILED VERIFICATIONS")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Time: {datetime.now().isoformat()}")
    print()
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        return
    
    print("✅ API keys found")
    print()
    
    # Load data
    print("📂 Loading experiment data...")
    verified_data, rejected_data, retryable_samples = load_experiment_data(run_id)
    
    print(f"   Verified samples: {len(verified_data['samples']):,}")
    print(f"   Rejected samples: {len(rejected_data['samples']):,}")
    print(f"   Retryable (API errors): {len(retryable_samples):,}")
    print()
    
    if not retryable_samples:
        print("✅ No samples to retry!")
        return
    
    # Initialize clients
    clients = {
        "openai": OpenAIClient(),
        "anthropic": AnthropicClient(),
    }
    
    # Track results
    newly_verified = []
    still_rejected = []
    retry_errors = []
    
    print(f"🔄 Retrying {len(retryable_samples)} samples...")
    print()
    
    for i, sample in enumerate(retryable_samples):
        # Determine which verifier to use (use the one that was originally assigned)
        verifier_name = sample.get("verifier", "anthropic")
        client = clients[verifier_name]
        
        print(f"  [{i+1}/{len(retryable_samples)}] ID: {sample['id'][:8]}... ({verifier_name})", end=" ", flush=True)
        
        try:
            result = verify_translation(
                client,
                sample["nl_statement"],
                sample["atl_formula"],
                sample["domain"]
            )
            
            verdict = result.get("verdict", "REJECT").upper()
            
            # Update sample with new verification
            sample["verification_verdict"] = verdict
            sample["verification_confidence"] = result.get("confidence", 0.0)
            sample["verification_notes"] = [result.get("explanation", "")]
            sample["verified_at"] = datetime.now().isoformat()
            sample["retry_attempt"] = True
            
            if verdict == "ACCEPT":
                sample["verification_status"] = "verified"
                sample.pop("rejection_reasons", None)
                newly_verified.append(sample)
                print("✅ VERIFIED")
            elif verdict == "ERROR":
                # Still an error, keep in retry list
                sample["verification_status"] = "rejected"
                sample["rejection_reasons"] = result.get("issues", ["Verification error"])
                retry_errors.append(sample)
                print(f"⚠️ ERROR: {result.get('issues', ['Unknown'])[:1]}")
            else:
                sample["verification_status"] = "rejected"
                sample["rejection_reasons"] = result.get("issues", ["Rejected by verifier"])
                still_rejected.append(sample)
                print("❌ REJECTED")
            
            time.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            print(f"⚠️ Exception: {str(e)[:50]}")
            sample["rejection_reasons"] = [f"Retry error: {str(e)}"]
            retry_errors.append(sample)
    
    # Handle retry errors - add them back to still_rejected
    still_rejected.extend(retry_errors)
    
    # Print summary
    print()
    print("=" * 70)
    print("📊 RETRY SUMMARY")
    print("=" * 70)
    print(f"  Total retried: {len(retryable_samples)}")
    print(f"  ✅ Newly verified: {len(newly_verified)}")
    print(f"  ❌ Still rejected: {len(still_rejected)}")
    print(f"  ⚠️ Retry errors: {len(retry_errors)}")
    print()
    
    retry_stats = {
        "total_retried": len(retryable_samples),
        "newly_verified": len(newly_verified),
        "still_rejected": len(still_rejected),
        "retry_errors": len(retry_errors),
    }
    
    # Save merged results
    print("💾 Saving merged results...")
    verified_path, rejected_path, report_path = save_merged_results(
        run_id, verified_data, rejected_data,
        newly_verified, still_rejected, retry_stats
    )
    
    print(f"   ✅ Updated: {verified_path}")
    print(f"   ✅ Updated: {rejected_path}")
    print(f"   ✅ Report: {report_path}")
    print()
    
    # Final counts
    print("=" * 70)
    print("📈 FINAL COUNTS")
    print("=" * 70)
    print(f"  Verified: {verified_data['total_verified']:,}")
    print(f"  Rejected: {rejected_data['total_rejected']:,}")
    print(f"  Total: {verified_data['total_verified'] + rejected_data['total_rejected']:,}")
    print()
    print("🎉 Retry complete!")


if __name__ == "__main__":
    main()
