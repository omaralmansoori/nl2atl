#!/usr/bin/env python3
"""
Large-Scale NL-ATL Dataset Generation Experiment

This experiment generates 2000+ unique NL statements across 10 domains using both
OpenAI and Anthropic LLMs, then translates and cross-verifies them to ATL formulas.

Pipeline:
1. Generate 100 NL statements per domain × 10 domains × 2 LLMs = 2000 NL statements
2. Deduplicate and store unique NL statements by domain
3. Split NL statements: 50% to OpenAI for ATL translation, 50% to Anthropic
4. Cross-verify: Each LLM verifies 50% of its own + 50% of the other's ATL formulas
5. Output: 3 files - unique NL statements, verified pairs, rejected pairs
"""

import os
import sys
import json
import re
import signal
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment
from dotenv import load_dotenv
load_dotenv()

# LLM clients
import openai
import anthropic

# ============================================================================
# CONFIGURATION
# ============================================================================

# 10 Diverse domains for maximum variety
DOMAINS = {
    "autonomous_vehicles": {
        "description": "Self-driving cars, trucks, and autonomous vehicle systems",
        "agents": ["vehicle", "sensor_array", "navigation", "collision_avoidance", "traffic_controller", 
                   "pedestrian_detector", "lane_keeper", "emergency_brake", "route_planner", "v2v_communicator"],
        "concepts": ["obstacle detection", "lane keeping", "traffic rules", "pedestrian safety", 
                     "emergency stops", "route optimization", "sensor fusion", "vehicle coordination"],
    },
    "smart_grid": {
        "description": "Power grid management, renewable energy, and smart energy systems",
        "agents": ["grid_controller", "solar_panel", "wind_turbine", "battery_storage", "load_balancer",
                   "demand_predictor", "fault_detector", "power_router", "consumer_meter", "energy_trader"],
        "concepts": ["load balancing", "fault detection", "renewable integration", "demand response",
                     "power quality", "grid stability", "energy storage", "peak shaving"],
    },
    "healthcare_monitoring": {
        "description": "Patient monitoring, medical devices, and healthcare automation",
        "agents": ["patient_monitor", "vital_sensor", "alarm_system", "drug_dispenser", "nurse_station",
                   "diagnostic_ai", "emergency_responder", "record_keeper", "dosage_calculator", "triage_system"],
        "concepts": ["vital sign monitoring", "medication dosing", "emergency alerts", "patient safety",
                     "diagnostic accuracy", "treatment protocols", "privacy compliance", "resource allocation"],
    },
    "financial_trading": {
        "description": "Algorithmic trading, market making, and financial systems",
        "agents": ["trading_bot", "risk_manager", "market_analyzer", "order_executor", "portfolio_optimizer",
                   "fraud_detector", "compliance_checker", "price_predictor", "liquidity_provider", "arbitrage_finder"],
        "concepts": ["order execution", "risk limits", "market manipulation prevention", "fair pricing",
                     "position limits", "regulatory compliance", "latency requirements", "portfolio rebalancing"],
    },
    "space_systems": {
        "description": "Satellites, spacecraft, and space mission control",
        "agents": ["satellite", "ground_station", "mission_control", "orbit_controller", "payload_manager",
                   "communication_relay", "power_subsystem", "thermal_controller", "collision_predictor", "data_downlink"],
        "concepts": ["orbit maintenance", "collision avoidance", "power conservation", "data transmission",
                     "thermal regulation", "mission objectives", "ground contact windows", "contingency handling"],
    },
    "manufacturing_robotics": {
        "description": "Industrial robots, assembly lines, and smart manufacturing",
        "agents": ["assembly_robot", "quality_inspector", "conveyor_controller", "tool_changer", "material_handler",
                   "safety_fence", "production_scheduler", "maintenance_predictor", "inventory_tracker", "defect_detector"],
        "concepts": ["assembly sequence", "quality control", "worker safety", "throughput optimization",
                     "defect detection", "tool wear monitoring", "inventory replenishment", "downtime prevention"],
    },
    "air_traffic_control": {
        "description": "Aircraft management, airspace control, and aviation safety",
        "agents": ["aircraft", "atc_controller", "radar_system", "weather_monitor", "runway_manager",
                   "conflict_detector", "approach_sequencer", "departure_manager", "emergency_coordinator", "flight_tracker"],
        "concepts": ["separation assurance", "runway allocation", "weather avoidance", "conflict resolution",
                     "approach sequencing", "emergency procedures", "airspace capacity", "delay management"],
    },
    "smart_building": {
        "description": "Building automation, HVAC, security, and facility management",
        "agents": ["hvac_controller", "lighting_system", "security_monitor", "access_controller", "fire_alarm",
                   "elevator_manager", "occupancy_sensor", "energy_optimizer", "maintenance_scheduler", "visitor_manager"],
        "concepts": ["comfort optimization", "energy efficiency", "access control", "fire safety",
                     "occupancy management", "maintenance scheduling", "security monitoring", "resource allocation"],
    },
    "supply_chain": {
        "description": "Logistics, warehousing, and supply chain management",
        "agents": ["warehouse_robot", "inventory_manager", "order_processor", "shipping_coordinator", "demand_forecaster",
                   "supplier_monitor", "route_optimizer", "quality_checker", "customs_handler", "returns_processor"],
        "concepts": ["inventory optimization", "order fulfillment", "shipping deadlines", "quality assurance",
                     "supplier reliability", "demand prediction", "route efficiency", "customs compliance"],
    },
    "telecommunications": {
        "description": "Network management, 5G, and communication systems",
        "agents": ["base_station", "network_controller", "traffic_manager", "spectrum_allocator", "handover_manager",
                   "interference_detector", "qos_controller", "security_monitor", "load_balancer", "fault_manager"],
        "concepts": ["coverage optimization", "handover management", "interference mitigation", "quality of service",
                     "spectrum efficiency", "network security", "load distribution", "fault recovery"],
    },
}

# ATL Pattern templates for diverse formula generation
ATL_PATTERNS = {
    "safety": {
        "description": "Something bad never happens",
        "nl_templates": [
            "must always prevent {bad_thing}",
            "must never allow {bad_thing} to occur",
            "guarantees that {bad_thing} is always avoided",
            "ensures {bad_thing} never happens",
            "must maintain the invariant that {bad_thing} does not occur",
        ],
        "atl_template": "<<{agents}>> G(!{predicate})",
    },
    "liveness": {
        "description": "Something good eventually happens",
        "nl_templates": [
            "must eventually achieve {good_thing}",
            "ensures that {good_thing} will eventually occur",
            "guarantees {good_thing} is reached at some point",
            "must make sure {good_thing} eventually happens",
            "can ensure {good_thing} will be accomplished",
        ],
        "atl_template": "<<{agents}>> F({predicate})",
    },
    "response": {
        "description": "If trigger then eventually response",
        "nl_templates": [
            "whenever {trigger} occurs, must eventually ensure {response}",
            "if {trigger} happens, then {response} must eventually follow",
            "must guarantee that {trigger} leads to eventual {response}",
            "upon {trigger}, must ensure {response} eventually occurs",
            "in response to {trigger}, must achieve {response}",
        ],
        "atl_template": "<<{agents}>> G({trigger} -> F({response}))",
    },
    "until": {
        "description": "Condition holds until goal is reached",
        "nl_templates": [
            "must maintain {condition} until {goal} is achieved",
            "ensures {condition} holds until {goal} occurs",
            "must keep {condition} true until reaching {goal}",
            "guarantees {condition} persists until {goal}",
            "must sustain {condition} until {goal} is satisfied",
        ],
        "atl_template": "<<{agents}>> ({condition} U {goal})",
    },
    "persistence": {
        "description": "Something becomes true and stays true forever",
        "nl_templates": [
            "must eventually reach and maintain {state} permanently",
            "ensures {state} eventually becomes and remains true",
            "must achieve {state} and preserve it indefinitely",
            "guarantees eventual stabilization in {state}",
            "must reach {state} and never leave it",
        ],
        "atl_template": "<<{agents}>> F(G({predicate}))",
    },
    "recurrence": {
        "description": "Something happens infinitely often",
        "nl_templates": [
            "must ensure {event} happens infinitely often",
            "guarantees {event} recurs repeatedly forever",
            "must make {event} occur again and again",
            "ensures {event} keeps happening periodically",
            "must repeatedly achieve {event}",
        ],
        "atl_template": "<<{agents}>> G(F({predicate}))",
    },
    "precedence": {
        "description": "One thing must happen before another",
        "nl_templates": [
            "must ensure {first} happens before {second} can occur",
            "guarantees that {second} only happens after {first}",
            "must achieve {first} prior to allowing {second}",
            "{second} is blocked until {first} occurs",
            "must sequence {first} before {second}",
        ],
        "atl_template": "<<{agents}>> (!{second} U {first})",
    },
    "conditional_safety": {
        "description": "If condition, then always safe",
        "nl_templates": [
            "when {condition} holds, must always ensure {safe_state}",
            "if {condition} is true, then {safe_state} must always follow",
            "under {condition}, must guarantee permanent {safe_state}",
            "given {condition}, must maintain {safe_state} forever",
            "assuming {condition}, must preserve {safe_state} always",
        ],
        "atl_template": "<<{agents}>> G({condition} -> G({safe}))",
    },
}

# Number of NL statements to generate per domain per LLM
NL_STATEMENTS_PER_DOMAIN_PER_LLM = 100

# Batch size for generation (to show progress)
BATCH_SIZE = 10

# ============================================================================
# GLOBAL STATE
# ============================================================================

@dataclass
class ExperimentState:
    """Tracks experiment state for incremental saving."""
    run_id: str = ""
    start_time: str = ""
    
    # Stage 1: NL statements by domain and generator
    nl_statements: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    unique_nl_statements: Dict[str, List[dict]] = field(default_factory=lambda: defaultdict(list))
    
    # Stage 2: ATL translations
    translations: List[dict] = field(default_factory=list)
    
    # Stage 3: Verified and rejected pairs
    verified_pairs: List[dict] = field(default_factory=list)
    rejected_pairs: List[dict] = field(default_factory=list)
    
    # Tracking
    errors: List[dict] = field(default_factory=list)
    interrupted: bool = False
    current_stage: str = "init"
    
    def save_nl_statements(self):
        """Save unique NL statements sorted by domain."""
        output = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "run_id": self.run_id,
            "total_unique": sum(len(v) for v in self.unique_nl_statements.values()),
            "by_domain": {},
        }
        
        for domain in sorted(self.unique_nl_statements.keys()):
            statements = self.unique_nl_statements[domain]
            output["by_domain"][domain] = {
                "count": len(statements),
                "statements": sorted(statements, key=lambda x: x["nl_statement"]),
            }
        
        output_file = Path(f"data/experiment_{self.run_id}_nl_statements.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return output_file
    
    def save_verified_pairs(self):
        """Save verified NL-ATL pairs."""
        output = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "run_id": self.run_id,
            "total_verified": len(self.verified_pairs),
            "samples": self.verified_pairs,
            "stats": self._compute_stats(self.verified_pairs),
        }
        
        output_file = Path(f"data/experiment_{self.run_id}_verified.json")
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        # Also save as JSONL for easy processing
        jsonl_file = Path(f"data/experiment_{self.run_id}_verified.jsonl")
        with open(jsonl_file, "w") as f:
            for sample in self.verified_pairs:
                f.write(json.dumps(sample) + "\n")
        
        return output_file
    
    def save_rejected_pairs(self):
        """Save rejected NL-ATL pairs."""
        output = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "run_id": self.run_id,
            "total_rejected": len(self.rejected_pairs),
            "samples": self.rejected_pairs,
            "rejection_reasons": self._compute_rejection_stats(),
        }
        
        output_file = Path(f"data/experiment_{self.run_id}_rejected.json")
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return output_file
    
    def save_all(self):
        """Save all output files."""
        files = {}
        if self.unique_nl_statements:
            files["nl_statements"] = self.save_nl_statements()
        if self.verified_pairs:
            files["verified"] = self.save_verified_pairs()
        if self.rejected_pairs:
            files["rejected"] = self.save_rejected_pairs()
        
        # Save experiment report
        report = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "interrupted": self.interrupted,
            "current_stage": self.current_stage,
            "summary": {
                "total_nl_generated": sum(
                    len(stmts) 
                    for domain_dict in self.nl_statements.values() 
                    for stmts in domain_dict.values()
                ),
                "total_unique_nl": sum(len(v) for v in self.unique_nl_statements.values()),
                "total_translations": len(self.translations),
                "total_verified": len(self.verified_pairs),
                "total_rejected": len(self.rejected_pairs),
                "total_errors": len(self.errors),
            },
            "files": {k: str(v) for k, v in files.items()},
            "errors": self.errors[:50],  # Last 50 errors
        }
        
        report_file = Path(f"reports/experiment_{self.run_id}_report.json")
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        files["report"] = report_file
        return files
    
    def _compute_stats(self, samples: List[dict]) -> dict:
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
    
    def _compute_rejection_stats(self) -> dict:
        """Compute rejection reason statistics."""
        reason_counts = Counter()
        for s in self.rejected_pairs:
            for reason in s.get("rejection_reasons", ["unknown"]):
                reason_counts[reason[:100]] += 1
        return dict(reason_counts.most_common(20))


# Global state
STATE = ExperimentState()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\n\n⚠️  INTERRUPT RECEIVED - Saving current progress...")
    STATE.interrupted = True
    files = STATE.save_all()
    print(f"\n✅ Saved files:")
    for name, path in files.items():
        print(f"   {name}: {path}")
    print(f"\n📊 Summary: {len(STATE.verified_pairs)} verified, {len(STATE.rejected_pairs)} rejected")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# LLM CLIENTS
# ============================================================================

class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        self.name = "openai"
    
    def generate(self, prompt: str, system: str = "", temperature: float = 0.8) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=4000,
        )
        return response.choices[0].message.content.strip()


class AnthropicClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-20250514"
        self.name = "anthropic"
    
    def generate(self, prompt: str, system: str = "", temperature: float = 0.8) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=temperature,
            system=system if system else "You are an expert in formal verification and temporal logic.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


# ============================================================================
# STAGE 1: NL STATEMENT GENERATION
# ============================================================================

def generate_nl_batch(client, domain: str, domain_info: dict, batch_num: int, batch_size: int = 10) -> List[str]:
    """Generate a batch of unique NL statements for a domain."""
    
    system_prompt = """You are an expert in multi-agent systems, formal verification, and temporal logic.
Your task is to generate unique, diverse, and realistic natural language requirements for multi-agent systems.

Requirements for generated statements:
1. Each statement must describe a clear temporal property (always, eventually, until, response)
2. Statements must involve 1-3 specific agents from the provided list
3. Statements must be realistic requirements that would appear in a real specification
4. Use varied sentence structures and vocabulary
5. Focus on safety, liveness, fairness, and coordination properties
6. Avoid vague or ambiguous language
7. Each statement should be self-contained and understandable without context

DO NOT include numbered lists, bullet points, or explanations. Output ONLY the statements, one per line."""

    user_prompt = f"""Generate {batch_size} unique natural language requirements for a {domain_info['description']} system.

Domain: {domain}
Available Agents: {', '.join(domain_info['agents'])}
Key Concepts: {', '.join(domain_info['concepts'])}

This is batch {batch_num}, so ensure these are DIFFERENT from typical/obvious requirements.
Be creative and cover edge cases, failure scenarios, coordination requirements, and complex temporal properties.

Focus on variety:
- Mix single-agent and multi-agent requirements
- Include safety properties (something bad never happens)
- Include liveness properties (something good eventually happens)
- Include response properties (if X then eventually Y)
- Include until properties (maintain X until Y)
- Include fairness properties (infinitely often, eventually always)

Generate exactly {batch_size} statements, one per line, no numbering or bullets:"""

    try:
        response = client.generate(user_prompt, system_prompt, temperature=0.9)
        
        # Parse response - split by newlines and clean
        statements = []
        for line in response.split('\n'):
            line = line.strip()
            # Remove numbering if present
            line = re.sub(r'^[\d]+[\.\)\-\:]?\s*', '', line)
            # Remove bullet points
            line = re.sub(r'^[\-\*\•]\s*', '', line)
            # Skip empty lines or too short
            if len(line) > 20:
                statements.append(line)
        
        return statements[:batch_size]
    except Exception as e:
        print(f"    ❌ Error generating batch: {e}")
        return []


def run_stage1_nl_generation():
    """Stage 1: Generate NL statements from both LLMs for all domains."""
    print("\n" + "=" * 70)
    print("📝 STAGE 1: NL STATEMENT GENERATION")
    print("=" * 70)
    print(f"Generating {NL_STATEMENTS_PER_DOMAIN_PER_LLM} statements per domain per LLM")
    print(f"Domains: {len(DOMAINS)}")
    print(f"Expected total: {NL_STATEMENTS_PER_DOMAIN_PER_LLM * len(DOMAINS) * 2:,} statements")
    print()
    
    STATE.current_stage = "nl_generation"
    
    clients = {
        "openai": OpenAIClient(),
        "anthropic": AnthropicClient(),
    }
    
    total_generated = 0
    
    for domain_idx, (domain, domain_info) in enumerate(DOMAINS.items(), 1):
        print(f"\n[{domain_idx}/{len(DOMAINS)}] 📂 Domain: {domain}")
        print(f"    Description: {domain_info['description']}")
        
        for llm_name, client in clients.items():
            print(f"    🤖 {llm_name.upper()}: ", end="", flush=True)
            
            domain_statements = []
            num_batches = NL_STATEMENTS_PER_DOMAIN_PER_LLM // BATCH_SIZE
            
            for batch_num in range(1, num_batches + 1):
                batch = generate_nl_batch(client, domain, domain_info, batch_num, BATCH_SIZE)
                domain_statements.extend(batch)
                print(f".", end="", flush=True)
                time.sleep(0.5)  # Rate limiting
            
            STATE.nl_statements[domain][llm_name] = domain_statements
            total_generated += len(domain_statements)
            print(f" {len(domain_statements)} statements")
    
    # Deduplicate and store unique statements
    print("\n\n🔄 Deduplicating statements...")
    seen_hashes = set()
    
    for domain in DOMAINS:
        for llm_name in ["openai", "anthropic"]:
            for stmt in STATE.nl_statements[domain][llm_name]:
                # Normalize and hash
                normalized = stmt.lower().strip()
                stmt_hash = hashlib.md5(normalized.encode()).hexdigest()
                
                if stmt_hash not in seen_hashes:
                    seen_hashes.add(stmt_hash)
                    STATE.unique_nl_statements[domain].append({
                        "nl_statement": stmt,
                        "generator": llm_name,
                        "domain": domain,
                        "id": stmt_hash[:12],
                    })
    
    total_unique = sum(len(v) for v in STATE.unique_nl_statements.values())
    print(f"✅ Generated {total_generated:,} statements, {total_unique:,} unique")
    
    # Save intermediate results
    nl_file = STATE.save_nl_statements()
    print(f"💾 Saved to: {nl_file}")
    
    return total_unique


# ============================================================================
# STAGE 2: ATL TRANSLATION
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
        if re.search(rf'\b{op}\b|\b{op}\(', formula):
            components["operators"].append(op)
    
    # Extract logical operators
    if '&' in formula or '∧' in formula:
        components["operators"].append('&')
    if '|' in formula or '∨' in formula:
        components["operators"].append('|')
    if '->' in formula or '→' in formula:
        components["operators"].append('->')
    if '!' in formula or '¬' in formula:
        components["operators"].append('!')
    
    # Extract atoms
    atom_pattern = r'\b[a-z][a-z0-9_]*\b'
    potential_atoms = re.findall(atom_pattern, formula.lower())
    reserved = {'true', 'false', 'and', 'or', 'not', 'g', 'f', 'x', 'u', 'w', 'r'}
    components["atoms"] = list(set(a for a in potential_atoms if a not in reserved))
    
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


def detect_pattern(formula: str) -> str:
    """Detect the ATL pattern used in a formula."""
    formula_clean = formula.replace(' ', '')
    
    if 'G(F(' in formula_clean or 'G(F' in formula_clean:
        return "recurrence"
    if 'F(G(' in formula_clean or 'F(G' in formula_clean:
        return "persistence"
    if 'G(' in formula_clean and '->F(' in formula_clean.replace(' ', ''):
        return "response"
    if ')U(' in formula_clean or ' U ' in formula or ')U ' in formula:
        return "until"
    if 'G(!' in formula_clean or 'G(¬' in formula_clean:
        return "safety"
    if 'G(' in formula_clean:
        return "invariant"
    if 'F(' in formula_clean:
        return "liveness"
    
    return "unknown"


def validate_atl_syntax(formula: str) -> Tuple[bool, List[str]]:
    """Validate ATL formula syntax."""
    errors = []
    formula = formula.strip()
    
    if not formula:
        return False, ["Empty formula"]
    
    # Check for coalition operators
    if not re.search(r'<<[^>]+>>', formula):
        errors.append("No coalition operator <<agents>> found")
    
    # Check for temporal operators
    temporal_ops = ['F', 'G', 'X', 'U', 'W', 'R']
    has_temporal = any(re.search(rf'\b{op}\b|\b{op}\(', formula) for op in temporal_ops)
    if not has_temporal:
        errors.append("No temporal operators found")
    
    # Check bracket balance
    if formula.count('(') != formula.count(')'):
        errors.append("Unbalanced parentheses")
    
    # Check angle brackets
    formula_no_arrows = formula.replace('->', '').replace('<-', '')
    open_angles = formula_no_arrows.count('<')
    close_angles = formula_no_arrows.count('>')
    if open_angles != close_angles:
        errors.append("Unbalanced angle brackets")
    
    return len(errors) == 0, errors


def translate_nl_to_atl(client, nl_statement: str, domain: str, domain_info: dict) -> Optional[str]:
    """Translate a single NL statement to ATL formula."""
    
    system_prompt = """You are an expert in Alternating-time Temporal Logic (ATL) and formal verification.
Your task is to translate natural language requirements into precise ATL formulas.

ATL Syntax:
- Coalition operator: <<agent1, agent2, ...>> means "the agents can cooperate to ensure"
- G(φ) = "globally/always φ" - φ holds in all future states
- F(φ) = "finally/eventually φ" - φ will hold at some future state
- X(φ) = "next φ" - φ holds in the next state
- φ U ψ = "φ until ψ" - φ holds until ψ becomes true
- Logical operators: & (and), | (or), -> (implies), ! (not)
- Propositions should be lowercase_with_underscores

Common Patterns:
- Safety: <<agents>> G(!bad_state) - agents can always avoid bad states
- Liveness: <<agents>> F(good_state) - agents can eventually reach good state
- Response: <<agents>> G(trigger -> F(response)) - agents ensure response to trigger
- Until: <<agents>> (maintain U goal) - agents maintain condition until goal
- Persistence: <<agents>> F(G(stable_state)) - agents reach and maintain state
- Recurrence: <<agents>> G(F(event)) - agents ensure event repeats forever

Output ONLY the ATL formula, nothing else. No explanations, no markdown."""

    user_prompt = f"""Translate this natural language requirement to an ATL formula:

"{nl_statement}"

Domain: {domain}
Available agents: {', '.join(domain_info['agents'])}

Important:
1. Use ONLY agents mentioned or implied in the statement
2. Create meaningful proposition names that reflect the concepts
3. Match the temporal pattern to the requirement's intent
4. Keep the formula as simple as possible while being accurate

Output ONLY the ATL formula:"""

    try:
        response = client.generate(user_prompt, system_prompt, temperature=0.3)
        
        # Clean the response
        formula = response.strip()
        # Remove markdown code blocks if present
        formula = re.sub(r'^```\w*\n?', '', formula)
        formula = re.sub(r'\n?```$', '', formula)
        formula = formula.strip()
        
        # Basic validation
        if '<<' in formula and '>>' in formula:
            return formula
        
        return None
    except Exception as e:
        print(f"      ❌ Translation error: {e}")
        return None


def run_stage2_translation():
    """Stage 2: Translate NL statements to ATL formulas."""
    print("\n" + "=" * 70)
    print("🔄 STAGE 2: ATL TRANSLATION")
    print("=" * 70)
    
    STATE.current_stage = "translation"
    
    # Collect all unique statements
    all_statements = []
    for domain, statements in STATE.unique_nl_statements.items():
        for stmt in statements:
            stmt["domain_info"] = DOMAINS[domain]
            all_statements.append(stmt)
    
    print(f"Translating {len(all_statements):,} unique NL statements")
    print("Splitting 50/50 between OpenAI and Anthropic for translation")
    print()
    
    # Shuffle and split
    random.shuffle(all_statements)
    midpoint = len(all_statements) // 2
    
    openai_batch = all_statements[:midpoint]
    anthropic_batch = all_statements[midpoint:]
    
    print(f"  OpenAI will translate: {len(openai_batch):,} statements")
    print(f"  Anthropic will translate: {len(anthropic_batch):,} statements")
    
    clients = {
        "openai": OpenAIClient(),
        "anthropic": AnthropicClient(),
    }
    
    translations = []
    
    # Process OpenAI batch
    print(f"\n  🤖 OpenAI translating: ", end="", flush=True)
    for i, stmt in enumerate(openai_batch):
        if i % 50 == 0 and i > 0:
            print(f"{i}", end="", flush=True)
        elif i % 10 == 0:
            print(".", end="", flush=True)
        
        atl = translate_nl_to_atl(clients["openai"], stmt["nl_statement"], stmt["domain"], stmt["domain_info"])
        
        if atl:
            valid, syntax_errors = validate_atl_syntax(atl)
            components = extract_components(atl)
            
            translations.append({
                "id": stmt["id"],
                "nl_statement": stmt["nl_statement"],
                "atl_formula": atl,
                "atl_unicode": formula_to_unicode(atl),
                "domain": stmt["domain"],
                "nl_generator": stmt["generator"],
                "atl_generator": "openai",
                "agents": components["agents"],
                "operators": components["operators"],
                "atoms": components["atoms"],
                "pattern": detect_pattern(atl),
                "syntax_valid": valid,
                "syntax_errors": syntax_errors,
                "created_at": datetime.now().isoformat(),
            })
        
        time.sleep(0.2)  # Rate limiting
    
    print(f" {len([t for t in translations if t['atl_generator'] == 'openai'])} done")
    
    # Process Anthropic batch
    print(f"  🤖 Anthropic translating: ", end="", flush=True)
    for i, stmt in enumerate(anthropic_batch):
        if i % 50 == 0 and i > 0:
            print(f"{i}", end="", flush=True)
        elif i % 10 == 0:
            print(".", end="", flush=True)
        
        atl = translate_nl_to_atl(clients["anthropic"], stmt["nl_statement"], stmt["domain"], stmt["domain_info"])
        
        if atl:
            valid, syntax_errors = validate_atl_syntax(atl)
            components = extract_components(atl)
            
            translations.append({
                "id": stmt["id"],
                "nl_statement": stmt["nl_statement"],
                "atl_formula": atl,
                "atl_unicode": formula_to_unicode(atl),
                "domain": stmt["domain"],
                "nl_generator": stmt["generator"],
                "atl_generator": "anthropic",
                "agents": components["agents"],
                "operators": components["operators"],
                "atoms": components["atoms"],
                "pattern": detect_pattern(atl),
                "syntax_valid": valid,
                "syntax_errors": syntax_errors,
                "created_at": datetime.now().isoformat(),
            })
        
        time.sleep(0.2)  # Rate limiting
    
    print(f" {len([t for t in translations if t['atl_generator'] == 'anthropic'])} done")
    
    STATE.translations = translations
    
    valid_count = len([t for t in translations if t["syntax_valid"]])
    print(f"\n✅ Generated {len(translations):,} ATL translations, {valid_count:,} syntactically valid")
    
    return len(translations)


# ============================================================================
# STAGE 3: CROSS-VERIFICATION
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
        return {"verdict": "REJECT", "confidence": 0.0, "issues": [str(e)], "explanation": "Verification failed"}


def run_stage3_verification():
    """Stage 3: Cross-verify ATL translations."""
    print("\n" + "=" * 70)
    print("✅ STAGE 3: CROSS-VERIFICATION")
    print("=" * 70)
    
    STATE.current_stage = "verification"
    
    # Filter to syntactically valid translations
    valid_translations = [t for t in STATE.translations if t["syntax_valid"]]
    invalid_translations = [t for t in STATE.translations if not t["syntax_valid"]]
    
    print(f"Syntactically valid: {len(valid_translations):,}")
    print(f"Syntactically invalid (auto-rejected): {len(invalid_translations):,}")
    
    # Auto-reject syntactically invalid
    for t in invalid_translations:
        STATE.rejected_pairs.append({
            **t,
            "verification_status": "rejected",
            "verifier": "syntax_check",
            "rejection_reasons": t["syntax_errors"],
            "verification_notes": ["Auto-rejected due to syntax errors"],
        })
    
    # Split valid translations for cross-verification
    # OpenAI-generated -> 50% verified by OpenAI, 50% by Anthropic
    # Anthropic-generated -> 50% verified by OpenAI, 50% by Anthropic
    
    openai_generated = [t for t in valid_translations if t["atl_generator"] == "openai"]
    anthropic_generated = [t for t in valid_translations if t["atl_generator"] == "anthropic"]
    
    random.shuffle(openai_generated)
    random.shuffle(anthropic_generated)
    
    # Create verification assignments
    verification_queue = []
    
    # OpenAI-generated: half to each verifier
    mid_o = len(openai_generated) // 2
    for t in openai_generated[:mid_o]:
        verification_queue.append((t, "openai"))  # Self-verify
    for t in openai_generated[mid_o:]:
        verification_queue.append((t, "anthropic"))  # Cross-verify
    
    # Anthropic-generated: half to each verifier
    mid_a = len(anthropic_generated) // 2
    for t in anthropic_generated[:mid_a]:
        verification_queue.append((t, "anthropic"))  # Self-verify
    for t in anthropic_generated[mid_a:]:
        verification_queue.append((t, "openai"))  # Cross-verify
    
    print(f"\nVerification assignments:")
    print(f"  OpenAI verifying: {len([v for v in verification_queue if v[1] == 'openai']):,}")
    print(f"  Anthropic verifying: {len([v for v in verification_queue if v[1] == 'anthropic']):,}")
    
    clients = {
        "openai": OpenAIClient(),
        "anthropic": AnthropicClient(),
    }
    
    print(f"\n  Verifying: ", end="", flush=True)
    
    for i, (translation, verifier_name) in enumerate(verification_queue):
        if i % 100 == 0 and i > 0:
            print(f"\n  [{i}/{len(verification_queue)}] ", end="", flush=True)
        elif i % 20 == 0:
            print(".", end="", flush=True)
        
        client = clients[verifier_name]
        result = verify_translation(client, translation["nl_statement"], translation["atl_formula"], translation["domain"])
        
        verdict = result.get("verdict", "REJECT").upper()
        
        sample = {
            **translation,
            "verifier": verifier_name,
            "verification_verdict": verdict,
            "verification_confidence": result.get("confidence", 0.0),
            "verification_notes": [result.get("explanation", "")],
            "verified_at": datetime.now().isoformat(),
        }
        
        if verdict == "ACCEPT":
            sample["verification_status"] = "verified"
            STATE.verified_pairs.append(sample)
        else:
            sample["verification_status"] = "rejected"
            sample["rejection_reasons"] = result.get("issues", ["Verification rejected"])
            STATE.rejected_pairs.append(sample)
        
        time.sleep(0.2)  # Rate limiting
    
    print(f"\n\n✅ Verification complete!")
    print(f"  Verified: {len(STATE.verified_pairs):,}")
    print(f"  Rejected: {len(STATE.rejected_pairs):,}")
    
    return len(STATE.verified_pairs)


# ============================================================================
# MAIN
# ============================================================================

def print_final_summary():
    """Print final experiment summary."""
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)
    
    total_nl = sum(len(v) for v in STATE.unique_nl_statements.values())
    
    print(f"\n📝 NL Statements:")
    print(f"   Total unique: {total_nl:,}")
    print(f"   By domain:")
    for domain in sorted(STATE.unique_nl_statements.keys()):
        count = len(STATE.unique_nl_statements[domain])
        print(f"     {domain}: {count}")
    
    print(f"\n🔄 ATL Translations:")
    print(f"   Total: {len(STATE.translations):,}")
    valid = len([t for t in STATE.translations if t["syntax_valid"]])
    print(f"   Syntactically valid: {valid:,}")
    
    print(f"\n✅ Verified Pairs: {len(STATE.verified_pairs):,}")
    print(f"❌ Rejected Pairs: {len(STATE.rejected_pairs):,}")
    
    if STATE.verified_pairs:
        print(f"\n   Verified by domain:")
        domain_counts = Counter(p["domain"] for p in STATE.verified_pairs)
        for domain, count in domain_counts.most_common():
            print(f"     {domain}: {count}")
        
        print(f"\n   Verified by pattern:")
        pattern_counts = Counter(p.get("pattern", "unknown") for p in STATE.verified_pairs)
        for pattern, count in pattern_counts.most_common():
            print(f"     {pattern}: {count}")
    
    if STATE.rejected_pairs:
        print(f"\n   Top rejection reasons:")
        reason_counts = Counter()
        for p in STATE.rejected_pairs:
            for r in p.get("rejection_reasons", ["unknown"])[:1]:
                reason_counts[r[:60]] += 1
        for reason, count in reason_counts.most_common(5):
            print(f"     [{count}] {reason}...")


def main():
    """Main experiment runner."""
    STATE.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE.start_time = datetime.now().isoformat()
    
    print("=" * 70)
    print("🚀 LARGE-SCALE NL-ATL DATASET EXPERIMENT")
    print("=" * 70)
    print(f"Run ID: {STATE.run_id}")
    print(f"Started: {STATE.start_time}")
    print(f"\nConfiguration:")
    print(f"  Domains: {len(DOMAINS)}")
    print(f"  NL per domain per LLM: {NL_STATEMENTS_PER_DOMAIN_PER_LLM}")
    print(f"  Expected NL statements: {NL_STATEMENTS_PER_DOMAIN_PER_LLM * len(DOMAINS) * 2:,}")
    print(f"  Target verified pairs: ~2000")
    print()
    print("⚠️  Press Ctrl+C anytime to save progress and exit")
    print("=" * 70)
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        return
    
    print("\n✅ API keys found")
    
    # Run stages
    try:
        # Stage 1: Generate NL statements
        total_nl = run_stage1_nl_generation()
        
        # Stage 2: Translate to ATL
        total_translations = run_stage2_translation()
        
        # Stage 3: Cross-verify
        total_verified = run_stage3_verification()
        
    except Exception as e:
        print(f"\n❌ Experiment error: {e}")
        import traceback
        traceback.print_exc()
        STATE.errors.append({"error": str(e), "stage": STATE.current_stage})
    
    # Save all results
    print("\n\n💾 Saving results...")
    files = STATE.save_all()
    
    print(f"\n📁 Output files:")
    for name, path in files.items():
        print(f"   {name}: {path}")
    
    # Print summary
    print_final_summary()
    
    print(f"\n🎉 Experiment complete!")
    print(f"   Run ID: {STATE.run_id}")
    print(f"   Duration: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
