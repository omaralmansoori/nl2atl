"""
Dataset Generation Module
==========================

Generates synthetic NL-ATL pairs for training and evaluation.

Pipeline Overview
-----------------
```
                    ┌─────────────────────────────────────────┐
                    │           Generation Modes              │
                    └─────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ Template  │      │   LLM     │      │  Hybrid   │
    │   Mode    │      │   Mode    │      │   Mode    │
    └───────────┘      └───────────┘      └───────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    Template         LLM generates       Template base +
    Instantiation    NL (high temp)      LLM paraphrasing
          │                   │                   │
          │                   ▼                   │
          │           LLM translates              │
          │           to ATL (low temp)           │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  Cross-Check    │
                    │   (Optional)    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Output Dataset │
                    └─────────────────┘
```

Features
--------
1. **Template Mode**: Combine ATL templates with concrete coalitions
   and atomic propositions to generate ground truth pairs.

2. **LLM Mode**: Use LLM to generate creative NL statements (high temperature),
   then translate to ATL (low temperature) for accuracy.

3. **Hybrid Mode**: Template-based generation with LLM paraphrasing for variety.

4. **Cross-Checking**: Validate pairs using LLM critique to filter 
   low-quality entries.

5. **Domain Diversity**: Support for multiple domains with domain-specific
   vocabulary and agent names.

For AI Integration
------------------
Key classes and functions:
- `DatasetGenerator`: Main generator class with configurable pipeline
- `GenerationConfig`: Configuration dataclass for generation options
- `GenerationMode`: Enum for template/llm/hybrid modes
- `NLATLPair`: Data structure for individual pairs (alias for ATLSample)
- `sample_coalitions()`, `sample_atoms()`: Sampling utilities

Example Usage
-------------
>>> from dataset_gen import DatasetGenerator, GenerationConfig, GenerationMode
>>>
>>> # Template-only generation (fast, no API)
>>> config = GenerationConfig(
...     num_examples=100,
...     mode=GenerationMode.TEMPLATE,
... )
>>> generator = DatasetGenerator(config)
>>> pairs = generator.generate()
>>>
>>> # LLM generation (creative, requires API)
>>> config = GenerationConfig(
...     num_examples=50,
...     mode=GenerationMode.LLM,
...     nl_temperature=0.9,
...     atl_temperature=0.1,
... )
>>> generator = DatasetGenerator(config)
>>> pairs = generator.generate()

CLI Usage
---------
# Template mode (fast, no API)
python -m dataset_gen --mode template --num-examples 100

# LLM mode (creative, requires API)
python -m dataset_gen --mode llm --num-examples 50 --nl-temp 0.9 --atl-temp 0.1

# Hybrid mode (template base + LLM paraphrasing)
python -m dataset_gen --mode hybrid --num-examples 200 --verbose
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atl_syntax import validate_atl_string, normalize_atl, parse_atl
from nl2atl import (
    get_llm_client,
    critique_nl_atl_pair,
    paraphrase_nl,
    load_templates_config,
    LLMClient,
    generate_nl_statements,
    generate_nl_atl_pair,
    generate_nl_atl_batch,
    translate_nl_to_atl,
)
from sample_store import (
    ATLSample,
    SyntaxNormalizer,
    SampleParser,
    Domain,
    DOMAIN_KEYWORDS,
)


# =============================================================================
# Generation Modes
# =============================================================================

class GenerationMode(str, Enum):
    """Dataset generation modes."""
    TEMPLATE = "template"    # Pure template-based generation
    LLM = "llm"              # LLM generates NL, then translates to ATL
    HYBRID = "hybrid"        # Template base with LLM paraphrasing


# =============================================================================
# Legacy Compatibility - NLATLPair maps to ATLSample
# =============================================================================


def create_sample(
    nl_statement: str,
    atl_formula: str,
    template_id: str = "",
    llm_provider: str = "none",
    paraphrase_of: Optional[str] = None,
    critique_ok: bool = True,
    critique_issues: Optional[List[str]] = None,
    confidence: float = 1.0,
) -> ATLSample:
    """
    Create an ATLSample from generation parameters.
    
    This is the factory function for generated samples, ensuring
    consistent format with parsed samples from sample_store.
    """
    # Normalize formula
    atl_ascii = SyntaxNormalizer.to_ascii(atl_formula)
    atl_unicode = SyntaxNormalizer.to_unicode(atl_formula)
    
    # Extract components
    agents = SyntaxNormalizer.extract_agents(atl_formula)
    operators = SyntaxNormalizer.extract_operators(atl_formula)
    atoms = SyntaxNormalizer.extract_atoms(atl_formula)
    
    # Classify domain
    domain = SampleParser.classify_domain(nl_statement, atl_formula).value
    
    # Generate ID
    sample_id = SampleParser.generate_id(nl_statement, atl_formula)
    
    # Build generation context
    generation = {
        "provider": llm_provider,
    }
    if paraphrase_of:
        generation["paraphrase_of"] = paraphrase_of
    
    return ATLSample(
        id=sample_id,
        nl_statement=nl_statement,
        atl_formula=atl_ascii,
        atl_unicode=atl_unicode,
        domain=domain,
        template_id=template_id,
        agents=agents,
        operators=operators,
        atoms=atoms,
        generation=generation,
        syntax_valid=critique_ok,  # Use critique result as initial validation
        confidence=confidence if critique_ok else None,
        verification_notes=critique_issues or [],
    )


# Backward compatibility alias
NLATLPair = ATLSample


# =============================================================================
# Default Phrases and Coalitions
# =============================================================================

# Sample atomic propositions with English descriptions
DEFAULT_ATOM_PHRASES: Dict[str, str] = {
    "crash": "the system crashes",
    "error": "an error occurs",
    "safe": "the system is safe",
    "goal": "the goal is reached",
    "goal_reached": "the goal is reached",
    "request": "a request is made",
    "response": "a response is given",
    "alarm": "the alarm is triggered",
    "stable": "the system is stable",
    "temp_stable": "the temperature is stable",
    "power": "the power is on",
    "power_on": "the power is on",
    "connected": "the connection is established",
    "authenticated": "the user is authenticated",
    "completed": "the task is completed",
    "waiting": "the system is waiting",
    "active": "the process is active",
    "idle": "the system is idle",
    "locked": "the resource is locked",
    "ready": "the system is ready",
    "running": "the process is running",
    "stopped": "the process is stopped",
    "timeout": "a timeout occurs",
    "success": "the operation succeeds",
    "failure": "the operation fails",
}

# Sample coalition descriptions
DEFAULT_COALITION_PHRASES: Dict[str, str] = {
    "1": "agent 1",
    "2": "agent 2",
    "3": "agent 3",
    "1,2": "agents 1 and 2",
    "1,3": "agents 1 and 3",
    "2,3": "agents 2 and 3",
    "1,2,3": "agents 1, 2, and 3",
    "robot": "the robot",
    "controller": "the controller",
    "server_team": "the server team",
    "driver": "the driver",
}


# =============================================================================
# Sampling Utilities
# =============================================================================


def sample_coalitions(
    max_agents: int = 3, 
    num_samples: int = 5,
    include_named: bool = True,
) -> List[set]:
    """
    Generate sample coalitions.
    
    Args:
        max_agents: Maximum number of numeric agents
        num_samples: Number of coalitions to generate
        include_named: Include named agents (robot, controller, etc.)
        
    Returns:
        List of coalition sets (each set contains agent identifiers)
    """
    coalitions = []

    # Single numeric agents
    for i in range(1, max_agents + 1):
        coalitions.append({str(i)})

    # Pairs of numeric agents
    for i in range(1, max_agents + 1):
        for j in range(i + 1, max_agents + 1):
            coalitions.append({str(i), str(j)})

    # Full coalition
    if max_agents > 1:
        coalitions.append({str(i) for i in range(1, max_agents + 1)})

    # Named agents
    if include_named:
        named_agents = ["robot", "controller", "server_team", "driver"]
        for agent in named_agents:
            coalitions.append({agent})

    # Shuffle and return requested number
    random.shuffle(coalitions)
    return coalitions[:num_samples]


def sample_atoms(
    num_atoms: int = 2,
    atom_phrases: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """
    Sample atomic propositions with their English phrases.
    
    Args:
        num_atoms: Number of atoms to sample
        atom_phrases: Custom atom→phrase mapping
        
    Returns:
        List of (atom_name, english_phrase) tuples
    """
    if atom_phrases is None:
        atom_phrases = DEFAULT_ATOM_PHRASES
    
    atoms = list(atom_phrases.items())
    random.shuffle(atoms)
    return atoms[:num_atoms]


def coalition_to_nl(coalition: set) -> str:
    """
    Convert a coalition set to a natural language phrase.
    
    Args:
        coalition: Set of agent identifiers
        
    Returns:
        English phrase describing the coalition
    """
    agents = sorted(coalition, key=lambda x: (x.isdigit(), x))
    
    if len(agents) == 1:
        agent = agents[0]
        # Check if it's a named agent
        if agent.isdigit():
            return f"agent {agent}"
        elif agent in DEFAULT_COALITION_PHRASES:
            return DEFAULT_COALITION_PHRASES[agent]
        else:
            return f"the {agent}"
    elif len(agents) == 2:
        def format_agent(a):
            return f"agent {a}" if a.isdigit() else f"the {a}"
        return f"{format_agent(agents[0])} and {format_agent(agents[1])}"
    else:
        # Three or more agents
        formatted = []
        for a in agents:
            formatted.append(f"agent {a}" if a.isdigit() else a)
        return "agents " + ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def instantiate_template(
    template: Dict[str, str],
    coalition: set,
    atoms: List[Tuple[str, str]],
) -> Tuple[str, str]:
    """
    Instantiate an ATL template with concrete values.
    
    Args:
        template: Template dict with 'atl_template' and 'nl_template'
        coalition: Set of agent identifiers
        atoms: List of (atom_name, phrase) tuples
        
    Returns:
        Tuple of (atl_formula, nl_sentence)
    """
    atl = template["atl_template"]
    nl = template["nl_template"]

    # Replace coalition
    coalition_str = ",".join(sorted(coalition, key=lambda x: (x.isdigit(), x)))
    atl = atl.replace("{coalition}", coalition_str)
    nl = nl.replace("{coalition_nl}", coalition_to_nl(coalition))

    # Replace atoms
    atom_placeholders = ["{p}", "{q}", "{r}"]
    phrase_placeholders = ["{p_nl}", "{q_nl}", "{r_nl}"]

    for i, (atom_name, phrase) in enumerate(atoms):
        if i < len(atom_placeholders):
            atl = atl.replace(atom_placeholders[i], atom_name)
            nl = nl.replace(phrase_placeholders[i], phrase)

    return atl, nl


# =============================================================================
# Generation Configuration
# =============================================================================


@dataclass
class GenerationConfig:
    """
    Configuration for dataset generation.
    
    Attributes:
        num_examples: Target number of examples to generate
        mode: Generation mode (template, llm, or hybrid)
        
        # Template mode options
        paraphrases_per_template: Number of paraphrases per base pair
        
        # LLM mode options  
        nl_temperature: Temperature for NL generation (higher = more creative)
        atl_temperature: Temperature for ATL translation (lower = more accurate)
        domains: List of domains to sample from (None = all)
        patterns: List of patterns to sample from (None = all)
        
        # Common options
        use_llm_paraphrasing: Enable LLM-based paraphrasing (for hybrid/template modes)
        use_cross_checking: Enable LLM-based quality checking
        verification_provider: Optional provider override for verification
        llm_provider: Provider for LLM calls
        max_agents: Maximum agents in coalitions
        output_format: Output format (jsonl or csv)
        seed: Random seed for reproducibility
        verbose: Print progress information
    """
    num_examples: int = 100
    mode: GenerationMode = GenerationMode.TEMPLATE
    
    # Template mode options
    paraphrases_per_template: int = 3
    
    # LLM mode options
    nl_temperature: float = 0.9
    atl_temperature: float = 0.1
    domains: Optional[List[str]] = None
    patterns: Optional[List[str]] = None
    
    # Common options
    use_llm_paraphrasing: bool = True
    use_cross_checking: bool = True
    verification_provider: Optional[str] = None
    llm_provider: str = "openai"
    max_agents: int = 3
    output_format: str = "jsonl"
    seed: Optional[int] = None
    verbose: bool = False
    
    # Advanced options
    include_named_agents: bool = True
    min_coalition_size: int = 1
    max_coalition_size: int = 5
    max_nesting_depth: int = 4
    use_domain_vocabulary: bool = True  # Use domain-specific atoms/agents


# =============================================================================
# Dataset Generator
# =============================================================================


class DatasetGenerator:
    """
    Generator for NL-ATL pair datasets.
    
    Supports three generation modes:
    
    1. TEMPLATE: Pure template-based generation (fast, no API needed)
       - Instantiates templates with sampled coalitions and atoms
       - Uses NL template variations for diversity
       
    2. LLM: LLM-generated pairs (creative, requires API)
       - Generates NL statements with high temperature
       - Translates to ATL with low temperature for accuracy
       
    3. HYBRID: Template base with LLM paraphrasing
       - Starts with template instantiation
       - Uses LLM to paraphrase for variety
    
    Example:
        >>> config = GenerationConfig(num_examples=50, mode=GenerationMode.TEMPLATE)
        >>> generator = DatasetGenerator(config)
        >>> pairs = generator.generate()
    """

    def __init__(
        self,
        config: GenerationConfig,
        client: Optional[LLMClient] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            config: Generation configuration
            client: Optional pre-configured LLM client
        """
        self.config = config
        self.client = client
        self.verification_client: Optional[LLMClient] = (
            client if client and (config.verification_provider is None or config.verification_provider == config.llm_provider) else None
        )

        if config.seed is not None:
            random.seed(config.seed)

        # Load templates from config
        templates_config = load_templates_config()
        self.templates = templates_config.get("templates", [])
        self.atom_phrases = templates_config.get("atom_phrases", DEFAULT_ATOM_PHRASES)
        self.coalition_phrases = templates_config.get(
            "coalition_phrases", DEFAULT_COALITION_PHRASES
        )
        
        if not self.templates:
            self._log("Warning: No templates found in config, using defaults")
            self.templates = self._get_default_templates()

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            print(message)

    def _get_generation_client(self) -> LLMClient:
        """Get or create LLM client for generation/paraphrasing."""
        if self.client is None:
            self.client = get_llm_client(self.config.llm_provider)
        return self.client

    def _get_verification_client(self) -> LLMClient:
        """Get or create LLM client for verification/cross-checking."""
        if self.verification_client is None:
            provider = self.config.verification_provider or self.config.llm_provider
            self.verification_client = get_llm_client(provider)
        return self.verification_client

    def _get_default_templates(self) -> List[Dict[str, str]]:
        """Get default templates if config not found."""
        return [
            {
                "id": "safety_basic",
                "atl_template": "⟨⟨{coalition}⟩⟩ G ¬{p}",
                "nl_templates": ["The coalition {coalition_nl} can ensure that {p_nl} never happens."],
            },
            {
                "id": "reachability_basic",
                "atl_template": "⟨⟨{coalition}⟩⟩ F {p}",
                "nl_templates": ["The coalition {coalition_nl} can eventually make {p_nl} happen."],
            },
            {
                "id": "safety_always",
                "atl_template": "⟨⟨{coalition}⟩⟩ G {p}",
                "nl_templates": ["The coalition {coalition_nl} can guarantee that {p_nl} always holds."],
            },
        ]

    def _get_domain_vocabulary(self, domain: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Get domain-specific agents and atoms.
        
        Args:
            domain: Domain name
            
        Returns:
            Tuple of (agent_list, atom_phrases_dict)
        """
        templates_config = load_templates_config()
        domains = templates_config.get("domains", {})
        
        if domain in domains:
            domain_info = domains[domain]
            agents = domain_info.get("agents", [])
            atoms = domain_info.get("atoms", {})
            return agents, atoms
        
        return [], {}

    def _sample_domain_coalition(self, domain: Optional[str] = None) -> set:
        """
        Sample a coalition, optionally from a specific domain.
        
        Args:
            domain: Optional domain to sample from
            
        Returns:
            Coalition set
        """
        if domain and self.config.use_domain_vocabulary:
            agents, _ = self._get_domain_vocabulary(domain)
            if agents:
                # Sample 1-2 agents from domain
                num_agents = random.randint(1, min(2, len(agents)))
                selected = random.sample(agents, num_agents)
                return set(selected)
        
        # Fall back to generic sampling
        coalitions = sample_coalitions(
            max_agents=self.config.max_agents,
            num_samples=1,
            include_named=self.config.include_named_agents,
        )
        return coalitions[0] if coalitions else {"1"}

    def _sample_domain_atoms(
        self, 
        num_atoms: int, 
        domain: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Sample atoms, optionally from a specific domain.
        
        Args:
            num_atoms: Number of atoms to sample
            domain: Optional domain to sample from
            
        Returns:
            List of (atom_name, phrase) tuples
        """
        if domain and self.config.use_domain_vocabulary:
            _, domain_atoms = self._get_domain_vocabulary(domain)
            if domain_atoms:
                atoms = list(domain_atoms.items())
                random.shuffle(atoms)
                return atoms[:num_atoms]
        
        # Fall back to default atoms
        return sample_atoms(num_atoms, self.atom_phrases)

    def generate_base_pairs(self) -> List[ATLSample]:
        """
        Generate base NL-ATL pairs from templates.
        
        Uses the expanded template format with multiple NL variations.
        
        Returns:
            List of ATLSample objects
        """
        pairs = []
        templates_config = load_templates_config()
        available_domains = list(templates_config.get("domains", {}).keys())
        
        # Calculate how many instantiations per template
        base_per_template = max(
            1, 
            self.config.num_examples // len(self.templates) // 2
        )

        for template in self.templates:
            # Get NL templates (support both old and new format)
            nl_templates = template.get("nl_templates", [])
            if not nl_templates and "nl_template" in template:
                nl_templates = [template["nl_template"]]
            
            if not nl_templates:
                self._log(f"Skipping template {template.get('id', 'unknown')}: no NL templates")
                continue

            for _ in range(base_per_template):
                # Sample a domain for vocabulary diversity
                domain = random.choice(available_domains) if available_domains else None
                
                # Sample coalition (domain-aware)
                coalition = self._sample_domain_coalition(domain)
                
                # Determine how many atoms we need
                atl_template = template["atl_template"]
                num_atoms = sum(1 for ph in ["{p}", "{q}", "{r}"] if ph in atl_template)
                atoms = self._sample_domain_atoms(num_atoms, domain)
                
                if len(atoms) < num_atoms:
                    # Fall back to default atoms if domain doesn't have enough
                    atoms = sample_atoms(num_atoms, self.atom_phrases)

                # Choose a random NL template variation
                nl_template_text = random.choice(nl_templates)
                template_with_nl = {
                    "atl_template": template["atl_template"],
                    "nl_template": nl_template_text,
                    "id": template.get("id", "unknown"),
                }

                try:
                    atl, nl = instantiate_template(template_with_nl, coalition, atoms)
                except Exception as e:
                    self._log(f"Template instantiation failed: {e}")
                    continue

                # Validate ATL
                is_valid, errors = validate_atl_string(
                    atl, 
                    max_depth=self.config.max_nesting_depth,
                    max_coalition_size=self.config.max_coalition_size
                )
                if not is_valid:
                    self._log(f"Skipping invalid formula: {atl} - {errors}")
                    continue

                pair = create_sample(
                    nl_statement=nl,
                    atl_formula=atl,
                    template_id=template.get("id", "unknown"),
                    llm_provider="none",
                )
                pairs.append(pair)

        return pairs

    def paraphrase_pairs(self, pairs: List[ATLSample]) -> List[ATLSample]:
        """
        Generate paraphrased versions of NL-ATL pairs.
        
        Args:
            pairs: Base pairs to paraphrase
            
        Returns:
            Extended list with original and paraphrased pairs
        """
        if not self.config.use_llm_paraphrasing:
            return pairs

        client = self._get_generation_client()
        result = list(pairs)  # Keep originals

        for i, pair in enumerate(pairs):
            self._log(f"  Paraphrasing {i+1}/{len(pairs)}: {pair.nl_statement[:40]}...")
            
            try:
                paraphrases = paraphrase_nl(
                    pair.nl_statement,
                    num_paraphrases=self.config.paraphrases_per_template,
                    client=client,
                )

                for para in paraphrases:
                    new_pair = create_sample(
                        nl_statement=para,
                        atl_formula=pair.atl_formula,
                        template_id=pair.template_id or "",
                        llm_provider=self.config.llm_provider,
                        paraphrase_of=pair.nl_statement,
                    )
                    result.append(new_pair)

            except Exception as e:
                self._log(f"  Paraphrasing failed: {e}")

        return result

    def cross_check_pairs(self, pairs: List[ATLSample]) -> List[ATLSample]:
        """
        Cross-check pairs using LLM critique.
        
        Args:
            pairs: Pairs to check
            
        Returns:
            Filtered list of pairs that passed checking
        """
        if not self.config.use_cross_checking:
            return pairs

        client = self._get_verification_client()
        checked = []

        for i, pair in enumerate(pairs):
            self._log(f"  Checking {i+1}/{len(pairs)}: {pair.nl_statement[:40]}...")
            
            try:
                critique = critique_nl_atl_pair(
                    pair.nl_statement,
                    pair.atl_formula,
                    client=client,
                )

                critique_ok = critique.get("ok", False)
                critique_issues = critique.get("issues", [])

                if critique_ok:
                    pair.syntax_valid = True
                    pair.confidence = 1.0
                    pair.verification_notes = []
                    checked.append(pair)
                elif critique.get("suggested_fix"):
                    # Try the suggested fix
                    fix = critique["suggested_fix"]
                    is_valid, _ = validate_atl_string(fix)
                    if is_valid:
                        pair.atl_formula = SyntaxNormalizer.to_ascii(fix)
                        pair.atl_unicode = SyntaxNormalizer.to_unicode(fix)
                        pair.syntax_valid = True
                        pair.verification_notes = ["Fixed by LLM"]
                        pair.confidence = 0.8
                        checked.append(pair)
                    else:
                        self._log(f"  Suggested fix invalid: {fix}")
                else:
                    self._log(f"  Pair failed critique")

            except Exception as e:
                self._log(f"  Critique failed: {e}")
                # Keep pair but mark as unchecked
                pair.syntax_valid = False
                pair.verification_notes = [f"Critique error: {str(e)}"]
                pair.confidence = 0.5
                checked.append(pair)

        return checked

    def generate_llm_pairs(self) -> List[ATLSample]:
        """
        Generate NL-ATL pairs using LLM (two-stage process).
        
        Stage 1: Generate creative NL statements (high temperature)
        Stage 2: Translate to ATL (low temperature for accuracy)
        
        Returns:
            List of ATLSample objects
        """
        client = self._get_generation_client()
        pairs = []
        
        # Get available domains and patterns
        templates_config = load_templates_config()
        available_domains = list(templates_config.get("domains", {}).keys())
        available_patterns = ["safety", "liveness", "response", "until", "fairness"]
        
        # Use configured domains/patterns or all available
        domains = self.config.domains or available_domains
        patterns = self.config.patterns or available_patterns
        
        self._log(f"  Using domains: {domains}")
        self._log(f"  Using patterns: {patterns}")
        self._log(f"  NL temperature: {self.config.nl_temperature}")
        self._log(f"  ATL temperature: {self.config.atl_temperature}")
        
        attempts = 0
        max_attempts = self.config.num_examples * 3
        
        while len(pairs) < self.config.num_examples and attempts < max_attempts:
            attempts += 1
            
            # Sample domain and pattern for diversity
            domain = random.choice(domains) if domains else None
            pattern = random.choice(patterns) if patterns else None
            
            self._log(f"  [{len(pairs)+1}/{self.config.num_examples}] "
                     f"Generating (domain={domain}, pattern={pattern})...")
            
            try:
                # Stage 1: Generate NL with high temperature
                nl_statements = generate_nl_statements(
                    num_statements=1,
                    domain=domain,
                    pattern=pattern,
                    temperature=self.config.nl_temperature,
                    client=client,
                )
                
                if not nl_statements:
                    self._log(f"    Failed to generate NL statement")
                    continue
                
                nl_text = nl_statements[0]
                self._log(f"    NL: {nl_text[:60]}...")
                
                # Stage 2: Translate to ATL with low temperature
                formulas = translate_nl_to_atl(
                    nl_text,
                    config={"temperature": self.config.atl_temperature},
                    client=client,
                    validate=True,
                )
                
                if not formulas:
                    self._log(f"    Failed to translate to valid ATL")
                    continue
                
                atl_formula = formulas[0]
                self._log(f"    ATL: {atl_formula}")
                
                # Create sample
                pair = create_sample(
                    nl_statement=nl_text,
                    atl_formula=atl_formula,
                    template_id=f"llm_{domain}_{pattern}" if domain and pattern else "llm_generated",
                    llm_provider=self.config.llm_provider,
                )
                pairs.append(pair)
                
            except Exception as e:
                self._log(f"    Error: {e}")
                continue
        
        return pairs

    def generate(self) -> List[ATLSample]:
        """
        Run the full generation pipeline based on configured mode.
        
        Modes:
        - TEMPLATE: Pure template-based generation
        - LLM: LLM generates NL, then translates to ATL
        - HYBRID: Template base with LLM paraphrasing
        
        Returns:
            List of generated NL-ATL pairs
        """
        self._log("=" * 60)
        self._log("NL-ATL Dataset Generation")
        self._log(f"Mode: {self.config.mode.value}")
        self._log("=" * 60)
        
        pairs = []
        
        if self.config.mode == GenerationMode.TEMPLATE:
            # Pure template mode - no LLM needed
            self._log("\n[1/2] Generating pairs from templates...")
            pairs = self.generate_base_pairs()
            self._log(f"  Generated {len(pairs)} template-based pairs")
            
            # Skip LLM paraphrasing in pure template mode
            self._log("\n[2/2] Cross-checking skipped (template mode)")
            
        elif self.config.mode == GenerationMode.LLM:
            # LLM generation mode
            self._log("\n[1/2] Generating pairs using LLM (two-stage)...")
            pairs = self.generate_llm_pairs()
            self._log(f"  Generated {len(pairs)} LLM-based pairs")
            
            # Cross-check LLM-generated pairs
            if self.config.use_cross_checking:
                self._log("\n[2/2] Cross-checking LLM-generated pairs...")
                pairs = self.cross_check_pairs(pairs)
                self._log(f"  Pairs passing cross-check: {len(pairs)}")
            else:
                self._log("\n[2/2] Cross-checking skipped (disabled)")
            
        elif self.config.mode == GenerationMode.HYBRID:
            # Hybrid mode - template + paraphrasing
            self._log("\n[1/3] Generating base pairs from templates...")
            pairs = self.generate_base_pairs()
            self._log(f"  Generated {len(pairs)} base pairs")
            
            # Paraphrase
            if self.config.use_llm_paraphrasing:
                self._log("\n[2/3] Generating paraphrases...")
                pairs = self.paraphrase_pairs(pairs)
                self._log(f"  Total pairs after paraphrasing: {len(pairs)}")
            else:
                self._log("\n[2/3] Paraphrasing skipped (disabled)")
            
            # Cross-check
            if self.config.use_cross_checking:
                self._log("\n[3/3] Cross-checking pairs...")
                pairs = self.cross_check_pairs(pairs)
                self._log(f"  Pairs passing cross-check: {len(pairs)}")
            else:
                self._log("\n[3/3] Cross-checking skipped (disabled)")

        # Limit to requested number
        if len(pairs) > self.config.num_examples:
            random.shuffle(pairs)
            pairs = pairs[:self.config.num_examples]
            self._log(f"\nTrimmed to {len(pairs)} examples")

        self._log("\n" + "=" * 60)
        self._log(f"Generation complete: {len(pairs)} pairs")
        self._log("=" * 60)

        return pairs


# =============================================================================
# File I/O
# =============================================================================


def save_dataset(
    pairs: List[ATLSample], 
    output_path: Path, 
    format: str = "jsonl"
) -> None:
    """
    Save dataset to file.
    
    Args:
        pairs: List of ATLSample objects
        output_path: Output file path
        format: Output format ("jsonl" or "csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    elif format == "csv":
        fieldnames = [
            "id", "nl_statement", "atl_formula", "atl_unicode", "domain",
            "template_id", "agents", "operators", "atoms", 
            "syntax_valid", "confidence"
        ]
        
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for pair in pairs:
                row = pair.to_dict()
                row["agents"] = ",".join(row.get("agents", []))
                row["operators"] = ",".join(row.get("operators", []))
                row["atoms"] = ",".join(row.get("atoms", []))
                writer.writerow(row)

    else:
        raise ValueError(f"Unknown format: {format}. Use 'jsonl' or 'csv'.")


def load_dataset(path: Path) -> List[ATLSample]:
    """
    Load dataset from JSONL file.
    
    Args:
        path: Path to dataset file
        
    Returns:
        List of ATLSample objects
    """
    path = Path(path)
    pairs = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                pairs.append(ATLSample.from_dict(data))
    
    return pairs


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate NL-ATL pair datasets for training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Template mode (fast, no API needed)
  python -m dataset_gen --mode template --num-examples 100

  # LLM mode (creative, requires API)
  python -m dataset_gen --mode llm --num-examples 50 --nl-temp 0.9 --atl-temp 0.1

  # Hybrid mode (template + paraphrasing, requires API)
  python -m dataset_gen --mode hybrid --num-examples 200 --verbose

  # LLM mode with specific domains
  python -m dataset_gen --mode llm --num-examples 30 --domains robotics,safety_critical

  # Generate with cross-checking for quality
  python -m dataset_gen --mode llm --num-examples 50 --crosscheck --verbose

  # Reproducible generation
  python -m dataset_gen --mode template --num-examples 100 --seed 42 --out data/test.jsonl
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="template",
        choices=["template", "llm", "hybrid"],
        help="Generation mode: 'template' (fast, no API), 'llm' (creative, API needed), 'hybrid' (template + paraphrasing)",
    )
    
    parser.add_argument(
        "--num-examples", "-n",
        type=int,
        default=100,
        help="Number of examples to generate (default: 100)",
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default="data/nl_atl_dataset.jsonl",
        help="Output file path (default: data/nl_atl_dataset.jsonl)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    
    # LLM options
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "azure", "mock"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--verification-provider",
        type=str,
        default=None,
        choices=["openai", "azure", "mock"],
        help="LLM provider to use for verification/cross-checking (default: match --provider)",
    )
    
    # Temperature settings for LLM mode
    parser.add_argument(
        "--nl-temp",
        type=float,
        default=0.9,
        help="Temperature for NL generation in LLM mode (default: 0.9, higher = more creative)",
    )
    parser.add_argument(
        "--atl-temp",
        type=float,
        default=0.1,
        help="Temperature for ATL translation in LLM mode (default: 0.1, lower = more accurate)",
    )
    
    # Domain and pattern filtering
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated list of domains to use (e.g., 'robotics,safety_critical'). "
             "Available: autonomous_systems, distributed_systems, safety_critical, robotics, "
             "access_control, industrial_control, networking, transaction_processing",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default=None,
        help="Comma-separated list of patterns to use (e.g., 'safety,liveness'). "
             "Available: safety, liveness, response, until, fairness",
    )
    
    # Paraphrasing options (for hybrid mode)
    parser.add_argument(
        "--paraphrases",
        type=int,
        default=3,
        help="Paraphrases per template in hybrid mode (default: 3)",
    )
    parser.add_argument(
        "--no-paraphrase",
        action="store_true",
        help="Disable LLM-based paraphrasing in hybrid mode",
    )
    
    # Cross-checking
    parser.add_argument(
        "--crosscheck",
        action="store_true",
        help="Enable LLM-based cross-checking for quality validation",
    )
    parser.add_argument(
        "--no-crosscheck",
        action="store_true",
        help="Disable LLM-based cross-checking (default for template mode)",
    )
    
    # Other options
    parser.add_argument(
        "--max-agents",
        type=int,
        default=3,
        help="Maximum agents in coalitions (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()
    
    # Parse domains and patterns
    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    
    patterns = None
    if args.patterns:
        patterns = [p.strip() for p in args.patterns.split(",")]
    
    # Determine cross-checking default based on mode
    use_crosscheck = args.crosscheck
    if args.no_crosscheck:
        use_crosscheck = False
    
    # Parse mode
    mode = GenerationMode(args.mode)

    # Build configuration
    config = GenerationConfig(
        num_examples=args.num_examples,
        mode=mode,
        paraphrases_per_template=args.paraphrases,
        nl_temperature=args.nl_temp,
        atl_temperature=args.atl_temp,
        domains=domains,
        patterns=patterns,
        use_llm_paraphrasing=not args.no_paraphrase,
        use_cross_checking=use_crosscheck,
        verification_provider=args.verification_provider,
        llm_provider=args.provider,
        max_agents=args.max_agents,
        output_format=args.format,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Create generator and run
    generator = DatasetGenerator(config)

    try:
        pairs = generator.generate()
    except ValueError as e:
        # Handle missing API key gracefully
        if "API key" in str(e):
            if mode == GenerationMode.TEMPLATE:
                # Template mode shouldn't need API
                raise
            print(f"Warning: {e}")
            print("Falling back to template-only generation...")
            config.mode = GenerationMode.TEMPLATE
            config.use_llm_paraphrasing = False
            config.use_cross_checking = False
            generator = DatasetGenerator(config)
            pairs = generator.generate()
        else:
            raise

    # Save output
    output_path = Path(args.out)
    save_dataset(pairs, output_path, args.format)

    print(f"\n✓ Generated {len(pairs)} NL-ATL pairs")
    print(f"✓ Saved to: {output_path}")
    
    # Print summary statistics
    if args.verbose:
        domains_used = set(p.domain for p in pairs if p.domain)
        templates_used = set(p.template_id for p in pairs if p.template_id)
        print(f"\nSummary:")
        print(f"  Domains: {len(domains_used)} ({', '.join(list(domains_used)[:5])}{'...' if len(domains_used) > 5 else ''})")
        print(f"  Templates: {len(templates_used)}")


# Allow running as module
if __name__ == "__main__":
    main()
