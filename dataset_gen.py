"""
Dataset Generation Module

Generates synthetic NL-ATL pairs for training and evaluation:
- Template instantiation with concrete coalitions and propositions
- LLM-based paraphrasing and enrichment
- Cross-checking and quality filtering
- CLI interface for batch generation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from atl_syntax import validate_atl_string, normalize_atl, parse_atl
from nl2atl import (
    get_llm_client,
    critique_nl_atl_pair,
    paraphrase_nl,
    load_templates_config,
    LLMClient,
)

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class NLATLPair:
    """A natural language - ATL formula pair."""

    nl_text: str
    atl_formula: str
    agents: list[str] = field(default_factory=list)
    coalition: str = ""
    atoms: list[str] = field(default_factory=list)
    template_id: str = ""
    llm_provider: str = ""
    critique_ok: bool = True
    critique_issues: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "nl_text": self.nl_text,
            "atl_formula": self.atl_formula,
            "agents": self.agents,
            "coalition": self.coalition,
            "atoms": self.atoms,
            "template_id": self.template_id,
            "llm_provider": self.llm_provider,
            "critique_ok": self.critique_ok,
            "critique_issues": self.critique_issues,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# =============================================================================
# Template Instantiation
# =============================================================================


# Sample atomic propositions with English descriptions
DEFAULT_ATOM_PHRASES = {
    "crash": "the system crashes",
    "error": "an error occurs",
    "safe": "the system is safe",
    "goal": "the goal is reached",
    "request": "a request is made",
    "response": "a response is given",
    "alarm": "the alarm is triggered",
    "stable": "the system is stable",
    "power": "the power is on",
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
}


# Sample coalitions with English descriptions
DEFAULT_COALITION_PHRASES = {
    "1": "agent 1",
    "2": "agent 2",
    "3": "agent 3",
    "1,2": "agents 1 and 2",
    "1,3": "agents 1 and 3",
    "2,3": "agents 2 and 3",
    "1,2,3": "agents 1, 2, and 3",
}


def sample_coalitions(max_agents: int = 3, num_samples: int = 5) -> list[set[str]]:
    """
    Generate sample coalitions.
    
    Args:
        max_agents: Maximum number of agents in the system
        num_samples: Number of coalitions to generate
        
    Returns:
        List of coalition sets
    """
    all_agents = [str(i) for i in range(1, max_agents + 1)]
    coalitions = []

    # Include single agents
    for a in all_agents:
        coalitions.append({a})

    # Include some pairs
    for i in range(len(all_agents)):
        for j in range(i + 1, len(all_agents)):
            coalitions.append({all_agents[i], all_agents[j]})

    # Include full coalition
    if max_agents > 1:
        coalitions.append(set(all_agents))

    # Shuffle and return requested number
    random.shuffle(coalitions)
    return coalitions[:num_samples]


def sample_atoms(num_atoms: int = 2) -> list[tuple[str, str]]:
    """
    Sample atomic propositions with their English phrases.
    
    Args:
        num_atoms: Number of atoms to sample
        
    Returns:
        List of (atom_name, english_phrase) tuples
    """
    atoms = list(DEFAULT_ATOM_PHRASES.items())
    random.shuffle(atoms)
    return atoms[:num_atoms]


def coalition_to_nl(coalition: set[str]) -> str:
    """Convert a coalition set to a natural language phrase."""
    agents = sorted(coalition)
    if len(agents) == 1:
        return f"agent {agents[0]}"
    elif len(agents) == 2:
        return f"agents {agents[0]} and {agents[1]}"
    else:
        return "agents " + ", ".join(agents[:-1]) + f", and {agents[-1]}"


def instantiate_template(
    template: dict,
    coalition: set[str],
    atoms: list[tuple[str, str]],
) -> tuple[str, str]:
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
    coalition_str = ",".join(sorted(coalition))
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
# Dataset Generation Pipeline
# =============================================================================


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""

    num_examples: int = 100
    paraphrases_per_template: int = 3
    use_llm_paraphrasing: bool = True
    use_cross_checking: bool = True
    llm_provider: str = "openai"
    max_agents: int = 3
    output_format: str = "jsonl"
    seed: Optional[int] = None
    verbose: bool = False


class DatasetGenerator:
    """Generator for NL-ATL pair datasets."""

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

        if config.seed is not None:
            random.seed(config.seed)

        # Load templates
        templates_config = load_templates_config()
        self.templates = templates_config.get("templates", [])
        self.atom_phrases = templates_config.get("atom_phrases", DEFAULT_ATOM_PHRASES)
        self.coalition_phrases = templates_config.get(
            "coalition_phrases", DEFAULT_COALITION_PHRASES
        )

    def _get_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self.client is None:
            self.client = get_llm_client(self.config.llm_provider)
        return self.client

    def generate_base_pairs(self) -> list[NLATLPair]:
        """
        Generate base NL-ATL pairs from templates.
        
        Returns:
            List of NLATLPair objects
        """
        pairs = []

        for template in self.templates:
            # Generate multiple instantiations per template
            coalitions = sample_coalitions(
                self.config.max_agents,
                num_samples=max(1, self.config.num_examples // len(self.templates) // 2),
            )

            for coalition in coalitions:
                # Determine how many atoms we need
                atl_template = template["atl_template"]
                num_atoms = sum(1 for ph in ["{p}", "{q}", "{r}"] if ph in atl_template)
                atoms = sample_atoms(num_atoms)

                atl, nl = instantiate_template(template, coalition, atoms)

                # Validate ATL
                is_valid, errors = validate_atl_string(atl)
                if not is_valid:
                    if self.config.verbose:
                        print(f"Skipping invalid formula: {atl} - {errors}")
                    continue

                pair = NLATLPair(
                    nl_text=nl,
                    atl_formula=atl,
                    agents=list(coalition),
                    coalition=",".join(sorted(coalition)),
                    atoms=[a[0] for a in atoms],
                    template_id=template.get("id", ""),
                    llm_provider=self.config.llm_provider if self.config.use_llm_paraphrasing else "none",
                )
                pairs.append(pair)

        return pairs

    def paraphrase_pairs(self, pairs: list[NLATLPair]) -> list[NLATLPair]:
        """
        Generate paraphrased versions of NL-ATL pairs.
        
        Args:
            pairs: Base pairs to paraphrase
            
        Returns:
            Extended list with original and paraphrased pairs
        """
        if not self.config.use_llm_paraphrasing:
            return pairs

        client = self._get_client()
        result = list(pairs)  # Keep originals

        for pair in pairs:
            try:
                paraphrases = paraphrase_nl(
                    pair.nl_text,
                    num_paraphrases=self.config.paraphrases_per_template,
                    client=client,
                )

                for para in paraphrases:
                    new_pair = NLATLPair(
                        nl_text=para,
                        atl_formula=pair.atl_formula,
                        agents=pair.agents.copy(),
                        coalition=pair.coalition,
                        atoms=pair.atoms.copy(),
                        template_id=pair.template_id,
                        llm_provider=self.config.llm_provider,
                        metadata={"paraphrase_of": pair.nl_text},
                    )
                    result.append(new_pair)

            except Exception as e:
                if self.config.verbose:
                    print(f"Paraphrasing failed for: {pair.nl_text[:50]}... - {e}")

        return result

    def cross_check_pairs(self, pairs: list[NLATLPair]) -> list[NLATLPair]:
        """
        Cross-check pairs using LLM critique.
        
        Args:
            pairs: Pairs to check
            
        Returns:
            Filtered list of pairs that passed checking
        """
        if not self.config.use_cross_checking:
            return pairs

        client = self._get_client()
        checked = []

        for pair in pairs:
            try:
                critique = critique_nl_atl_pair(
                    pair.nl_text,
                    pair.atl_formula,
                    client=client,
                )

                pair.critique_ok = critique.get("ok", False)
                pair.critique_issues = critique.get("issues", [])

                if pair.critique_ok:
                    checked.append(pair)
                elif critique.get("suggested_fix"):
                    # Try the suggested fix
                    fix = critique["suggested_fix"]
                    is_valid, _ = validate_atl_string(fix)
                    if is_valid:
                        pair.atl_formula = fix
                        pair.critique_ok = True
                        pair.critique_issues = ["Fixed by LLM"]
                        checked.append(pair)
                    elif self.config.verbose:
                        print(f"Suggested fix invalid: {fix}")
                elif self.config.verbose:
                    print(f"Pair failed critique: {pair.nl_text[:50]}...")

            except Exception as e:
                if self.config.verbose:
                    print(f"Critique failed: {e}")
                # Keep pair but mark as unchecked
                pair.critique_ok = False
                pair.critique_issues = [f"Critique error: {str(e)}"]

        return checked

    def generate(self) -> list[NLATLPair]:
        """
        Run the full generation pipeline.
        
        Returns:
            List of generated NL-ATL pairs
        """
        if self.config.verbose:
            print("Generating base pairs from templates...")

        pairs = self.generate_base_pairs()
        if self.config.verbose:
            print(f"  Generated {len(pairs)} base pairs")

        if self.config.use_llm_paraphrasing:
            if self.config.verbose:
                print("Generating paraphrases...")
            pairs = self.paraphrase_pairs(pairs)
            if self.config.verbose:
                print(f"  Total pairs after paraphrasing: {len(pairs)}")

        if self.config.use_cross_checking:
            if self.config.verbose:
                print("Cross-checking pairs...")
            pairs = self.cross_check_pairs(pairs)
            if self.config.verbose:
                print(f"  Pairs passing cross-check: {len(pairs)}")

        # Limit to requested number
        if len(pairs) > self.config.num_examples:
            random.shuffle(pairs)
            pairs = pairs[: self.config.num_examples]

        return pairs


def save_dataset(pairs: list[NLATLPair], output_path: Path, format: str = "jsonl"):
    """
    Save dataset to file.
    
    Args:
        pairs: List of NL-ATL pairs
        output_path: Output file path
        format: Output format ("jsonl" or "csv")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    elif format == "csv":
        import csv

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "nl_text",
                    "atl_formula",
                    "agents",
                    "coalition",
                    "atoms",
                    "template_id",
                    "llm_provider",
                    "critique_ok",
                    "critique_issues",
                    "confidence",
                ],
            )
            writer.writeheader()
            for pair in pairs:
                row = pair.to_dict()
                row["agents"] = ",".join(row["agents"])
                row["atoms"] = ",".join(row["atoms"])
                row["critique_issues"] = "; ".join(row["critique_issues"])
                del row["metadata"]  # Skip complex field for CSV
                writer.writerow(row)

    else:
        raise ValueError(f"Unknown format: {format}")


def load_dataset(path: Path) -> list[NLATLPair]:
    """
    Load dataset from file.
    
    Args:
        path: Path to dataset file (JSONL format)
        
    Returns:
        List of NLATLPair objects
    """
    pairs = []
    # Known fields in NLATLPair dataclass
    known_fields = {
        "nl_text", "atl_formula", "agents", "coalition", "atoms",
        "template_id", "llm_provider", "critique_ok", "critique_issues",
        "confidence", "metadata"
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Only pass known fields to avoid errors with extra data
                filtered_data = {k: v for k, v in data.items() if k in known_fields}
                pairs.append(NLATLPair(**filtered_data))
    return pairs


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate NL-ATL pair datasets for training and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/nl_atl_dataset.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider (openai, azure, mock)",
    )
    parser.add_argument(
        "--paraphrases",
        type=int,
        default=3,
        help="Number of paraphrases per template instantiation",
    )
    parser.add_argument(
        "--no-paraphrase",
        action="store_true",
        help="Disable LLM-based paraphrasing",
    )
    parser.add_argument(
        "--no-crosscheck",
        action="store_true",
        help="Disable LLM-based cross-checking",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=3,
        help="Maximum number of agents in coalitions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    config = GenerationConfig(
        num_examples=args.num_examples,
        paraphrases_per_template=args.paraphrases,
        use_llm_paraphrasing=not args.no_paraphrase,
        use_cross_checking=not args.no_crosscheck,
        llm_provider=args.provider,
        max_agents=args.max_agents,
        output_format=args.format,
        seed=args.seed,
        verbose=args.verbose,
    )

    generator = DatasetGenerator(config)

    try:
        pairs = generator.generate()
    except ValueError as e:
        # Handle missing API key gracefully
        if "API key" in str(e) and (
            config.use_llm_paraphrasing or config.use_cross_checking
        ):
            print(f"Warning: {e}")
            print("Falling back to template-only generation...")
            config.use_llm_paraphrasing = False
            config.use_cross_checking = False
            generator = DatasetGenerator(config)
            pairs = generator.generate()
        else:
            raise

    output_path = Path(args.out)
    save_dataset(pairs, output_path, args.format)

    print(f"Generated {len(pairs)} NL-ATL pairs")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
