"""
NL2ATL Translation Module

Provides prompting and LLM-based translation from Natural Language to ATL formulas:
- Prompt building with few-shot examples
- Provider-agnostic LLM client abstraction
- Translation and cross-checking functions
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from atl_syntax import is_valid, parse_atl, validate_atl_string

# =============================================================================
# Configuration Loading
# =============================================================================

CONFIG_DIR = Path(__file__).parent / "config"


def load_templates_config() -> dict:
    """Load templates and few-shot examples from config file."""
    config_path = CONFIG_DIR / "templates_atl.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"templates": [], "few_shot_examples": []}


def load_fragment_config() -> dict:
    """Load ATL fragment configuration from YAML file."""
    config_path = CONFIG_DIR / "atl_fragment.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            # If PyYAML not available, return defaults
            pass
    return {
        "temporal_operators": ["X", "F", "G", "U"],
        "boolean_connectives": ["∧", "∨", "¬", "→"],
        "constraints": {"max_nesting_depth": 4, "max_coalition_size": 5},
    }


# =============================================================================
# Prompt Building
# =============================================================================

ATL_SYNTAX_DESCRIPTION = """
ATL (Alternating-time Temporal Logic) Syntax:

Strategic Modality:
- ⟨⟨A⟩⟩ φ : Coalition A has a strategy to ensure φ
- A is a comma-separated list of agent identifiers (e.g., ⟨⟨1,2⟩⟩ or ⟨⟨robot⟩⟩)

Temporal Operators:
- X φ : φ holds in the next state (neXt)
- F φ : φ eventually holds (Future)
- G φ : φ always holds (Globally)
- φ U ψ : φ holds until ψ becomes true (Until)

Boolean Connectives:
- ¬φ : negation (NOT)
- φ ∧ ψ : conjunction (AND)
- φ ∨ ψ : disjunction (OR)
- φ → ψ : implication (IF-THEN)

Atomic Propositions:
- Simple identifiers like: crash, goal_reached, request, response, safe, error

Examples of valid ATL formulas:
- ⟨⟨1,2⟩⟩ G ¬crash
- ⟨⟨robot⟩⟩ F goal_reached
- ⟨⟨controller⟩⟩ (temp_stable U alarm)
- ⟨⟨server_team⟩⟩ G (request → F response)
""".strip()


def build_translation_prompt(
    nl_text: str,
    few_shot_examples: Optional[list[dict]] = None,
    include_syntax_description: bool = True,
) -> str:
    """
    Build a structured prompt for NL to ATL translation.
    
    Args:
        nl_text: The natural language requirement to translate
        few_shot_examples: List of {"nl": ..., "atl": ...} examples
        include_syntax_description: Whether to include ATL syntax reference
        
    Returns:
        A formatted prompt string for the LLM
    """
    if few_shot_examples is None:
        config = load_templates_config()
        few_shot_examples = config.get("few_shot_examples", [])

    parts = []

    # System instruction
    parts.append(
        "You are an expert in formal verification and temporal logic. "
        "Your task is to translate natural language requirements into ATL "
        "(Alternating-time Temporal Logic) formulas."
    )

    # ATL syntax description
    if include_syntax_description:
        parts.append("\n" + ATL_SYNTAX_DESCRIPTION)

    # Few-shot examples
    if few_shot_examples:
        parts.append("\nHere are some examples of NL to ATL translations:\n")
        for ex in few_shot_examples:
            parts.append(f"NL: {ex['nl']}")
            parts.append(f"ATL: {ex['atl']}\n")

    # The actual request
    parts.append(
        "\nNow translate the following requirement into a single ATL formula. "
        "Output ONLY the ATL formula, nothing else.\n"
    )
    parts.append(f"NL: {nl_text}")
    parts.append("ATL:")

    return "\n".join(parts)


def build_critique_prompt(nl_text: str, atl_text: str) -> str:
    """
    Build a prompt for critiquing an NL-ATL pair.
    
    Args:
        nl_text: The natural language requirement
        atl_text: The proposed ATL formula
        
    Returns:
        A formatted prompt for critique
    """
    prompt = f"""You are an expert in formal verification and temporal logic.

Analyze whether the following ATL formula correctly captures the natural language requirement.

{ATL_SYNTAX_DESCRIPTION}

Natural Language Requirement:
{nl_text}

Proposed ATL Formula:
{atl_text}

Analyze this pair and respond in the following JSON format:
{{
    "ok": true/false,
    "issues": ["list of any issues found"],
    "explanation": "brief explanation of your analysis",
    "suggested_fix": "corrected ATL formula if needed, or null if the original is correct"
}}

Respond with ONLY the JSON object, no other text."""

    return prompt


# =============================================================================
# LLM Client Abstraction
# =============================================================================


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)
    raw_response: Optional[dict] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with generated text and metadata
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            model: Model to use
            base_url: Optional custom base URL for API
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model=self.model,
            provider="openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    @property
    def provider_name(self) -> str:
        return "openai"


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI API client implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: str = "gpt-4o-mini",
        api_version: str = "2024-02-01",
    ):
        """
        Initialize Azure OpenAI client.
        
        Args:
            api_key: API key (defaults to AZURE_OPENAI_API_KEY env var)
            endpoint: Azure endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            deployment: Deployment name
            api_version: API version
        """
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")

        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY environment variable."
            )
        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT environment variable."
            )

        self.deployment = deployment
        self.api_version = api_version

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using Azure OpenAI API."""
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

        client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

        response = client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model=self.deployment,
            provider="azure_openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    @property
    def provider_name(self) -> str:
        return "azure_openai"


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""

    def __init__(self, responses: Optional[list[str]] = None):
        """
        Initialize mock client.
        
        Args:
            responses: List of responses to return in order
        """
        self.responses = responses or ["⟨⟨1⟩⟩ G ¬error"]
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Return a mock response."""
        response = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return LLMResponse(
            text=response,
            model="mock",
            provider="mock",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

    @property
    def provider_name(self) -> str:
        return "mock"


def get_llm_client(
    provider: str = "openai",
    **kwargs,
) -> LLMClient:
    """
    Factory function to get an LLM client.
    
    Args:
        provider: Provider name ("openai", "azure", "mock")
        **kwargs: Provider-specific arguments
        
    Returns:
        An LLMClient instance
    """
    providers = {
        "openai": OpenAIClient,
        "azure": AzureOpenAIClient,
        "azure_openai": AzureOpenAIClient,
        "mock": MockLLMClient,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")

    return providers[provider](**kwargs)


# =============================================================================
# Core Translation Functions
# =============================================================================


def extract_atl_from_response(response: str) -> list[str]:
    """
    Extract ATL formula candidates from an LLM response.
    
    The LLM might return extra text; this function extracts
    plausible ATL formulas.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        List of candidate ATL formula strings
    """
    candidates = []

    # Clean up the response
    text = response.strip()

    # Try to extract from code blocks
    code_block_pattern = r"```(?:atl)?\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        candidates.extend([m.strip() for m in matches if m.strip()])

    # Look for lines starting with coalition brackets
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("⟨⟨") or line.startswith("<<"):
            candidates.append(line)
        elif line and not line.startswith(("#", "//", "NL:", "ATL:")):
            # Could be a simple formula
            if any(op in line for op in ["⟨⟨", "<<", "G ", "F ", "X ", " U "]):
                candidates.append(line)

    # If no candidates found, try the whole response
    if not candidates and text:
        # Remove common prefixes
        for prefix in ["ATL:", "Formula:", "Answer:"]:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        candidates.append(text)

    return candidates


def translate_nl_to_atl(
    nl_text: str,
    config: Optional[dict] = None,
    client: Optional[LLMClient] = None,
    validate: bool = True,
) -> list[str]:
    """
    Translate a natural language requirement to ATL formula(s).
    
    Builds the prompt, calls the LLM, extracts candidate ATL formulas,
    and optionally validates them syntactically.
    
    Args:
        nl_text: The natural language requirement
        config: Optional configuration dict with:
            - provider: LLM provider name
            - few_shot_examples: Custom examples
            - max_tokens, temperature: LLM parameters
        client: Optional pre-configured LLM client
        validate: Whether to validate extracted formulas
        
    Returns:
        List of valid ATL formula strings (may be empty if all invalid)
    """
    config = config or {}

    # Get or create client
    if client is None:
        provider = config.get("provider", "openai")
        client = get_llm_client(provider)

    # Build prompt
    few_shot = config.get("few_shot_examples")
    prompt = build_translation_prompt(nl_text, few_shot)

    # Call LLM
    max_tokens = config.get("max_tokens", 256)
    temperature = config.get("temperature", 0.0)
    response = client.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    # Extract candidates
    candidates = extract_atl_from_response(response.text)

    if not validate:
        return candidates

    # Validate and filter
    fragment_config = load_fragment_config()
    constraints = fragment_config.get("constraints", {})
    max_depth = constraints.get("max_nesting_depth", 4)
    max_coalition = constraints.get("max_coalition_size", 5)

    valid_formulas = []
    for candidate in candidates:
        is_ok, errors = validate_atl_string(candidate, max_depth, max_coalition)
        if is_ok:
            valid_formulas.append(candidate)

    return valid_formulas


def critique_nl_atl_pair(
    nl_text: str,
    atl_text: str,
    client: Optional[LLMClient] = None,
    provider: str = "openai",
) -> dict:
    """
    Ask the LLM to critique an NL-ATL pair.
    
    Checks whether the ATL formula correctly captures the NL requirement.
    
    Args:
        nl_text: The natural language requirement
        atl_text: The proposed ATL formula
        client: Optional pre-configured LLM client
        provider: LLM provider to use if client not provided
        
    Returns:
        Dictionary with:
        - ok: bool - whether the pair is correct
        - issues: list[str] - any issues found
        - explanation: str - analysis explanation
        - suggested_fix: str|None - corrected formula if needed
    """
    if client is None:
        client = get_llm_client(provider)

    prompt = build_critique_prompt(nl_text, atl_text)
    response = client.generate(prompt, max_tokens=512, temperature=0.0)

    # Parse JSON response
    try:
        # Extract JSON from response
        text = response.text.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if text:
                text = text.group(1)
        elif "```" in text:
            text = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if text:
                text = text.group(1)

        result = json.loads(text)
        return {
            "ok": result.get("ok", False),
            "issues": result.get("issues", []),
            "explanation": result.get("explanation", ""),
            "suggested_fix": result.get("suggested_fix"),
        }
    except (json.JSONDecodeError, AttributeError):
        # If parsing fails, return a default response
        return {
            "ok": False,
            "issues": ["Failed to parse LLM response"],
            "explanation": response.text,
            "suggested_fix": None,
        }


def paraphrase_nl(
    nl_text: str,
    num_paraphrases: int = 3,
    client: Optional[LLMClient] = None,
    provider: str = "openai",
    preserve_entities: bool = True,
) -> list[str]:
    """
    Generate paraphrases of a natural language requirement.
    
    Args:
        nl_text: The original NL text
        num_paraphrases: Number of variations to generate
        client: Optional pre-configured LLM client
        provider: LLM provider to use
        preserve_entities: Whether to preserve agent/proposition names
        
    Returns:
        List of paraphrased sentences
    """
    if client is None:
        client = get_llm_client(provider)

    constraint = ""
    if preserve_entities:
        constraint = (
            "IMPORTANT: Keep all agent names, coalition references, and proposition "
            "names exactly as they appear in the original. Only rephrase the "
            "surrounding language."
        )

    prompt = f"""Generate {num_paraphrases} different paraphrases of the following requirement.
Each paraphrase should preserve the exact meaning, agents, and temporal relationships.

{constraint}

Original: {nl_text}

Provide exactly {num_paraphrases} paraphrases, one per line, numbered 1-{num_paraphrases}.
Do not include the original sentence in your response."""

    response = client.generate(prompt, max_tokens=512, temperature=0.7)

    # Parse numbered list
    paraphrases = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        # Remove numbering like "1.", "1)", etc.
        line = re.sub(r"^\d+[\.\)\:]?\s*", "", line)
        if line and line != nl_text:
            paraphrases.append(line)

    return paraphrases[:num_paraphrases]


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate NL to ATL formula")
    parser.add_argument("nl_text", nargs="?", help="Natural language text to translate")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--critique", action="store_true", help="Also critique the result")

    args = parser.parse_args()

    if args.nl_text:
        config = {"provider": args.provider}
        results = translate_nl_to_atl(
            args.nl_text, config=config, validate=not args.no_validate
        )
        print("Translation results:")
        for r in results:
            print(f"  {r}")

        if args.critique and results:
            print("\nCritique:")
            critique = critique_nl_atl_pair(args.nl_text, results[0], provider=args.provider)
            print(f"  OK: {critique['ok']}")
            if critique["issues"]:
                print(f"  Issues: {critique['issues']}")
            if critique["suggested_fix"]:
                print(f"  Suggested fix: {critique['suggested_fix']}")
    else:
        parser.print_help()
