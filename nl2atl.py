"""
NL2ATL Translation Module
=========================

Provides LLM-based translation from Natural Language to ATL formulas.

This module implements:
1. Prompt building with configurable few-shot examples
2. Provider-agnostic LLM client abstraction (OpenAI, Azure, mock)
3. Core translation functions with validation
4. Cross-checking and critique functions

Architecture Overview
---------------------
```
NL Text → build_translation_prompt() → LLM → extract_atl_from_response() → validate → ATL
                     ↓
            critique_nl_atl_pair() → quality assessment
```

For AI Integration
------------------
Key functions for pipeline integration:
- `translate_nl_to_atl()`: Main translation entry point
- `critique_nl_atl_pair()`: Quality assessment of NL-ATL pairs
- `paraphrase_nl()`: Generate NL variations for data augmentation
- `get_llm_client()`: Factory for LLM client instances

Example Usage
-------------
>>> from nl2atl import translate_nl_to_atl, critique_nl_atl_pair
>>> 
>>> # Translate NL to ATL
>>> formulas = translate_nl_to_atl("Agents 1 and 2 can ensure the system never crashes")
>>> print(formulas)  # ['⟨⟨1,2⟩⟩ G ¬crash']
>>>
>>> # Critique a translation
>>> result = critique_nl_atl_pair("The robot can reach the goal", "⟨⟨robot⟩⟩ F goal")
>>> print(result["ok"])  # True

Environment Variables
---------------------
- OPENAI_API_KEY: Required for OpenAI provider
- AZURE_OPENAI_API_KEY: Required for Azure provider
- AZURE_OPENAI_ENDPOINT: Required for Azure provider
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from atl_syntax import is_valid, parse_atl, validate_atl_string


# =============================================================================
# Configuration Loading
# =============================================================================

CONFIG_DIR = Path(__file__).parent / "config"


def load_templates_config() -> dict:
    """
    Load templates and few-shot examples from config file.
    
    Returns:
        Dictionary with 'templates', 'few_shot_examples', 'atom_phrases', etc.
        Returns defaults if config file not found.
    """
    config_path = CONFIG_DIR / "templates_atl.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"templates": [], "few_shot_examples": []}


def load_fragment_config() -> dict:
    """
    Load ATL fragment configuration from YAML file.
    
    Returns:
        Dictionary with temporal operators, constraints, etc.
        Returns defaults if config file not found.
    """
    config_path = CONFIG_DIR / "atl_fragment.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            pass  # YAML not available, return defaults
    return {
        "temporal_operators": ["X", "F", "G", "U"],
        "boolean_connectives": ["∧", "∨", "¬", "→"],
        "constraints": {"max_nesting_depth": 4, "max_coalition_size": 5},
    }


# =============================================================================
# ATL Syntax Reference for Prompts
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


# =============================================================================
# Prompt Building
# =============================================================================


def build_translation_prompt(
    nl_text: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    include_syntax_description: bool = True,
    custom_instructions: Optional[str] = None,
) -> str:
    """
    Build a structured prompt for NL to ATL translation.
    
    This constructs a prompt that includes:
    1. System instruction defining the task
    2. ATL syntax reference (optional)
    3. Few-shot examples
    4. The actual NL text to translate
    
    Args:
        nl_text: The natural language requirement to translate
        few_shot_examples: List of {"nl": ..., "atl": ...} examples
        include_syntax_description: Whether to include ATL syntax reference
        custom_instructions: Additional instructions to append
        
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

    # Custom instructions
    if custom_instructions:
        parts.append(f"\n{custom_instructions}\n")

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
    
    The critique checks whether the ATL formula correctly captures
    the natural language requirement.
    
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


def build_paraphrase_prompt(
    nl_text: str,
    num_paraphrases: int = 3,
    preserve_entities: bool = True,
) -> str:
    """
    Build a prompt for generating NL paraphrases.
    
    Args:
        nl_text: The original NL text
        num_paraphrases: Number of variations to generate
        preserve_entities: Whether to preserve agent/proposition names
        
    Returns:
        A formatted prompt for paraphrasing
    """
    constraint = ""
    if preserve_entities:
        constraint = (
            "IMPORTANT: Keep all agent names, coalition references, and proposition "
            "names exactly as they appear in the original. Only rephrase the "
            "surrounding language."
        )

    return f"""Generate {num_paraphrases} different paraphrases of the following requirement.
Each paraphrase should preserve the exact meaning, agents, and temporal relationships.

{constraint}

Original: {nl_text}

Provide exactly {num_paraphrases} paraphrases, one per line, numbered 1-{num_paraphrases}.
Do not include the original sentence in your response."""


# =============================================================================
# LLM Client Abstraction
# =============================================================================


@dataclass
class LLMResponse:
    """
    Response from an LLM call.
    
    Attributes:
        text: The generated text content
        model: Model identifier used
        provider: Provider name (openai, azure, mock)
        usage: Token usage statistics
        raw_response: Full response object (provider-specific)
    """
    text: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Optional[Any] = None


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Implement this interface to add support for new LLM providers.
    """

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
        """Return the provider name identifier."""
        pass


class OpenAIClient(LLMClient):
    """
    OpenAI API client implementation.
    
    Supports both OpenAI direct API and compatible endpoints.
    """

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
            model: Model to use (default: gpt-4o-mini)
            base_url: Optional custom base URL for compatible APIs
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
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
    """
    Azure OpenAI API client implementation.
    """

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
    """
    Mock LLM client for testing without API calls.
    
    Returns predefined responses based on prompt type, or cycles through
    custom responses if provided.
    """
    
    # Default responses for different prompt types
    CRITIQUE_RESPONSE = json.dumps({
        "ok": True,
        "issues": [],
        "explanation": "The ATL formula correctly captures the natural language requirement.",
        "suggested_fix": None
    })
    
    TRANSLATION_RESPONSE = "⟨⟨1⟩⟩ G ¬error"
    
    PARAPHRASE_RESPONSE = """1. The system must always avoid errors.
2. Errors should never occur in the system.
3. The system guarantees no errors will happen."""

    def __init__(self, responses: Optional[List[str]] = None):
        """
        Initialize mock client.
        
        Args:
            responses: List of responses to return in order (overrides auto-detection)
        """
        self.responses = responses
        self._call_count = 0
        self._call_history: List[str] = []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Return a mock response based on prompt type."""
        self._call_history.append(prompt)
        
        # If custom responses provided, use them
        if self.responses:
            response = self.responses[self._call_count % len(self.responses)]
        # Otherwise, auto-detect prompt type
        elif '"ok"' in prompt or "critique" in prompt.lower() or "check if" in prompt.lower():
            response = self.CRITIQUE_RESPONSE
        elif "paraphrase" in prompt.lower() or "rephrase" in prompt.lower():
            response = self.PARAPHRASE_RESPONSE
        else:
            # Default: translation response
            response = self.TRANSLATION_RESPONSE
        
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

    @property
    def call_history(self) -> List[str]:
        """Get all prompts sent to this mock client."""
        return self._call_history


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
        
    Raises:
        ValueError: If provider is unknown
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
# Response Extraction
# =============================================================================


def extract_atl_from_response(response: str) -> List[str]:
    """
    Extract ATL formula candidates from an LLM response.
    
    The LLM might return extra text; this function extracts
    plausible ATL formulas using various heuristics.
    
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
        # Coalition patterns (Unicode or ASCII)
        if line.startswith("⟨⟨") or line.startswith("<<"):
            candidates.append(line)
        # Check for temporal operators or negation with coalition
        elif line and not line.startswith(("#", "//", "NL:", "ATL:", "-", "*")):
            # Look for ATL-like content
            if any(marker in line for marker in ["⟨⟨", "<<", "G ", "F ", "X ", " U "]):
                candidates.append(line)

    # If no candidates found, try the whole response
    if not candidates and text:
        # Remove common prefixes
        for prefix in ["ATL:", "Formula:", "Answer:", "Output:", "Result:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        candidates.append(text)

    # Clean up candidates
    cleaned = []
    for c in candidates:
        # Remove trailing punctuation
        c = c.rstrip(".,;:")
        if c:
            cleaned.append(c)

    return cleaned


# =============================================================================
# Core Translation Functions
# =============================================================================


def translate_nl_to_atl(
    nl_text: str,
    config: Optional[Dict[str, Any]] = None,
    client: Optional[LLMClient] = None,
    validate: bool = True,
) -> List[str]:
    """
    Translate a natural language requirement to ATL formula(s).
    
    This is the main entry point for NL→ATL translation. It:
    1. Builds a prompt with few-shot examples
    2. Calls the LLM
    3. Extracts candidate ATL formulas
    4. Optionally validates them syntactically
    
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
        
    Example:
        >>> formulas = translate_nl_to_atl("Agents 1 and 2 can ensure safety")
        >>> print(formulas)
        ['⟨⟨1,2⟩⟩ G safe']
    """
    config = config or {}

    # Get or create client
    if client is None:
        provider = config.get("provider", "openai")
        client = get_llm_client(provider)

    # Build prompt
    few_shot = config.get("few_shot_examples")
    custom_instructions = config.get("custom_instructions")
    prompt = build_translation_prompt(
        nl_text, 
        few_shot,
        custom_instructions=custom_instructions
    )

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
) -> Dict[str, Any]:
    """
    Ask the LLM to critique an NL-ATL pair.
    
    This function checks whether the ATL formula correctly captures
    the natural language requirement.
    
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
        text = response.text.strip()
        
        # Handle markdown code blocks
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

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
            "issues": ["Failed to parse LLM critique response"],
            "explanation": response.text,
            "suggested_fix": None,
        }


def paraphrase_nl(
    nl_text: str,
    num_paraphrases: int = 3,
    client: Optional[LLMClient] = None,
    provider: str = "openai",
    preserve_entities: bool = True,
) -> List[str]:
    """
    Generate paraphrases of a natural language requirement.
    
    Useful for data augmentation in dataset generation.
    
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

    prompt = build_paraphrase_prompt(nl_text, num_paraphrases, preserve_entities)
    response = client.generate(prompt, max_tokens=512, temperature=0.7)

    # Parse numbered list
    paraphrases = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        # Remove numbering like "1.", "1)", "1:", etc.
        line = re.sub(r"^\d+[\.\)\:]?\s*", "", line)
        if line and line != nl_text:
            paraphrases.append(line)

    return paraphrases[:num_paraphrases]


# =============================================================================
# Batch Processing
# =============================================================================


def translate_batch(
    nl_texts: List[str],
    config: Optional[Dict[str, Any]] = None,
    client: Optional[LLMClient] = None,
    validate: bool = True,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Translate multiple NL texts to ATL.
    
    Args:
        nl_texts: List of NL requirements
        config: Translation configuration
        client: LLM client
        validate: Whether to validate results
        progress_callback: Optional callback(i, total) for progress
        
    Returns:
        List of results, each with:
        - nl_text: original text
        - formulas: list of ATL formulas
        - success: whether translation succeeded
    """
    config = config or {}
    
    if client is None:
        provider = config.get("provider", "openai")
        client = get_llm_client(provider)
    
    results = []
    total = len(nl_texts)
    
    for i, nl_text in enumerate(nl_texts):
        try:
            formulas = translate_nl_to_atl(
                nl_text, 
                config=config, 
                client=client, 
                validate=validate
            )
            results.append({
                "nl_text": nl_text,
                "formulas": formulas,
                "success": len(formulas) > 0,
            })
        except Exception as e:
            results.append({
                "nl_text": nl_text,
                "formulas": [],
                "success": False,
                "error": str(e),
            })
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return results


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate Natural Language to ATL formulas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nl2atl.py "The robot can eventually reach the goal"
  python nl2atl.py "Agents 1 and 2 can ensure safety" --critique
  python nl2atl.py "..." --provider mock
        """
    )
    
    parser.add_argument(
        "nl_text", 
        nargs="?", 
        help="Natural language text to translate"
    )
    parser.add_argument(
        "--provider", 
        default="openai", 
        choices=["openai", "azure", "mock"],
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--no-validate", 
        action="store_true", 
        help="Skip ATL validation"
    )
    parser.add_argument(
        "--critique", 
        action="store_true", 
        help="Also critique the result"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    if args.nl_text:
        config = {"provider": args.provider}
        
        try:
            results = translate_nl_to_atl(
                args.nl_text, 
                config=config, 
                validate=not args.no_validate
            )
            
            output = {
                "nl_text": args.nl_text,
                "formulas": results,
            }
            
            if args.critique and results:
                critique = critique_nl_atl_pair(
                    args.nl_text, 
                    results[0], 
                    provider=args.provider
                )
                output["critique"] = critique
            
            if args.json:
                print(json.dumps(output, indent=2))
            else:
                print("Translation results:")
                for r in results:
                    print(f"  {r}")
                
                if args.critique and results:
                    print("\nCritique:")
                    print(f"  OK: {critique['ok']}")
                    if critique["issues"]:
                        print(f"  Issues: {critique['issues']}")
                    if critique["suggested_fix"]:
                        print(f"  Suggested fix: {critique['suggested_fix']}")
                        
        except ValueError as e:
            print(f"Error: {e}")
            print("Make sure to set the appropriate API key environment variable.")
    else:
        parser.print_help()
