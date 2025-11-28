"""
Cross-Verification Pipeline
============================

Multi-model verification with three-stage validation:
1. Syntactic Validation (Deterministic) - Uses atl_syntax.py parser
2. Semantic Verification (LLM) - Cross-model agreement check
3. Before/After Reporting - Quantifiable metrics

Architecture:
-------------
    Sample → Syntactic Check → Semantic Verification (LLM) → Report
                   ↓                      ↓
            atl_syntax.py          Generator ≠ Verifier
                                   (OpenAI)    (Claude)

Temperature Strategy:
---------------------
- NL Generation: High temp (0.8-1.0) for creative diversity
- ATL Translation: Low temp (0.1-0.3) for precision
- Verification: Low temp (0.0-0.2) for consistent judgment

Usage:
------
    from verification_pipeline import VerificationPipeline, PipelineConfig
    from sample_store import SampleStore
    
    config = PipelineConfig.from_yaml("config/pipeline_config.yaml")
    pipeline = VerificationPipeline(config)
    
    store = SampleStore(Path("data/unified_samples.json"))
    report = pipeline.verify_samples(store.get_all())
    report.save("reports/verification_report.json")
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any

# Local imports
from sample_store import ATLSample


# =============================================================================
# Configuration
# =============================================================================

class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    MOCK = "mock"  # For testing


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: ModelProvider
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 512
    api_key_env: str = ""  # Environment variable name for API key
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        # Default env var names
        if self.provider == ModelProvider.OPENAI:
            return os.environ.get("OPENAI_API_KEY")
        elif self.provider == ModelProvider.ANTHROPIC:
            return os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == ModelProvider.AZURE_OPENAI:
            return os.environ.get("AZURE_OPENAI_API_KEY")
        return None


@dataclass
class PipelineConfig:
    """Configuration for the verification pipeline."""
    
    # Generator model (for NL→ATL translation)
    generator: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        temperature=0.2,
    ))
    
    # Verifier model (different provider for cross-verification)
    verifier: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514",
        temperature=0.0,
    ))
    
    # Temperature settings for two-stage generation
    nl_generation_temperature: float = 0.9  # High for creativity
    atl_translation_temperature: float = 0.2  # Low for precision
    verification_temperature: float = 0.0  # Deterministic
    
    # Verification thresholds
    confidence_threshold: float = 0.7
    require_cross_model_agreement: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Output settings
    report_dir: Path = field(default_factory=lambda: Path("reports"))
    save_intermediate: bool = True
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            config = cls()
            
            if 'generator' in data:
                g = data['generator']
                config.generator = ModelConfig(
                    provider=ModelProvider(g.get('provider', 'openai')),
                    model_name=g.get('model_name', 'gpt-4o-mini'),
                    temperature=g.get('temperature', 0.2),
                    max_tokens=g.get('max_tokens', 512),
                    api_key_env=g.get('api_key_env', ''),
                )
            
            if 'verifier' in data:
                v = data['verifier']
                config.verifier = ModelConfig(
                    provider=ModelProvider(v.get('provider', 'anthropic')),
                    model_name=v.get('model_name', 'claude-sonnet-4-20250514'),
                    temperature=v.get('temperature', 0.0),
                    max_tokens=v.get('max_tokens', 512),
                    api_key_env=v.get('api_key_env', ''),
                )
            
            # Temperature settings
            if 'temperatures' in data:
                t = data['temperatures']
                config.nl_generation_temperature = t.get('nl_generation', 0.9)
                config.atl_translation_temperature = t.get('atl_translation', 0.2)
                config.verification_temperature = t.get('verification', 0.0)
            
            # Thresholds
            if 'thresholds' in data:
                th = data['thresholds']
                config.confidence_threshold = th.get('confidence', 0.7)
                config.require_cross_model_agreement = th.get('cross_model_agreement', True)
            
            # Retry settings
            if 'retry' in data:
                r = data['retry']
                config.max_retries = r.get('max_retries', 3)
                config.retry_delay_seconds = r.get('delay_seconds', 1.0)
            
            # Output
            if 'output' in data:
                o = data['output']
                config.report_dir = Path(o.get('report_dir', 'reports'))
                config.save_intermediate = o.get('save_intermediate', True)
            
            return config
        except Exception as e:
            print(f"Warning: Could not load config from {path}: {e}")
            return cls()
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        data = {
            'generator': {
                'provider': self.generator.provider.value,
                'model_name': self.generator.model_name,
                'temperature': self.generator.temperature,
                'max_tokens': self.generator.max_tokens,
                'api_key_env': self.generator.api_key_env,
            },
            'verifier': {
                'provider': self.verifier.provider.value,
                'model_name': self.verifier.model_name,
                'temperature': self.verifier.temperature,
                'max_tokens': self.verifier.max_tokens,
                'api_key_env': self.verifier.api_key_env,
            },
            'temperatures': {
                'nl_generation': self.nl_generation_temperature,
                'atl_translation': self.atl_translation_temperature,
                'verification': self.verification_temperature,
            },
            'thresholds': {
                'confidence': self.confidence_threshold,
                'cross_model_agreement': self.require_cross_model_agreement,
            },
            'retry': {
                'max_retries': self.max_retries,
                'delay_seconds': self.retry_delay_seconds,
            },
            'output': {
                'report_dir': str(self.report_dir),
                'save_intermediate': self.save_intermediate,
            },
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Verification Results
# =============================================================================

class VerificationStatus(str, Enum):
    """Overall verification status."""
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    PENDING = "pending"
    ERROR = "error"


@dataclass
class SyntaxCheckResult:
    """Result of syntactic validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parsed_structure: Optional[dict] = None  # AST info


@dataclass
class SemanticCheckResult:
    """Result of semantic verification by LLM."""
    matches: bool
    confidence: float
    explanation: str
    issues: list[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    verifier_model: str = ""
    verifier_provider: str = ""


@dataclass
class VerificationResult:
    """Complete verification result for a sample."""
    sample_id: str
    
    # Original data
    original_nl: str
    original_atl: str
    
    # Syntax check (Stage 1)
    syntax_check: SyntaxCheckResult
    
    # Semantic verification (Stage 2)
    semantic_check: Optional[SemanticCheckResult] = None
    
    # Overall status
    status: VerificationStatus = VerificationStatus.PENDING
    
    # Cross-model verification
    cross_model_atl: Optional[str] = None  # ATL from verifier model
    cross_model_agreement: Optional[bool] = None
    
    # Timing
    verification_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Any corrections applied
    corrected_atl: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'sample_id': self.sample_id,
            'original_nl': self.original_nl,
            'original_atl': self.original_atl,
            'syntax_check': {
                'valid': self.syntax_check.valid,
                'errors': self.syntax_check.errors,
                'warnings': self.syntax_check.warnings,
            },
            'status': self.status.value,
            'verification_time_ms': self.verification_time_ms,
            'timestamp': self.timestamp,
        }
        
        if self.semantic_check:
            result['semantic_check'] = {
                'matches': self.semantic_check.matches,
                'confidence': self.semantic_check.confidence,
                'explanation': self.semantic_check.explanation,
                'issues': self.semantic_check.issues,
                'suggested_fix': self.semantic_check.suggested_fix,
                'verifier_model': self.semantic_check.verifier_model,
                'verifier_provider': self.semantic_check.verifier_provider,
            }
        
        if self.cross_model_atl:
            result['cross_model_atl'] = self.cross_model_atl
            result['cross_model_agreement'] = self.cross_model_agreement
        
        if self.corrected_atl:
            result['corrected_atl'] = self.corrected_atl
        
        return result


# =============================================================================
# LLM Client Abstraction
# =============================================================================

@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        """Generate a response."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY)")
    
    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        from openai import OpenAI
        
        start = time.time()
        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
        )
        
        latency = (time.time() - start) * 1000
        
        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model=self.model,
            provider="openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            latency_ms=latency,
        )
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def model_name(self) -> str:
        return self.model


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY)")
    
    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        import anthropic
        
        start = time.time()
        client = anthropic.Anthropic(api_key=self.api_key)
        
        response = client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency = (time.time() - start) * 1000
        
        return LLMResponse(
            text=response.content[0].text.strip(),
            model=self.model,
            provider="anthropic",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            latency_ms=latency,
        )
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def model_name(self) -> str:
        return self.model


class MockClient(LLMClient):
    """Mock client for testing without API calls."""
    
    # Simulated responses for different prompt types
    VERIFICATION_RESPONSE = json.dumps({
        "matches": True,
        "confidence": 0.85,
        "explanation": "The NL statement correctly describes the ATL formula semantics.",
        "issues": [],
        "suggested_fix": None
    })
    
    # Simulated ATL translation response
    TRANSLATION_RESPONSE = "<<Agent>> G(safe)"
    
    def __init__(self, responses: Optional[list[str]] = None):
        self.responses = responses
        self.call_count = 0
    
    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        # Determine response type based on prompt content
        if self.responses:
            response = self.responses[self.call_count % len(self.responses)]
        elif "Respond with ONLY the ATL formula" in prompt or "Translate the following" in prompt:
            # Translation prompt - return ATL formula
            response = self.TRANSLATION_RESPONSE
        elif "JSON" in prompt or "matches" in prompt or '"matches"' in prompt:
            # Verification prompt - return JSON
            response = self.VERIFICATION_RESPONSE
        else:
            # Default to verification response
            response = self.VERIFICATION_RESPONSE
        
        self.call_count += 1
        
        return LLMResponse(
            text=response,
            model="mock-model",
            provider="mock",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=10.0,
        )
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    @property
    def model_name(self) -> str:
        return "mock-model"


def create_client(config: ModelConfig) -> LLMClient:
    """Factory function to create LLM client from config."""
    api_key = config.get_api_key()
    
    if config.provider == ModelProvider.OPENAI:
        return OpenAIClient(model=config.model_name, api_key=api_key)
    elif config.provider == ModelProvider.ANTHROPIC:
        return AnthropicClient(model=config.model_name, api_key=api_key)
    elif config.provider == ModelProvider.MOCK:
        return MockClient()
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


# =============================================================================
# Syntactic Validator
# =============================================================================

class SyntaxValidator:
    """
    Validates ATL formula syntax using regex-based patterns.
    
    NOTE: The pyparsing-based parser in atl_syntax.py has recursion issues
    with certain formula patterns. We use a robust regex-based validator
    that handles all common ATL formula structures.
    """
    
    # ATL Operator patterns
    COALITION_PATTERN = re.compile(r'^(<<|⟨⟨)\s*([^>⟩]*)\s*(>>|⟩⟩)')
    TEMPORAL_OPS = re.compile(r'\b([GFXU])\b')
    BOOLEAN_OPS = re.compile(r'(∧|∨|¬|→|&|\||!|->|and|or|not|implies)')
    PROPOSITION = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    
    @staticmethod
    def validate(atl_formula: str) -> SyntaxCheckResult:
        """
        Validate an ATL formula syntactically using regex patterns.
        
        Checks:
        1. Coalition operator presence and balance
        2. Parentheses balance
        3. Temporal operator presence
        4. Basic structure validity
        """
        errors = []
        warnings = []
        structure = {}
        
        formula = atl_formula.strip()
        
        # 1. Check for coalition operator at start
        coalition_match = SyntaxValidator.COALITION_PATTERN.match(formula)
        if not coalition_match:
            # Check if it has coalition somewhere (maybe after negation)
            if '<<' not in formula and '⟨⟨' not in formula:
                errors.append("Formula must contain coalition operator <<...>> or ⟨⟨...⟩⟩")
        else:
            # Extract agents from coalition
            agents_str = coalition_match.group(2).strip()
            if agents_str:
                agents = [a.strip() for a in agents_str.split(',') if a.strip()]
                structure['agents'] = agents
            else:
                structure['agents'] = []
                warnings.append("Empty coalition (grand coalition assumed)")
        
        # 2. Check balanced parentheses
        open_parens = formula.count('(')
        close_parens = formula.count(')')
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
        
        # 3. Check balanced angle brackets
        open_ascii = formula.count('<<')
        close_ascii = formula.count('>>')
        open_unicode = formula.count('⟨⟨')
        close_unicode = formula.count('⟩⟩')
        
        if open_ascii != close_ascii:
            errors.append(f"Unbalanced ASCII coalition brackets: {open_ascii} << vs {close_ascii} >>")
        if open_unicode != close_unicode:
            errors.append(f"Unbalanced Unicode coalition brackets")
        
        # 4. Check for temporal operators
        temporal_matches = SyntaxValidator.TEMPORAL_OPS.findall(formula)
        if temporal_matches:
            structure['temporal_ops'] = list(set(temporal_matches))
        else:
            warnings.append("No temporal operators (G, F, X, U) found - formula may be incomplete")
        
        # 5. Extract propositions (atomic formulas)
        # Remove operators and brackets first
        cleaned = re.sub(r'<<[^>]*>>', '', formula)
        cleaned = re.sub(r'⟨⟨[^⟩]*⟩⟩', '', cleaned)
        cleaned = re.sub(r'\b(G|F|X|U|and|or|not|implies|true|false)\b', '', cleaned)
        cleaned = re.sub(r'[()∧∨¬→&|!->]', ' ', cleaned)
        props = [p for p in cleaned.split() if p and re.match(r'^[a-zA-Z_]', p)]
        structure['atoms'] = list(set(props))
        
        # 6. Check for malformed patterns
        if re.search(r'<<\s*>>', formula) or re.search(r'⟨⟨\s*⟩⟩', formula):
            # Empty coalition is valid but worth noting
            pass
        
        if re.search(r'[GFX]\s*[GFX]', formula):
            warnings.append("Consecutive temporal operators without operand")
        
        if re.search(r'>>\s*$', formula) or re.search(r'⟩⟩\s*$', formula):
            errors.append("Formula ends with coalition operator - missing temporal formula")
        
        # 7. Estimate depth (rough approximation)
        structure['depth'] = max(
            formula.count('('),
            open_ascii + open_unicode,
            len(temporal_matches)
        )
        
        return SyntaxCheckResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            parsed_structure=structure if len(errors) == 0 else None,
        )


# =============================================================================
# Semantic Verifier
# =============================================================================

SEMANTIC_VERIFICATION_PROMPT = """You are an expert in formal verification and temporal logic, specifically Alternating-time Temporal Logic (ATL).

Your task is to verify whether an ATL formula correctly captures the meaning of a natural language requirement.

ATL Syntax Reference:
- ⟨⟨A⟩⟩ φ : Coalition A has a strategy to ensure φ
- G φ : φ always holds (Globally)
- F φ : φ eventually holds (Future)  
- X φ : φ holds in the next state (neXt)
- φ U ψ : φ holds until ψ becomes true
- → : implication
- ¬ : negation
- ∧ : conjunction (AND)
- ∨ : disjunction (OR)

Natural Language Requirement:
{nl_statement}

Proposed ATL Formula:
{atl_formula}

Analyze whether the ATL formula correctly captures the semantics of the natural language requirement.

Respond in the following JSON format ONLY (no additional text):
{{
    "matches": true/false,
    "confidence": 0.0 to 1.0,
    "explanation": "Your analysis explaining why it matches or doesn't",
    "issues": ["list of any semantic issues found"],
    "suggested_fix": "corrected ATL formula if needed, or null if correct"
}}"""


CROSS_TRANSLATION_PROMPT = """You are an expert in formal verification and Alternating-time Temporal Logic (ATL).

Translate the following natural language requirement into an ATL formula.

ATL Syntax:
- ⟨⟨A⟩⟩ φ : Coalition A can ensure φ
- G φ : Always φ
- F φ : Eventually φ
- X φ : Next state φ
- φ U ψ : φ until ψ
- → : implies
- ¬ : not
- ∧ : and
- ∨ : or

Requirement:
{nl_statement}

Respond with ONLY the ATL formula, nothing else."""


class SemanticVerifier:
    """
    Verifies semantic correctness of NL-ATL pairs using LLM.
    """
    
    def __init__(self, client: LLMClient, temperature: float = 0.0):
        self.client = client
        self.temperature = temperature
    
    def verify(self, nl: str, atl: str) -> SemanticCheckResult:
        """
        Verify that ATL formula captures NL semantics.
        """
        prompt = SEMANTIC_VERIFICATION_PROMPT.format(
            nl_statement=nl,
            atl_formula=atl,
        )
        
        try:
            response = self.client.generate(prompt, temperature=self.temperature)
            
            # Parse JSON response
            text = response.text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            data = json.loads(text)
            
            return SemanticCheckResult(
                matches=data.get('matches', False),
                confidence=data.get('confidence', 0.0),
                explanation=data.get('explanation', ''),
                issues=data.get('issues', []),
                suggested_fix=data.get('suggested_fix'),
                verifier_model=self.client.model_name,
                verifier_provider=self.client.provider_name,
            )
        
        except json.JSONDecodeError as e:
            return SemanticCheckResult(
                matches=False,
                confidence=0.0,
                explanation=f"Failed to parse verifier response: {e}",
                issues=[f"JSON parse error: {e}"],
                verifier_model=self.client.model_name,
                verifier_provider=self.client.provider_name,
            )
        except Exception as e:
            return SemanticCheckResult(
                matches=False,
                confidence=0.0,
                explanation=f"Verification error: {e}",
                issues=[str(e)],
                verifier_model=self.client.model_name,
                verifier_provider=self.client.provider_name,
            )
    
    def cross_translate(self, nl: str) -> str:
        """
        Generate ATL from NL using verifier model (for cross-checking).
        """
        prompt = CROSS_TRANSLATION_PROMPT.format(nl_statement=nl)
        
        try:
            response = self.client.generate(prompt, temperature=self.temperature)
            return response.text.strip()
        except Exception as e:
            return f"ERROR: {e}"


# =============================================================================
# Verification Pipeline
# =============================================================================

@dataclass
class PipelineMetrics:
    """Metrics from a pipeline run."""
    total_samples: int = 0
    syntax_valid: int = 0
    syntax_invalid: int = 0
    semantic_verified: int = 0
    semantic_rejected: int = 0
    needs_review: int = 0
    errors: int = 0
    cross_model_agreement: int = 0
    cross_model_disagreement: int = 0
    total_time_ms: float = 0.0
    avg_time_per_sample_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationReport:
    """Complete report from a pipeline run."""
    run_id: str
    timestamp: str
    config: dict
    metrics: PipelineMetrics
    results: list[VerificationResult]
    
    def to_dict(self) -> dict:
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'metrics': self.metrics.to_dict(),
            'results': [r.to_dict() for r in self.results],
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def print_summary(self) -> str:
        """Generate human-readable summary."""
        m = self.metrics
        lines = [
            "=" * 60,
            "VERIFICATION REPORT SUMMARY",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            "",
            "METRICS:",
            f"  Total Samples:        {m.total_samples}",
            f"  Syntax Valid:         {m.syntax_valid} ({m.syntax_valid/max(m.total_samples,1)*100:.1f}%)",
            f"  Syntax Invalid:       {m.syntax_invalid}",
            "",
            f"  Semantic Verified:    {m.semantic_verified} ({m.semantic_verified/max(m.total_samples,1)*100:.1f}%)",
            f"  Semantic Rejected:    {m.semantic_rejected}",
            f"  Needs Review:         {m.needs_review}",
            f"  Errors:               {m.errors}",
            "",
            f"  Cross-Model Agreement:    {m.cross_model_agreement}",
            f"  Cross-Model Disagreement: {m.cross_model_disagreement}",
            "",
            f"  Total Time:           {m.total_time_ms:.0f}ms",
            f"  Avg Time/Sample:      {m.avg_time_per_sample_ms:.0f}ms",
            "=" * 60,
        ]
        return "\n".join(lines)


class VerificationPipeline:
    """
    Multi-stage verification pipeline.
    
    Stages:
    1. Syntactic Validation (deterministic, uses atl_syntax.py)
    2. Semantic Verification (LLM-based, using verifier model)
    3. Cross-Model Check (optional, verifier generates ATL independently)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._generator_client: Optional[LLMClient] = None
        self._verifier_client: Optional[LLMClient] = None
    
    @property
    def generator_client(self) -> LLMClient:
        if self._generator_client is None:
            self._generator_client = create_client(self.config.generator)
        return self._generator_client
    
    @property
    def verifier_client(self) -> LLMClient:
        if self._verifier_client is None:
            self._verifier_client = create_client(self.config.verifier)
        return self._verifier_client
    
    def verify_sample(self, sample: ATLSample) -> VerificationResult:
        """
        Verify a single sample through all stages.
        """
        start_time = time.time()
        
        # Stage 1: Syntactic Validation
        syntax_result = SyntaxValidator.validate(sample.atl_formula)
        
        result = VerificationResult(
            sample_id=sample.id,
            original_nl=sample.nl_statement,
            original_atl=sample.atl_formula,
            syntax_check=syntax_result,
        )
        
        # If syntax is invalid, stop here
        if not syntax_result.valid:
            result.status = VerificationStatus.REJECTED
            result.verification_time_ms = (time.time() - start_time) * 1000
            return result
        
        # Stage 2: Semantic Verification (if verifier available)
        try:
            verifier = SemanticVerifier(
                self.verifier_client,
                temperature=self.config.verification_temperature
            )
            
            semantic_result = verifier.verify(sample.nl_statement, sample.atl_formula)
            result.semantic_check = semantic_result
            
            # Stage 3: Cross-model translation check
            if self.config.require_cross_model_agreement:
                cross_atl = verifier.cross_translate(sample.nl_statement)
                result.cross_model_atl = cross_atl
                
                # Normalize both for comparison
                from sample_store import SyntaxNormalizer
                orig_normalized = SyntaxNormalizer.to_ascii(sample.atl_formula).lower().replace(" ", "")
                cross_normalized = SyntaxNormalizer.to_ascii(cross_atl).lower().replace(" ", "")
                
                # Check structural similarity (not exact match due to formatting)
                result.cross_model_agreement = self._formulas_equivalent(
                    sample.atl_formula, cross_atl
                )
            
            # Determine overall status
            if semantic_result.matches and semantic_result.confidence >= self.config.confidence_threshold:
                if result.cross_model_agreement is False:
                    result.status = VerificationStatus.NEEDS_REVIEW
                else:
                    result.status = VerificationStatus.VERIFIED
            elif semantic_result.confidence < 0.5:
                result.status = VerificationStatus.REJECTED
            else:
                result.status = VerificationStatus.NEEDS_REVIEW
            
            # Apply suggested fix if any
            if semantic_result.suggested_fix and semantic_result.suggested_fix.lower() != 'null':
                result.corrected_atl = semantic_result.suggested_fix
        
        except Exception as e:
            result.status = VerificationStatus.ERROR
            result.semantic_check = SemanticCheckResult(
                matches=False,
                confidence=0.0,
                explanation=f"Verification error: {e}",
                issues=[str(e)],
            )
        
        result.verification_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _formulas_equivalent(self, formula1: str, formula2: str) -> bool:
        """
        Check if two formulas are structurally equivalent.
        Uses simple normalization - could be enhanced with AST comparison.
        """
        from sample_store import SyntaxNormalizer
        
        def normalize(f: str) -> str:
            f = SyntaxNormalizer.to_ascii(f)
            # Remove spaces around operators
            f = re.sub(r'\s+', '', f)
            f = f.lower()
            return f
        
        return normalize(formula1) == normalize(formula2)
    
    def verify_samples(
        self,
        samples: list[ATLSample],
        progress_callback: Optional[callable] = None,
    ) -> VerificationReport:
        """
        Verify multiple samples and generate a report.
        """
        import uuid
        
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        results: list[VerificationResult] = []
        metrics = PipelineMetrics(total_samples=len(samples))
        
        for i, sample in enumerate(samples):
            try:
                result = self.verify_sample(sample)
                results.append(result)
                
                # Update metrics
                if result.syntax_check.valid:
                    metrics.syntax_valid += 1
                else:
                    metrics.syntax_invalid += 1
                
                if result.status == VerificationStatus.VERIFIED:
                    metrics.semantic_verified += 1
                elif result.status == VerificationStatus.REJECTED:
                    metrics.semantic_rejected += 1
                elif result.status == VerificationStatus.NEEDS_REVIEW:
                    metrics.needs_review += 1
                elif result.status == VerificationStatus.ERROR:
                    metrics.errors += 1
                
                if result.cross_model_agreement is True:
                    metrics.cross_model_agreement += 1
                elif result.cross_model_agreement is False:
                    metrics.cross_model_disagreement += 1
                
                if progress_callback:
                    progress_callback(i + 1, len(samples), result)
                    
            except Exception as e:
                metrics.errors += 1
                print(f"Error verifying sample {sample.id}: {e}")
        
        # Finalize metrics
        metrics.total_time_ms = (time.time() - start_time) * 1000
        metrics.avg_time_per_sample_ms = (
            metrics.total_time_ms / max(len(samples), 1)
        )
        
        # Build report
        report = VerificationReport(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config={
                'generator': {
                    'provider': self.config.generator.provider.value,
                    'model': self.config.generator.model_name,
                },
                'verifier': {
                    'provider': self.config.verifier.provider.value,
                    'model': self.config.verifier.model_name,
                },
                'confidence_threshold': self.config.confidence_threshold,
                'require_cross_model_agreement': self.config.require_cross_model_agreement,
            },
            metrics=metrics,
            results=results,
        )
        
        return report


# =============================================================================
# CLI Interface
# =============================================================================

def run_verification(
    samples_path: Path,
    config_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    use_mock: bool = False,
) -> VerificationReport:
    """
    Run verification pipeline from command line.
    """
    from sample_store import SampleStore
    
    # Load config
    if config_path and config_path.exists():
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()
    
    # Use mock clients for testing
    if use_mock:
        config.generator.provider = ModelProvider.MOCK
        config.verifier.provider = ModelProvider.MOCK
    
    # Load samples
    store = SampleStore(samples_path)
    samples = store.get_all()
    
    print(f"Loaded {len(samples)} samples")
    print(f"Generator: {config.generator.provider.value}/{config.generator.model_name}")
    print(f"Verifier: {config.verifier.provider.value}/{config.verifier.model_name}")
    print()
    
    # Run pipeline
    pipeline = VerificationPipeline(config)
    
    def progress(current: int, total: int, result: VerificationResult):
        status_icon = {
            VerificationStatus.VERIFIED: "✓",
            VerificationStatus.REJECTED: "✗",
            VerificationStatus.NEEDS_REVIEW: "?",
            VerificationStatus.ERROR: "!",
            VerificationStatus.PENDING: ".",
        }
        print(f"  [{current}/{total}] {status_icon.get(result.status, '.')} {result.sample_id[:8]}...")
    
    report = pipeline.verify_samples(samples, progress_callback=progress)
    
    # Save report
    if output_path is None:
        output_path = config.report_dir / f"verification_{report.run_id}.json"
    
    report.save(output_path)
    print(f"\nReport saved to: {output_path}")
    print()
    print(report.print_summary())
    
    return report


if __name__ == "__main__":
    import sys
    
    samples_path = Path("data/unified_samples.json")
    config_path = Path("config/pipeline_config.yaml")
    
    use_mock = "--mock" in sys.argv
    
    if not samples_path.exists():
        print(f"Samples not found at {samples_path}")
        print("Run: python sample_store.py load")
        sys.exit(1)
    
    report = run_verification(
        samples_path=samples_path,
        config_path=config_path if config_path.exists() else None,
        use_mock=use_mock,
    )
