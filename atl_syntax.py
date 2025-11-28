"""
ATL Syntax Module
=================

This module provides data structures and utilities for representing, parsing,
validating, and normalizing Alternating-time Temporal Logic (ATL) formulas.

ATL Overview
------------
ATL extends CTL (Computation Tree Logic) with strategic modalities. The key
addition is the coalition operator ⟨⟨Γ⟩⟩ where Γ is a set of agents.
The formula ⟨⟨Γ⟩⟩φ means "coalition Γ has a strategy to ensure φ".

Supported Operators
-------------------
- Strategic modality: ⟨⟨Γ⟩⟩ (coalition strategy quantifier)
- Temporal operators: X (next), F (eventually), G (globally), U (until)
- Boolean connectives: ∧ (and), ∨ (or), ¬ (not), → (implies)

Syntax Formats
--------------
The parser supports both Unicode and ASCII representations:
- Coalition: ⟨⟨1,2⟩⟩ or <<1,2>>
- And: ∧ or & or "and"
- Or: ∨ or | or "or"  
- Not: ¬ or ! or "not"
- Implies: → or -> or "implies"

Example Usage
-------------
>>> from atl_syntax import parse_atl, is_valid, normalize_atl
>>> formula = parse_atl("<<1,2>> G safe")
>>> print(formula.pretty_print())
⟨⟨1,2⟩⟩ G safe
>>> is_valid(formula)
ValidationResult(valid=True, errors=[], warnings=[])

For AI Integration
------------------
This module is designed to be used in NL→ATL translation pipelines:
1. Use `parse_atl()` to convert LLM-generated strings to AST
2. Use `validate_atl_string()` to filter invalid outputs
3. Use `normalize_atl()` for consistent representation
4. Use `extract_components()` to analyze formula structure
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, List, Optional, Set, Tuple, Union

import pyparsing as pp
from pyparsing import (
    Forward,
    Group,
    Keyword,
    Literal,
    Optional as Opt,
    Suppress,
    Word,
    alphanums,
    alphas,
    delimitedList,
    infixNotation,
    opAssoc,
)


# =============================================================================
# Type Definitions
# =============================================================================

# Agent IDs can be integers (1, 2, 3) or strings (robot, controller)
AgentId = Union[int, str]


class TemporalOperator(Enum):
    """Temporal operators in ATL."""
    NEXT = "X"       # Next state
    EVENTUALLY = "F" # Eventually/Future  
    GLOBALLY = "G"   # Globally/Always
    UNTIL = "U"      # Until


class BooleanOperator(Enum):
    """Boolean operators."""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"


# =============================================================================
# AST Data Classes for ATL Formulas
# =============================================================================


class ATLFormula(ABC):
    """
    Abstract base class for all ATL formula nodes.
    
    All formula nodes implement:
    - pretty_print(): Unicode string representation
    - to_ascii(): ASCII-only string representation
    - depth(): Nesting depth of the formula
    - get_atoms(): Set of atomic propositions
    - get_agents(): Set of agent identifiers
    """

    @abstractmethod
    def pretty_print(self) -> str:
        """Return a human-readable Unicode string representation."""
        pass

    @abstractmethod
    def to_ascii(self) -> str:
        """Return an ASCII-only string representation."""
        pass

    @abstractmethod
    def depth(self) -> int:
        """Return the nesting depth of the formula (temporal/strategic operators)."""
        pass

    @abstractmethod
    def get_atoms(self) -> Set[str]:
        """Return all atomic propositions in this formula."""
        pass

    @abstractmethod
    def get_agents(self) -> Set[str]:
        """Return all agent identifiers in this formula."""
        pass

    def __str__(self) -> str:
        return self.pretty_print()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pretty_print()})"


@dataclass(frozen=True)
class Atom(ATLFormula):
    """
    Atomic proposition (e.g., 'p', 'crash', 'goal_reached').
    
    Atoms are the base building blocks of ATL formulas.
    They represent boolean state properties.
    """
    name: str

    def pretty_print(self) -> str:
        return self.name

    def to_ascii(self) -> str:
        return self.name

    def depth(self) -> int:
        return 0

    def get_atoms(self) -> Set[str]:
        return {self.name}

    def get_agents(self) -> Set[str]:
        return set()


@dataclass(frozen=True)
class BoolTrue(ATLFormula):
    """Boolean constant True (⊤)."""

    def pretty_print(self) -> str:
        return "⊤"

    def to_ascii(self) -> str:
        return "true"

    def depth(self) -> int:
        return 0

    def get_atoms(self) -> Set[str]:
        return set()

    def get_agents(self) -> Set[str]:
        return set()


@dataclass(frozen=True)
class BoolFalse(ATLFormula):
    """Boolean constant False (⊥)."""

    def pretty_print(self) -> str:
        return "⊥"

    def to_ascii(self) -> str:
        return "false"

    def depth(self) -> int:
        return 0

    def get_atoms(self) -> Set[str]:
        return set()

    def get_agents(self) -> Set[str]:
        return set()


@dataclass(frozen=True)
class Not(ATLFormula):
    """Negation: ¬φ"""
    operand: ATLFormula

    def pretty_print(self) -> str:
        inner = self.operand.pretty_print()
        # Add parentheses for binary operations
        if isinstance(self.operand, (And, Or, Implies)):
            inner = f"({inner})"
        return f"¬{inner}"

    def to_ascii(self) -> str:
        inner = self.operand.to_ascii()
        if isinstance(self.operand, (And, Or, Implies)):
            inner = f"({inner})"
        return f"!{inner}"

    def depth(self) -> int:
        return self.operand.depth()

    def get_atoms(self) -> Set[str]:
        return self.operand.get_atoms()

    def get_agents(self) -> Set[str]:
        return self.operand.get_agents()


@dataclass(frozen=True)
class And(ATLFormula):
    """Conjunction: φ ∧ ψ"""
    left: ATLFormula
    right: ATLFormula

    def pretty_print(self) -> str:
        return f"({self.left.pretty_print()} ∧ {self.right.pretty_print()})"

    def to_ascii(self) -> str:
        return f"({self.left.to_ascii()} & {self.right.to_ascii()})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth())

    def get_atoms(self) -> Set[str]:
        return self.left.get_atoms() | self.right.get_atoms()

    def get_agents(self) -> Set[str]:
        return self.left.get_agents() | self.right.get_agents()


@dataclass(frozen=True)
class Or(ATLFormula):
    """Disjunction: φ ∨ ψ"""
    left: ATLFormula
    right: ATLFormula

    def pretty_print(self) -> str:
        return f"({self.left.pretty_print()} ∨ {self.right.pretty_print()})"

    def to_ascii(self) -> str:
        return f"({self.left.to_ascii()} | {self.right.to_ascii()})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth())

    def get_atoms(self) -> Set[str]:
        return self.left.get_atoms() | self.right.get_atoms()

    def get_agents(self) -> Set[str]:
        return self.left.get_agents() | self.right.get_agents()


@dataclass(frozen=True)
class Implies(ATLFormula):
    """Implication: φ → ψ"""
    left: ATLFormula
    right: ATLFormula

    def pretty_print(self) -> str:
        return f"({self.left.pretty_print()} → {self.right.pretty_print()})"

    def to_ascii(self) -> str:
        return f"({self.left.to_ascii()} -> {self.right.to_ascii()})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth())

    def get_atoms(self) -> Set[str]:
        return self.left.get_atoms() | self.right.get_atoms()

    def get_agents(self) -> Set[str]:
        return self.left.get_agents() | self.right.get_agents()


@dataclass(frozen=True)
class TemporalOp(ATLFormula):
    """
    Standalone temporal operator: X φ, F φ, G φ
    
    Used when temporal operators appear outside of coalition context.
    In standard ATL, temporal operators should be scoped by a coalition,
    but we support standalone operators for flexibility.
    """
    operator: str  # One of: X, F, G
    operand: ATLFormula

    def pretty_print(self) -> str:
        return f"{self.operator} {self.operand.pretty_print()}"

    def to_ascii(self) -> str:
        return f"{self.operator} {self.operand.to_ascii()}"

    def depth(self) -> int:
        return self.operand.depth() + 1

    def get_atoms(self) -> Set[str]:
        return self.operand.get_atoms()

    def get_agents(self) -> Set[str]:
        return self.operand.get_agents()


@dataclass(frozen=True)
class Until(ATLFormula):
    """
    Until operator: φ U ψ
    
    φ holds until ψ becomes true.
    """
    left: ATLFormula
    right: ATLFormula

    def pretty_print(self) -> str:
        return f"({self.left.pretty_print()} U {self.right.pretty_print()})"

    def to_ascii(self) -> str:
        return f"({self.left.to_ascii()} U {self.right.to_ascii()})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth()) + 1

    def get_atoms(self) -> Set[str]:
        return self.left.get_atoms() | self.right.get_atoms()

    def get_agents(self) -> Set[str]:
        return self.left.get_agents() | self.right.get_agents()


@dataclass(frozen=True)
class Coalition(ATLFormula):
    """
    Strategic modality with coalition: ⟨⟨Γ⟩⟩ φ
    
    Represents that coalition Γ (a set of agents) has a strategy
    to ensure formula φ holds. The formula φ typically contains
    a temporal operator.
    
    Attributes:
        agents: Frozenset of agent identifiers
        temporal_op: Optional temporal operator (X, F, G, or None)
        operand: The formula under the coalition scope
        operand2: Second operand for Until operator
    """
    agents: FrozenSet[str]
    temporal_op: Optional[str]  # One of: X, F, G, U, or None
    operand: ATLFormula
    operand2: Optional[ATLFormula] = None  # For Until operator

    def __init__(
        self,
        agents: Union[Set[str], FrozenSet[str], List[str]],
        temporal_op: Optional[str],
        operand: ATLFormula,
        operand2: Optional[ATLFormula] = None
    ):
        # Convert agents to frozenset for immutability
        if isinstance(agents, (set, list)):
            agents = frozenset(str(a) for a in agents)
        object.__setattr__(self, 'agents', agents)
        object.__setattr__(self, 'temporal_op', temporal_op)
        object.__setattr__(self, 'operand', operand)
        object.__setattr__(self, 'operand2', operand2)

    def pretty_print(self) -> str:
        agents_str = ",".join(sorted(self.agents, key=lambda x: (x.isdigit(), x)))
        if self.temporal_op == "U" and self.operand2:
            return f"⟨⟨{agents_str}⟩⟩ ({self.operand.pretty_print()} U {self.operand2.pretty_print()})"
        elif self.temporal_op:
            return f"⟨⟨{agents_str}⟩⟩ {self.temporal_op} {self.operand.pretty_print()}"
        else:
            return f"⟨⟨{agents_str}⟩⟩ {self.operand.pretty_print()}"

    def to_ascii(self) -> str:
        agents_str = ",".join(sorted(self.agents, key=lambda x: (x.isdigit(), x)))
        if self.temporal_op == "U" and self.operand2:
            return f"<<{agents_str}>> ({self.operand.to_ascii()} U {self.operand2.to_ascii()})"
        elif self.temporal_op:
            return f"<<{agents_str}>> {self.temporal_op} {self.operand.to_ascii()}"
        else:
            return f"<<{agents_str}>> {self.operand.to_ascii()}"

    def depth(self) -> int:
        base_depth = self.operand.depth() + 1
        if self.operand2:
            return max(base_depth, self.operand2.depth() + 1)
        return base_depth

    def get_atoms(self) -> Set[str]:
        atoms = self.operand.get_atoms()
        if self.operand2:
            atoms |= self.operand2.get_atoms()
        return atoms

    def get_agents(self) -> Set[str]:
        agents = set(self.agents)
        agents |= self.operand.get_agents()
        if self.operand2:
            agents |= self.operand2.get_agents()
        return agents


# =============================================================================
# Parser for ATL Formulas
# =============================================================================


class ATLParser:
    """
    A robust parser for ATL formulas using pyparsing.
    
    Supports both Unicode and ASCII syntax variants.
    
    Grammar (informal):
    -------------------
    formula := atom | true | false | not_formula | binary_formula 
             | temporal_formula | coalition_formula
    atom := [a-zA-Z_][a-zA-Z0-9_]*
    not_formula := (¬ | ! | not) formula
    binary_formula := formula (∧ | & | ∨ | | | → | ->) formula
    temporal_formula := (X | F | G) formula | formula U formula
    coalition_formula := (⟨⟨ | <<) agent_list (⟩⟩ | >>) [temporal_op] formula
    agent_list := agent (, agent)*
    agent := [a-zA-Z0-9_]+
    
    Thread Safety
    -------------
    Parser instances are thread-safe after construction.
    Use get_parser() for a shared global instance.
    """

    def __init__(self):
        self._parser = self._build_parser()

    def _build_parser(self):
        """Build the pyparsing grammar for ATL."""
        
        # Atoms and identifiers
        identifier = Word(alphas + "_", alphanums + "_")
        agent_id = Word(alphanums + "_")

        # Boolean constants
        true_const = (
            Keyword("true") | Keyword("⊤") | Keyword("True")
        ).setParseAction(lambda: BoolTrue())
        
        false_const = (
            Keyword("false") | Keyword("⊥") | Keyword("False")
        ).setParseAction(lambda: BoolFalse())

        # Atomic proposition
        atom = identifier.copy().setParseAction(lambda t: Atom(t[0]))

        # Coalition brackets - support both Unicode and ASCII
        coalition_open = Suppress(Literal("⟨⟨") | Literal("<<"))
        coalition_close = Suppress(Literal("⟩⟩") | Literal(">>"))

        # Agent list inside coalition (can be empty for grand coalition)
        agent_list = Group(Opt(delimitedList(agent_id, delim=",")))

        # Temporal operators
        temporal_op = Literal("X") | Literal("F") | Literal("G")
        until_op = Literal("U")

        # Boolean operators
        not_op = Literal("¬") | Literal("!") | Keyword("not")
        and_op = Literal("∧") | Literal("&") | Keyword("and")
        or_op = Literal("∨") | Literal("|") | Keyword("or")
        implies_op = Literal("→") | Literal("->") | Keyword("implies")

        # Forward declaration for recursive grammar
        expr = Forward()

        # Base terms
        base_term = true_const | false_const | atom

        # Parenthesized expression
        paren_expr = Suppress(Literal("(")) + expr + Suppress(Literal(")"))

        # Standalone temporal operator
        def make_standalone_temporal(tokens):
            return TemporalOp(operator=tokens[0], operand=tokens[1])
        
        standalone_temporal = (temporal_op + expr).setParseAction(make_standalone_temporal)

        # Until expression within parentheses: (φ U ψ)
        until_expr = (
            Suppress(Literal("("))
            + expr
            + Suppress(until_op)
            + expr
            + Suppress(Literal(")"))
        )

        # Coalition handlers
        def make_until_coalition(tokens):
            agents = frozenset(str(a) for a in tokens[0])
            return Coalition(agents=agents, temporal_op="U", operand=tokens[1], operand2=tokens[2])

        def make_temporal_coalition(tokens):
            agents = frozenset(str(a) for a in tokens[0])
            return Coalition(agents=agents, temporal_op=tokens[1], operand=tokens[2])

        def make_simple_coalition(tokens):
            agents = frozenset(str(a) for a in tokens[0])
            return Coalition(agents=agents, temporal_op=None, operand=tokens[1])

        # Coalition with until: ⟨⟨1,2⟩⟩ (p U q)
        coalition_until = (
            coalition_open + agent_list + coalition_close + until_expr
        ).setParseAction(make_until_coalition)

        # Coalition with temporal operator: ⟨⟨1,2⟩⟩ G p
        coalition_temporal = (
            coalition_open + agent_list + coalition_close + temporal_op + expr
        ).setParseAction(make_temporal_coalition)

        # Coalition with parenthesized formula: ⟨⟨1,2⟩⟩ (p -> q)
        coalition_simple = (
            coalition_open + agent_list + coalition_close + paren_expr
        ).setParseAction(make_simple_coalition)

        # Primary expression
        primary = (
            coalition_until | 
            coalition_temporal | 
            coalition_simple | 
            standalone_temporal | 
            paren_expr | 
            base_term
        )

        # Operator handlers for infix notation
        def make_not(tokens):
            return Not(tokens[0][1])

        def make_binary(tokens):
            result = tokens[0][0]
            i = 1
            while i < len(tokens[0]):
                op = tokens[0][i]
                right = tokens[0][i + 1]
                if op in ("∧", "&", "and"):
                    result = And(result, right)
                elif op in ("∨", "|", "or"):
                    result = Or(result, right)
                elif op in ("→", "->", "implies"):
                    result = Implies(result, right)
                elif op == "U":
                    result = Until(result, right)
                i += 2
            return result

        # Expression with operator precedence
        # Precedence (lowest to highest): →, ∨, ∧, U, ¬
        expr <<= infixNotation(
            primary,
            [
                (not_op, 1, opAssoc.RIGHT, make_not),
                (until_op, 2, opAssoc.LEFT, make_binary),
                (and_op, 2, opAssoc.LEFT, make_binary),
                (or_op, 2, opAssoc.LEFT, make_binary),
                (implies_op, 2, opAssoc.LEFT, make_binary),
            ],
        )

        return expr

    def parse(self, text: str) -> ATLFormula:
        """
        Parse a text ATL formula into an AST.
        
        Args:
            text: The ATL formula as a string
            
        Returns:
            An ATLFormula AST node
            
        Raises:
            pp.ParseException: If the formula is syntactically invalid
        """
        result = self._parser.parseString(text, parseAll=True)
        return result[0]


# Global parser instance (lazy initialization)
_parser: Optional[ATLParser] = None


def get_parser() -> ATLParser:
    """
    Get or create the global ATL parser instance.
    
    This provides a shared parser to avoid repeated initialization.
    Thread-safe for parsing operations.
    """
    global _parser
    if _parser is None:
        _parser = ATLParser()
    return _parser


def parse_atl(text: str) -> ATLFormula:
    """
    Parse a text ATL formula into an AST.
    
    This is the main entry point for parsing ATL formulas.
    
    Args:
        text: The ATL formula as a string (Unicode or ASCII)
        
    Returns:
        An ATLFormula AST node
        
    Raises:
        pp.ParseException: If the formula is syntactically invalid
        
    Example:
        >>> formula = parse_atl("<<1,2>> G safe")
        >>> print(formula.pretty_print())
        ⟨⟨1,2⟩⟩ G safe
    """
    return get_parser().parse(text)


# =============================================================================
# Validation
# =============================================================================


@dataclass
class ValidationResult:
    """
    Result of ATL formula validation.
    
    Attributes:
        valid: Whether the formula is valid
        errors: List of error messages (validation failures)
        warnings: List of warning messages (non-fatal issues)
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ATLFragmentConfig:
    """
    Configuration for an ATL fragment.
    
    Used to restrict the allowed operators and structure of formulas.
    """
    allowed_temporal_ops: Set[str] = field(
        default_factory=lambda: {"X", "F", "G", "U"}
    )
    max_nesting_depth: int = 10
    max_coalition_size: int = 10
    allow_empty_coalition: bool = False
    allowed_agents: Optional[Set[str]] = None  # None = any agent allowed


def is_valid(
    formula: Union[ATLFormula, str],
    max_depth: int = 4,
    max_coalition_size: int = 5,
    config: Optional[ATLFragmentConfig] = None,
) -> ValidationResult:
    """
    Check if an ATL formula is syntactically valid.
    
    Validates:
    - Formula can be parsed (if string)
    - Coalition syntax is correct
    - Temporal operators are in valid positions
    - Nesting depth is within limits
    - Coalition sizes are within limits
    
    Args:
        formula: The formula to validate (string or AST)
        max_depth: Maximum allowed nesting depth
        max_coalition_size: Maximum agents in a coalition
        config: Optional fragment configuration for advanced checks
        
    Returns:
        ValidationResult with valid flag and any errors/warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Parse if string
    if isinstance(formula, str):
        try:
            formula = parse_atl(formula)
        except pp.ParseException as e:
            return ValidationResult(valid=False, errors=[f"Parse error: {e}"])

    # Use config if provided, otherwise use simple limits
    if config:
        max_depth = config.max_nesting_depth
        max_coalition_size = config.max_coalition_size

    # Check depth
    depth = formula.depth()
    if depth > max_depth:
        errors.append(f"Nesting depth {depth} exceeds maximum {max_depth}")

    # Recursive validation
    def check_formula(f: ATLFormula) -> None:
        if isinstance(f, Atom):
            if not f.name:
                errors.append("Empty atom name")
            elif not f.name[0].isalpha() and f.name[0] != '_':
                warnings.append(f"Atom name '{f.name}' should start with letter or underscore")
                
        elif isinstance(f, Coalition):
            if len(f.agents) == 0:
                if config and not config.allow_empty_coalition:
                    errors.append("Coalition has empty agent set")
                else:
                    warnings.append("Coalition has empty agent set")
                    
            if len(f.agents) > max_coalition_size:
                errors.append(
                    f"Coalition size {len(f.agents)} exceeds maximum {max_coalition_size}"
                )
                
            if f.temporal_op and f.temporal_op not in ("X", "F", "G", "U"):
                errors.append(f"Unknown temporal operator: {f.temporal_op}")
                
            if config and config.allowed_agents:
                unknown = set(f.agents) - config.allowed_agents
                if unknown:
                    errors.append(f"Unknown agents: {unknown}")
                    
            check_formula(f.operand)
            if f.operand2:
                check_formula(f.operand2)
                
        elif isinstance(f, TemporalOp):
            if f.operator not in ("X", "F", "G"):
                errors.append(f"Unknown temporal operator: {f.operator}")
            check_formula(f.operand)
            
        elif isinstance(f, Until):
            check_formula(f.left)
            check_formula(f.right)
            
        elif isinstance(f, Not):
            check_formula(f.operand)
            
        elif isinstance(f, (And, Or, Implies)):
            check_formula(f.left)
            check_formula(f.right)

    check_formula(formula)

    return ValidationResult(
        valid=len(errors) == 0, 
        errors=errors, 
        warnings=warnings
    )


def validate_atl_string(
    formula_str: str,
    max_depth: int = 4,
    max_coalition_size: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate an ATL formula string.
    
    This is a simpler interface for quick validation checks,
    particularly useful in NL→ATL pipelines.
    
    Args:
        formula_str: The formula as a string
        max_depth: Maximum nesting depth
        max_coalition_size: Maximum coalition size
        
    Returns:
        Tuple of (is_valid, list of error messages)
        
    Example:
        >>> valid, errors = validate_atl_string("<<1,2>> G safe")
        >>> print(valid)
        True
    """
    result = is_valid(formula_str, max_depth, max_coalition_size)
    return (result.valid, result.errors)


# =============================================================================
# Normalization and Pretty-Printing
# =============================================================================


def normalize_atl(formula: Union[ATLFormula, str]) -> str:
    """
    Normalize an ATL formula to a canonical Unicode string.
    
    This ensures consistent formatting:
    - Sorted agent sets in coalitions
    - Consistent use of Unicode operators
    - Minimal parenthesization
    
    Args:
        formula: Either an ATLFormula AST or a string to parse
        
    Returns:
        A normalized Unicode string representation
    """
    if isinstance(formula, str):
        formula = parse_atl(formula)
    return formula.pretty_print()


def pretty_print(formula: Union[ATLFormula, str]) -> str:
    """
    Generate a human-readable Unicode representation.
    
    Alias for normalize_atl for convenience.
    """
    return normalize_atl(formula)


def to_ascii(formula: Union[ATLFormula, str]) -> str:
    """
    Generate an ASCII-only representation.
    
    Useful when Unicode is not supported.
    """
    if isinstance(formula, str):
        formula = parse_atl(formula)
    return formula.to_ascii()


# =============================================================================
# Utility Functions for Analysis
# =============================================================================


def extract_components(formula: Union[ATLFormula, str]) -> dict:
    """
    Extract structural components from an ATL formula.
    
    This is useful for analyzing formula structure, computing statistics,
    or filtering datasets by formula characteristics.
    
    Args:
        formula: The formula to analyze
        
    Returns:
        Dictionary with:
        - atoms: set of atomic propositions
        - agents: set of agent identifiers
        - operators: list of temporal operators used
        - depth: nesting depth
        - has_coalition: whether formula has coalition modality
        - coalition_count: number of coalitions
        - max_coalition_size: size of largest coalition
    """
    if isinstance(formula, str):
        formula = parse_atl(formula)

    operators: List[str] = []
    coalition_count = 0
    max_coalition_size = 0

    def collect(f: ATLFormula) -> None:
        nonlocal coalition_count, max_coalition_size
        
        if isinstance(f, Coalition):
            coalition_count += 1
            max_coalition_size = max(max_coalition_size, len(f.agents))
            if f.temporal_op:
                operators.append(f.temporal_op)
            collect(f.operand)
            if f.operand2:
                collect(f.operand2)
                
        elif isinstance(f, TemporalOp):
            operators.append(f.operator)
            collect(f.operand)
            
        elif isinstance(f, Until):
            operators.append("U")
            collect(f.left)
            collect(f.right)
            
        elif isinstance(f, Not):
            collect(f.operand)
            
        elif isinstance(f, (And, Or, Implies)):
            collect(f.left)
            collect(f.right)

    collect(formula)

    return {
        "atoms": formula.get_atoms(),
        "agents": formula.get_agents(),
        "operators": operators,
        "depth": formula.depth(),
        "has_coalition": coalition_count > 0,
        "coalition_count": coalition_count,
        "max_coalition_size": max_coalition_size,
    }


def get_coalition_info(formula: Union[ATLFormula, str]) -> List[dict]:
    """
    Extract information about all coalitions in a formula.
    
    Args:
        formula: The formula to analyze
        
    Returns:
        List of dicts, each with:
        - agents: sorted list of agent names
        - operator: temporal operator (X, F, G, U, or None)
        - has_until: whether it's an Until formula
    """
    if isinstance(formula, str):
        formula = parse_atl(formula)

    coalitions: List[dict] = []

    def collect(f: ATLFormula) -> None:
        if isinstance(f, Coalition):
            coalitions.append({
                "agents": sorted(f.agents),
                "operator": f.temporal_op,
                "has_until": f.operand2 is not None,
            })
            collect(f.operand)
            if f.operand2:
                collect(f.operand2)
        elif isinstance(f, TemporalOp):
            collect(f.operand)
        elif isinstance(f, Until):
            collect(f.left)
            collect(f.right)
        elif isinstance(f, Not):
            collect(f.operand)
        elif isinstance(f, (And, Or, Implies)):
            collect(f.left)
            collect(f.right)

    collect(formula)
    return coalitions


def get_operator_stats(formula: Union[ATLFormula, str]) -> dict:
    """
    Get statistics about operators used in a formula.
    
    Args:
        formula: The formula to analyze
        
    Returns:
        Dictionary with operator counts (X, F, G, U)
    """
    components = extract_components(formula)
    ops = components["operators"]
    
    return {
        "X": ops.count("X"),
        "F": ops.count("F"),
        "G": ops.count("G"),
        "U": ops.count("U"),
    }


# =============================================================================
# Formula Construction Helpers
# =============================================================================


def make_coalition(
    agents: Union[List[str], Set[str]],
    temporal_op: str,
    operand: ATLFormula,
    operand2: Optional[ATLFormula] = None
) -> Coalition:
    """
    Helper function to construct a Coalition formula.
    
    Args:
        agents: List or set of agent identifiers
        temporal_op: Temporal operator (X, F, G, U)
        operand: Main operand
        operand2: Second operand (for Until)
        
    Returns:
        Coalition formula
    """
    return Coalition(
        agents=frozenset(str(a) for a in agents),
        temporal_op=temporal_op,
        operand=operand,
        operand2=operand2
    )


def make_safety(agents: Union[List[str], Set[str]], prop: str) -> Coalition:
    """
    Create a safety formula: ⟨⟨agents⟩⟩ G ¬prop
    
    Agents can ensure the bad property never happens.
    """
    return make_coalition(agents, "G", Not(Atom(prop)))


def make_reachability(agents: Union[List[str], Set[str]], prop: str) -> Coalition:
    """
    Create a reachability formula: ⟨⟨agents⟩⟩ F prop
    
    Agents can eventually achieve the property.
    """
    return make_coalition(agents, "F", Atom(prop))


def make_invariant(agents: Union[List[str], Set[str]], prop: str) -> Coalition:
    """
    Create an invariant formula: ⟨⟨agents⟩⟩ G prop
    
    Agents can maintain the property forever.
    """
    return make_coalition(agents, "G", Atom(prop))


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    # Example formulas to test parsing
    examples = [
        "<<1,2>> G safe",
        "<<1>> F goal",
        "<<1,2,3>> (waiting U served)",
        "<<controller>> G (request -> F response)",
        "<<a,b>> X (p & q)",
        "!<<1>> F error",
        "⟨⟨robot⟩⟩ G ¬crash",
        "⟨⟨1,2⟩⟩ (temp_stable U alarm)",
    ]
    
    print("ATL Syntax Module - Examples")
    print("=" * 70)
    
    for text in examples:
        print(f"\nInput:      {text}")
        try:
            formula = parse_atl(text)
            print(f"Unicode:    {formula.pretty_print()}")
            print(f"ASCII:      {formula.to_ascii()}")
            print(f"Atoms:      {formula.get_atoms()}")
            print(f"Agents:     {formula.get_agents()}")
            print(f"Depth:      {formula.depth()}")
            
            result = is_valid(formula)
            status = "✓ Valid" if result.valid else f"✗ Invalid: {result.errors}"
            print(f"Validation: {status}")
            
        except Exception as e:
            print(f"Error:      {e}")
    
    print("\n" + "=" * 70)
    print("Module loaded successfully.")
