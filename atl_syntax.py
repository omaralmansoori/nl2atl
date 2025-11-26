"""
ATL Syntax Module

Defines data structures and utilities for ATL (Alternating-time Temporal Logic) formulas:
- Data classes representing ATL formula AST nodes
- Parser for text ATL formulas 
- Pretty-printing and normalization functions
- Syntactic validity checking
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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
    pyparsing_common,
)


# =============================================================================
# AST Data Classes for ATL Formulas
# =============================================================================


@dataclass
class ATLFormula(ABC):
    """Abstract base class for all ATL formula nodes."""

    @abstractmethod
    def pretty_print(self) -> str:
        """Return a human-readable string representation."""
        pass

    @abstractmethod
    def depth(self) -> int:
        """Return the nesting depth of the formula."""
        pass

    @abstractmethod
    def get_atoms(self) -> set[str]:
        """Return all atomic propositions in this formula."""
        pass

    @abstractmethod
    def get_agents(self) -> set[str]:
        """Return all agent identifiers in this formula."""
        pass


@dataclass
class Atom(ATLFormula):
    """Atomic proposition (e.g., 'p', 'crash', 'goal_reached')."""

    name: str

    def pretty_print(self) -> str:
        return self.name

    def depth(self) -> int:
        return 0

    def get_atoms(self) -> set[str]:
        return {self.name}

    def get_agents(self) -> set[str]:
        return set()


@dataclass
class BoolTrue(ATLFormula):
    """Boolean constant True."""

    def pretty_print(self) -> str:
        return "⊤"

    def depth(self) -> int:
        return 0

    def get_atoms(self) -> set[str]:
        return set()

    def get_agents(self) -> set[str]:
        return set()


@dataclass
class BoolFalse(ATLFormula):
    """Boolean constant False."""

    def pretty_print(self) -> str:
        return "⊥"

    def depth(self) -> int:
        return 0

    def get_atoms(self) -> set[str]:
        return set()

    def get_agents(self) -> set[str]:
        return set()


@dataclass
class Not(ATLFormula):
    """Negation: ¬φ"""

    operand: ATLFormula

    def pretty_print(self) -> str:
        return f"¬{self.operand.pretty_print()}"

    def depth(self) -> int:
        return self.operand.depth()

    def get_atoms(self) -> set[str]:
        return self.operand.get_atoms()

    def get_agents(self) -> set[str]:
        return self.operand.get_agents()


@dataclass
class And(ATLFormula):
    """Conjunction: φ ∧ ψ"""

    left: ATLFormula
    right: ATLFormula

    def pretty_print(self) -> str:
        return f"({self.left.pretty_print()} ∧ {self.right.pretty_print()})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth())

    def get_atoms(self) -> set[str]:
        return self.left.get_atoms() | self.right.get_atoms()

    def get_agents(self) -> set[str]:
        return self.left.get_agents() | self.right.get_agents()


@dataclass
class Or(ATLFormula):
    """Disjunction: φ ∨ ψ"""

    left: ATLFormula
    right: ATLFormula

    def pretty_print(self) -> str:
        return f"({self.left.pretty_print()} ∨ {self.right.pretty_print()})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth())

    def get_atoms(self) -> set[str]:
        return self.left.get_atoms() | self.right.get_atoms()

    def get_agents(self) -> set[str]:
        return self.left.get_agents() | self.right.get_agents()


@dataclass
class Implies(ATLFormula):
    """Implication: φ → ψ"""

    left: ATLFormula
    right: ATLFormula

    def pretty_print(self) -> str:
        return f"({self.left.pretty_print()} → {self.right.pretty_print()})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth())

    def get_atoms(self) -> set[str]:
        return self.left.get_atoms() | self.right.get_atoms()

    def get_agents(self) -> set[str]:
        return self.left.get_agents() | self.right.get_agents()


@dataclass
class TemporalOp(ATLFormula):
    """
    Standalone temporal operator: X φ, F φ, G φ
    Used when temporal operators appear outside of coalition context.
    """

    operator: str  # One of: X, F, G
    operand: ATLFormula

    def pretty_print(self) -> str:
        return f"{self.operator} {self.operand.pretty_print()}"

    def depth(self) -> int:
        return self.operand.depth() + 1

    def get_atoms(self) -> set[str]:
        return self.operand.get_atoms()

    def get_agents(self) -> set[str]:
        return self.operand.get_agents()


@dataclass
class Coalition(ATLFormula):
    """
    Strategic modality with coalition: ⟨⟨Γ⟩⟩ φ
    Γ is a set of agent identifiers.
    """

    agents: set[str]
    temporal_op: str  # One of: X, F, G, or None for base coalition
    operand: ATLFormula
    # For Until operator, we need a second operand
    operand2: Optional[ATLFormula] = None

    def pretty_print(self) -> str:
        agents_str = ",".join(sorted(self.agents))
        if self.temporal_op == "U" and self.operand2:
            return f"⟨⟨{agents_str}⟩⟩ ({self.operand.pretty_print()} U {self.operand2.pretty_print()})"
        elif self.temporal_op:
            return f"⟨⟨{agents_str}⟩⟩ {self.temporal_op} {self.operand.pretty_print()}"
        else:
            return f"⟨⟨{agents_str}⟩⟩ {self.operand.pretty_print()}"

    def depth(self) -> int:
        base_depth = self.operand.depth() + 1
        if self.operand2:
            return max(base_depth, self.operand2.depth() + 1)
        return base_depth

    def get_atoms(self) -> set[str]:
        atoms = self.operand.get_atoms()
        if self.operand2:
            atoms |= self.operand2.get_atoms()
        return atoms

    def get_agents(self) -> set[str]:
        agents = self.agents.copy()
        agents |= self.operand.get_agents()
        if self.operand2:
            agents |= self.operand2.get_agents()
        return agents


# =============================================================================
# Parser for ATL Formulas
# =============================================================================


class ATLParser:
    """
    A lightweight parser for ATL formulas using pyparsing.
    
    Supports:
    - Coalition modalities: ⟨⟨agents⟩⟩ or <<agents>>
    - Temporal operators: X (next), F (eventually), G (always), U (until)
    - Boolean connectives: ∧/&/and, ∨/|/or, ¬/!/not, →/->/implies
    - Atomic propositions: alphanumeric identifiers
    - Parentheses for grouping
    """

    def __init__(self):
        self._parser = self._build_parser()

    def _build_parser(self):
        # Atoms and identifiers
        identifier = Word(alphas + "_", alphanums + "_")
        agent_id = Word(alphanums + "_")

        # Boolean constants
        true_const = (Keyword("true") | Keyword("⊤") | Keyword("True")).setParseAction(
            lambda _s, _l, _t: BoolTrue()
        )
        false_const = (
            Keyword("false") | Keyword("⊥") | Keyword("False")
        ).setParseAction(lambda _s, _l, _t: BoolFalse())

        # Atomic proposition
        atom = identifier.copy().setParseAction(lambda _s, _l, t: Atom(t[0]))

        # Coalition brackets - support both Unicode and ASCII versions
        coalition_open = Suppress(Literal("⟨⟨") | Literal("<<"))
        coalition_close = Suppress(Literal("⟩⟩") | Literal(">>"))

        # Agent list inside coalition
        agent_list = Group(delimitedList(agent_id, delim=","))

        # Temporal operators as literals for flexibility
        temporal_op = Literal("X") | Literal("F") | Literal("G")
        until_op = Literal("U")

        # Boolean operators - support multiple representations
        not_op = Literal("¬") | Literal("!") | Keyword("not")
        and_op = Literal("∧") | Literal("&") | Keyword("and")
        or_op = Literal("∨") | Literal("|") | Keyword("or")
        implies_op = Literal("→") | Literal("->") | Keyword("implies")

        # Forward declaration for recursive formula
        expr = Forward()

        # Base terms
        base_term = true_const | false_const | atom

        # Parenthesized expression
        paren_expr = Suppress(Literal("(")) + expr + Suppress(Literal(")"))

        # Standalone temporal operator (F, G, X) applied to a formula
        def make_standalone_temporal(tokens):
            op = tokens[0]
            operand = tokens[1]
            return TemporalOp(operator=op, operand=operand)
        
        # Primary terms without coalition - includes standalone temporal ops
        standalone_temporal = (temporal_op + expr).setParseAction(make_standalone_temporal)

        # Until expression within parentheses: (φ U ψ)
        until_expr = (
            Suppress(Literal("("))
            + expr
            + Suppress(until_op)
            + expr
            + Suppress(Literal(")"))
        )

        def make_until_coalition(tokens):
            agents = set(tokens[0])
            left = tokens[1]
            right = tokens[2]
            return Coalition(agents=agents, temporal_op="U", operand=left, operand2=right)

        # Coalition with until
        coalition_until = (
            coalition_open + agent_list + coalition_close + until_expr
        ).setParseAction(make_until_coalition)

        def make_temporal_coalition(tokens):
            agents = set(tokens[0])
            op = tokens[1]
            operand = tokens[2]
            return Coalition(agents=agents, temporal_op=op, operand=operand)

        # Coalition with simple temporal operator (X, F, G)
        coalition_temporal = (
            coalition_open + agent_list + coalition_close + temporal_op + expr
        ).setParseAction(make_temporal_coalition)

        def make_simple_coalition(tokens):
            agents = set(tokens[0])
            operand = tokens[1]
            return Coalition(agents=agents, temporal_op=None, operand=operand)

        # Coalition without temporal operator (just the formula)
        coalition_simple = (
            coalition_open + agent_list + coalition_close + paren_expr
        ).setParseAction(make_simple_coalition)

        # Primary expression including coalitions
        primary = coalition_until | coalition_temporal | coalition_simple | standalone_temporal | paren_expr | base_term

        # Define operators for infix notation
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
                i += 2
            return result

        # Expression with operator precedence
        expr <<= infixNotation(
            primary,
            [
                (not_op, 1, opAssoc.RIGHT, make_not),
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
            ParseException: If the formula is syntactically invalid
        """
        result = self._parser.parseString(text, parseAll=True)
        return result[0]


# Global parser instance
_parser = None


def get_parser() -> ATLParser:
    """Get or create the global ATL parser instance."""
    global _parser
    if _parser is None:
        _parser = ATLParser()
    return _parser


def parse_atl(text: str) -> ATLFormula:
    """
    Parse a text ATL formula into an AST.
    
    Args:
        text: The ATL formula as a string
        
    Returns:
        An ATLFormula AST node
        
    Raises:
        ParseException: If the formula is syntactically invalid
    """
    return get_parser().parse(text)


# =============================================================================
# Normalization and Pretty-Printing
# =============================================================================


def normalize_atl(formula: ATLFormula | str) -> str:
    """
    Normalize an ATL formula to a canonical string representation.
    
    This ensures consistent formatting:
    - Sorted agent sets in coalitions
    - Consistent use of Unicode operators
    - Minimal parenthesization
    
    Args:
        formula: Either an ATLFormula AST or a string to parse
        
    Returns:
        A normalized string representation
    """
    if isinstance(formula, str):
        formula = parse_atl(formula)
    return formula.pretty_print()


def pretty_print(formula: ATLFormula | str) -> str:
    """
    Generate a human-readable representation of an ATL formula.
    
    Alias for normalize_atl for convenience.
    """
    return normalize_atl(formula)


# =============================================================================
# Validity Checking
# =============================================================================


@dataclass
class ValidationResult:
    """Result of ATL formula validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def is_valid(
    formula: ATLFormula | str,
    max_depth: int = 4,
    max_coalition_size: int = 5,
) -> ValidationResult:
    """
    Check if an ATL formula is syntactically valid.
    
    Validates:
    - Formula can be parsed (if string)
    - Coalition syntax is correct (balanced brackets, non-empty agent sets)
    - Temporal operators are in valid positions
    - Nesting depth is within limits
    - Coalition sizes are within limits
    
    Args:
        formula: The formula to validate (string or AST)
        max_depth: Maximum allowed nesting depth
        max_coalition_size: Maximum agents in a coalition
        
    Returns:
        ValidationResult with valid flag and any errors/warnings
    """
    errors = []
    warnings = []

    # Parse if string
    if isinstance(formula, str):
        try:
            formula = parse_atl(formula)
        except pp.ParseException as e:
            return ValidationResult(valid=False, errors=[f"Parse error: {e}"])

    # Check depth
    depth = formula.depth()
    if depth > max_depth:
        errors.append(f"Nesting depth {depth} exceeds maximum {max_depth}")

    # Check coalitions
    def check_coalitions(f: ATLFormula) -> None:
        if isinstance(f, Coalition):
            if len(f.agents) == 0:
                errors.append("Coalition has empty agent set")
            if len(f.agents) > max_coalition_size:
                errors.append(
                    f"Coalition size {len(f.agents)} exceeds maximum {max_coalition_size}"
                )
            if f.temporal_op and f.temporal_op not in ("X", "F", "G", "U"):
                errors.append(f"Unknown temporal operator: {f.temporal_op}")
            check_coalitions(f.operand)
            if f.operand2:
                check_coalitions(f.operand2)
        elif isinstance(f, TemporalOp):
            if f.operator not in ("X", "F", "G"):
                errors.append(f"Unknown temporal operator: {f.operator}")
            check_coalitions(f.operand)
        elif isinstance(f, Not):
            check_coalitions(f.operand)
        elif isinstance(f, (And, Or, Implies)):
            check_coalitions(f.left)
            check_coalitions(f.right)

    check_coalitions(formula)

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_atl_string(
    formula_str: str,
    max_depth: int = 4,
    max_coalition_size: int = 5,
) -> tuple[bool, list[str]]:
    """
    Convenience function to validate an ATL formula string.
    
    Args:
        formula_str: The formula as a string
        max_depth: Maximum nesting depth
        max_coalition_size: Maximum coalition size
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    result = is_valid(formula_str, max_depth, max_coalition_size)
    return (result.valid, result.errors)


# =============================================================================
# Utility Functions
# =============================================================================


def extract_components(formula: ATLFormula | str) -> dict:
    """
    Extract structural components from an ATL formula.
    
    Args:
        formula: The formula to analyze
        
    Returns:
        Dictionary with:
        - atoms: set of atomic propositions
        - agents: set of agent identifiers
        - operators: list of temporal operators used
        - depth: nesting depth
    """
    if isinstance(formula, str):
        formula = parse_atl(formula)

    operators = []

    def collect_ops(f: ATLFormula) -> None:
        if isinstance(f, Coalition):
            if f.temporal_op:
                operators.append(f.temporal_op)
            collect_ops(f.operand)
            if f.operand2:
                collect_ops(f.operand2)
        elif isinstance(f, TemporalOp):
            operators.append(f.operator)
            collect_ops(f.operand)
        elif isinstance(f, Not):
            collect_ops(f.operand)
        elif isinstance(f, (And, Or, Implies)):
            collect_ops(f.left)
            collect_ops(f.right)

    collect_ops(formula)

    return {
        "atoms": formula.get_atoms(),
        "agents": formula.get_agents(),
        "operators": operators,
        "depth": formula.depth(),
    }


def get_coalition_info(formula: ATLFormula | str) -> list[dict]:
    """
    Extract information about all coalitions in a formula.
    
    Args:
        formula: The formula to analyze
        
    Returns:
        List of dicts with coalition info (agents, operator, etc.)
    """
    if isinstance(formula, str):
        formula = parse_atl(formula)

    coalitions = []

    def collect_coalitions(f: ATLFormula) -> None:
        if isinstance(f, Coalition):
            coalitions.append(
                {
                    "agents": sorted(f.agents),
                    "operator": f.temporal_op,
                    "has_until": f.operand2 is not None,
                }
            )
            collect_coalitions(f.operand)
            if f.operand2:
                collect_coalitions(f.operand2)
        elif isinstance(f, TemporalOp):
            collect_coalitions(f.operand)
        elif isinstance(f, Not):
            collect_coalitions(f.operand)
        elif isinstance(f, (And, Or, Implies)):
            collect_coalitions(f.left)
            collect_coalitions(f.right)

    collect_coalitions(formula)
    return coalitions
