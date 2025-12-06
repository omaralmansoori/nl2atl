#!/usr/bin/env python3
"""
Interactive NL2ATL Interface
============================

An interactive command-line interface for translating natural language
requirements to ATL formulas using fine-tuned or base models.

Features:
- Interactive REPL mode
- Single translation mode
- Batch translation from file
- Syntax validation of generated formulas
- Support for custom models (fine-tuned or base)

Usage Examples
--------------
# Interactive mode with fine-tuned model
python interact.py --model ft:gpt-4o-mini-2024-07-18:personal:nl2atl-v1:CiGKGvnC

# Single translation
python interact.py translate "The robot can eventually reach the goal"

# Batch translation from file
python interact.py batch input.txt -o output.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv

# Import local modules
from atl_syntax import is_valid, parse_atl, validate_atl_string

# Load environment variables
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:nl2atl-v1:CiGKGvnC"

SYSTEM_PROMPT = (
    "You are an expert in translating natural language requirements into "
    "Alternating-time Temporal Logic (ATL) formulas. Given a natural language "
    "statement describing agent capabilities and temporal properties, generate "
    "the corresponding ATL formula using proper syntax with coalition operators "
    "⟨⟨...⟩⟩, temporal operators (G, F, X, U), and logical operators (∧, ∨, →, ¬)."
)


# =============================================================================
# Translation Functions
# =============================================================================

def get_openai_client():
    """Get OpenAI client instance."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai package: pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return OpenAI(api_key=api_key)


def translate_nl_to_atl(
    nl_statement: str,
    model: str = DEFAULT_MODEL,
    domain: Optional[str] = None,
    agents: Optional[List[str]] = None,
    temperature: float = 0.0,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Translate a natural language statement to ATL formula.
    
    Args:
        nl_statement: The natural language requirement
        model: Model to use for translation
        domain: Optional domain context
        agents: Optional list of agents
        temperature: Sampling temperature
        validate: Whether to validate the generated formula
        
    Returns:
        Dictionary with translation results
    """
    client = get_openai_client()
    
    # Build user message
    user_content = f"Translate the following natural language requirement to ATL:\n\n{nl_statement}"
    
    if domain:
        user_content += f"\n\nDomain: {domain}"
    
    if agents:
        user_content += f"\n\nAgents: {', '.join(agents)}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    
    atl_formula = response.choices[0].message.content.strip()
    
    result = {
        "nl_statement": nl_statement,
        "atl_formula": atl_formula,
        "model": model,
        "domain": domain,
        "agents": agents,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
    
    if validate:
        is_syntax_valid, errors = validate_atl_string(atl_formula)
        result["syntax_valid"] = is_syntax_valid
        result["syntax_errors"] = errors
    
    return result


def format_atl_unicode(formula: str) -> str:
    """Convert ASCII ATL to Unicode representation."""
    replacements = {
        "<<": "⟨⟨",
        ">>": "⟩⟩",
        "->": "→",
        "&": "∧",
        "|": "∨",
        "!": "¬",
    }
    result = formula
    for ascii_char, unicode_char in replacements.items():
        result = result.replace(ascii_char, unicode_char)
    return result


# =============================================================================
# CLI Commands
# =============================================================================

@click.group()
@click.option(
    '--model', '-m',
    type=str,
    default=DEFAULT_MODEL,
    envvar='NL2ATL_MODEL',
    help=f'Model to use for translation (default: {DEFAULT_MODEL}).',
)
@click.pass_context
def cli(ctx, model: str):
    """Interactive NL2ATL Translation Interface."""
    ctx.ensure_object(dict)
    ctx.obj['model'] = model


@cli.command()
@click.argument('statement', required=False)
@click.option('--domain', '-d', type=str, help='Domain context.')
@click.option('--agents', '-a', type=str, help='Comma-separated list of agents.')
@click.option('--temperature', '-t', type=float, default=0.0, help='Sampling temperature.')
@click.pass_context
def translate(ctx, statement: Optional[str], domain: Optional[str], agents: Optional[str], temperature: float):
    """
    Translate a single NL statement to ATL.
    
    If no statement is provided, enters interactive mode.
    """
    model = ctx.obj['model']
    agents_list = [a.strip() for a in agents.split(',')] if agents else None
    
    if statement:
        # Single translation
        result = translate_nl_to_atl(
            statement,
            model=model,
            domain=domain,
            agents=agents_list,
            temperature=temperature,
        )
        
        click.echo(f"\n{'='*60}")
        click.echo("Translation Result")
        click.echo(f"{'='*60}")
        click.echo(f"\nNL: {result['nl_statement']}")
        click.echo(f"\nATL (ASCII): {result['atl_formula']}")
        click.echo(f"ATL (Unicode): {format_atl_unicode(result['atl_formula'])}")
        click.echo(f"\nSyntax Valid: {result.get('syntax_valid', 'N/A')}")
        if result.get('syntax_errors'):
            click.echo(f"Errors: {result['syntax_errors']}")
        click.echo(f"\nTokens: {result['usage']['total_tokens']}")
    else:
        # Interactive mode
        interactive_mode(model, domain, agents_list, temperature)


def interactive_mode(
    model: str,
    default_domain: Optional[str] = None,
    default_agents: Optional[List[str]] = None,
    temperature: float = 0.0,
):
    """Run interactive REPL mode."""
    click.echo(f"\n{'='*60}")
    click.echo("NL2ATL Interactive Mode")
    click.echo(f"{'='*60}")
    click.echo(f"Model: {model}")
    click.echo(f"Temperature: {temperature}")
    click.echo("\nCommands:")
    click.echo("  :quit, :q     - Exit interactive mode")
    click.echo("  :domain <d>   - Set domain context")
    click.echo("  :agents <a,b> - Set agents list")
    click.echo("  :clear        - Clear context")
    click.echo("  :help         - Show this help")
    click.echo("\nEnter NL statements to translate:\n")
    
    domain = default_domain
    agents = default_agents
    
    while True:
        try:
            # Read input
            statement = click.prompt("NL", prompt_suffix="> ", default="", show_default=False)
            statement = statement.strip()
            
            if not statement:
                continue
            
            # Handle commands
            if statement.startswith(':'):
                cmd_parts = statement[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else None
                
                if cmd in ('quit', 'q', 'exit'):
                    click.echo("Goodbye!")
                    break
                elif cmd == 'domain':
                    domain = arg
                    click.echo(f"Domain set to: {domain}")
                elif cmd == 'agents':
                    agents = [a.strip() for a in arg.split(',')] if arg else None
                    click.echo(f"Agents set to: {agents}")
                elif cmd == 'clear':
                    domain = None
                    agents = None
                    click.echo("Context cleared.")
                elif cmd == 'help':
                    click.echo("\nCommands:")
                    click.echo("  :quit, :q     - Exit interactive mode")
                    click.echo("  :domain <d>   - Set domain context")
                    click.echo("  :agents <a,b> - Set agents list")
                    click.echo("  :clear        - Clear context")
                    click.echo("  :help         - Show this help\n")
                else:
                    click.echo(f"Unknown command: {cmd}")
                continue
            
            # Translate
            try:
                result = translate_nl_to_atl(
                    statement,
                    model=model,
                    domain=domain,
                    agents=agents,
                    temperature=temperature,
                )
                
                click.echo(f"\nATL: {result['atl_formula']}")
                click.echo(f"     {format_atl_unicode(result['atl_formula'])}")
                
                if not result.get('syntax_valid', True):
                    click.echo(f"⚠️  Syntax errors: {result.get('syntax_errors', [])}")
                else:
                    click.echo("✓ Syntax valid")
                click.echo()
                
            except Exception as e:
                click.echo(f"Error: {e}\n", err=True)
                
        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file (JSONL format).')
@click.option('--domain', '-d', type=str, help='Domain context for all statements.')
@click.option('--temperature', '-t', type=float, default=0.0, help='Sampling temperature.')
@click.pass_context
def batch(ctx, input_file: Path, output: Optional[Path], domain: Optional[str], temperature: float):
    """
    Translate multiple NL statements from a file.
    
    Input file should have one NL statement per line.
    """
    model = ctx.obj['model']
    
    # Read input statements
    with open(input_file, 'r', encoding='utf-8') as f:
        statements = [line.strip() for line in f if line.strip()]
    
    click.echo(f"Translating {len(statements)} statements...")
    
    results = []
    for i, statement in enumerate(statements, 1):
        try:
            result = translate_nl_to_atl(
                statement,
                model=model,
                domain=domain,
                temperature=temperature,
            )
            results.append(result)
            
            status = "✓" if result.get('syntax_valid', False) else "✗"
            click.echo(f"[{i}/{len(statements)}] {status} {statement[:50]}...")
            
        except Exception as e:
            click.echo(f"[{i}/{len(statements)}] ✗ Error: {e}", err=True)
            results.append({
                "nl_statement": statement,
                "error": str(e),
            })
    
    # Write output
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        click.echo(f"\nResults saved to: {output}")
    
    # Summary
    valid_count = sum(1 for r in results if r.get('syntax_valid', False))
    click.echo(f"\nSummary: {valid_count}/{len(results)} valid formulas")


@cli.command()
@click.pass_context
def repl(ctx):
    """Start interactive REPL mode."""
    model = ctx.obj['model']
    interactive_mode(model)


if __name__ == "__main__":
    cli()
