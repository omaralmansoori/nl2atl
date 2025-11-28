# Verification-Aware NL2ATL Project

This project provides NL→ATL (Natural Language to Alternating-time Temporal Logic) translation and dataset generation capabilities.

## Project Structure

- `atl_syntax.py` - ATL syntax module with AST, parser, and validator
- `nl2atl.py` - NL to ATL translation with LLM integration
- `dataset_gen.py` - Dataset generation pipeline with CLI
- `evaluation.py` - Basic evaluation and quality stats
- `config/` - Configuration files for ATL fragments and templates

## Development Guidelines

- Use Python 3.10+
- Follow PEP 8 style guidelines
- Use type hints throughout
- Environment variables for API keys (OPENAI_API_KEY, etc.)

## Key Dependencies

- openai - LLM API client
- pyparsing - ATL formula parsing
- pydantic - Data validation
- pyyaml - Configuration files
- click - CLI interface
- python-dotenv - Environment management

## Setup Checklist

- [x] Create copilot-instructions.md
- [x] Scaffold project structure
- [x] Implement atl_syntax.py
- [x] Implement nl2atl.py
- [x] Implement dataset_gen.py
- [x] Implement evaluation.py
- [x] Create config files
- [x] Create README.md
- [x] Install Python extension
- [x] Verify project compiles
