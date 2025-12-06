#!/usr/bin/env python3
"""
OpenAI Fine-tuning Script for NL2ATL Models
============================================

This script provides functionality to fine-tune OpenAI models using
the generated NL-to-ATL dataset.

Features:
- Support for multiple data sources (JSONL files)
- Configurable base model selection
- Data validation and preprocessing
- Fine-tuning job management (create, list, cancel, status)
- Training metrics monitoring

Usage Examples
--------------
# Fine-tune with a single source
python finetune.py train --source data/verified_100.jsonl

# Fine-tune with multiple sources
python finetune.py train --source data/verified_100.jsonl --source data/verified_20.jsonl

# Fine-tune with custom model and hyperparameters
python finetune.py train --source data/verified.jsonl --model gpt-4o-mini-2024-07-18 --epochs 3

# List all fine-tuning jobs
python finetune.py list

# Check job status
python finetune.py status --job-id ftjob-xxxxx

# Cancel a job
python finetune.py cancel --job-id ftjob-xxxxx

# Validate dataset without training
python finetune.py validate --source data/verified.jsonl

Environment Variables
---------------------
- OPENAI_API_KEY: Required for OpenAI API access
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning jobs."""
    
    # Data sources (JSONL files)
    sources: List[Path] = field(default_factory=list)
    
    # Base model to fine-tune
    model: str = "gpt-4o-mini-2024-07-18"
    
    # Training hyperparameters
    n_epochs: Optional[int] = None  # None = auto
    batch_size: Optional[int] = None  # None = auto
    learning_rate_multiplier: Optional[float] = None  # None = auto
    
    # Job metadata
    suffix: Optional[str] = None  # Custom suffix for the fine-tuned model name
    
    # Validation split
    validation_split: float = 0.1
    
    # System prompt for the fine-tuned model
    system_prompt: str = (
        "You are an expert in translating natural language requirements into "
        "Alternating-time Temporal Logic (ATL) formulas. Given a natural language "
        "statement describing agent capabilities and temporal properties, generate "
        "the corresponding ATL formula using proper syntax with coalition operators "
        "⟨⟨...⟩⟩, temporal operators (G, F, X, U), and logical operators (∧, ∨, →, ¬)."
    )
    
    # Whether to include domain context
    include_domain: bool = True
    
    # Whether to include agents list
    include_agents: bool = True
    
    # Filtering options
    min_confidence: float = 0.0
    verification_status_filter: str = "verified"
    domain_filter: Optional[List[str]] = None
    
    # Reproducibility
    seed: Optional[int] = None  # Seed for reproducible training
    
    @classmethod
    def from_yaml(cls, filepath: Path) -> "FinetuneConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            filepath: Path to the YAML configuration file
            
        Returns:
            FinetuneConfig instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Convert source paths
        sources = []
        if 'sources' in data:
            for source in data['sources']:
                sources.append(Path(source))
        
        return cls(
            sources=sources,
            model=data.get('model', cls.model),
            n_epochs=data.get('n_epochs'),
            batch_size=data.get('batch_size'),
            learning_rate_multiplier=data.get('learning_rate_multiplier'),
            suffix=data.get('suffix'),
            validation_split=data.get('validation_split', cls.validation_split),
            system_prompt=data.get('system_prompt', cls.system_prompt),
            include_domain=data.get('include_domain', cls.include_domain),
            include_agents=data.get('include_agents', cls.include_agents),
            min_confidence=data.get('min_confidence', cls.min_confidence),
            verification_status_filter=data.get('verification_status_filter', cls.verification_status_filter),
            domain_filter=data.get('domain_filter'),
            seed=data.get('seed'),
        )


# Supported models for fine-tuning
SUPPORTED_MODELS = [
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "gpt-4-0613",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
]


# =============================================================================
# Data Processing
# =============================================================================

def load_samples_from_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load samples from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of sample dictionaries
    """
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                click.echo(f"Warning: Skipping invalid JSON at line {line_num} in {filepath}: {e}", err=True)
    return samples


def load_samples_from_json(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load samples from a JSON file (with 'samples' array).
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of sample dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'samples' in data:
        return data['samples']
    else:
        raise ValueError(f"Unsupported JSON structure in {filepath}")


def load_samples(sources: List[Path]) -> List[Dict[str, Any]]:
    """
    Load samples from multiple source files.
    
    Supports both .jsonl and .json files.
    
    Args:
        sources: List of file paths
        
    Returns:
        Combined list of samples
    """
    all_samples = []
    
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if source.suffix == '.jsonl':
            samples = load_samples_from_jsonl(source)
        elif source.suffix == '.json':
            samples = load_samples_from_json(source)
        else:
            raise ValueError(f"Unsupported file format: {source.suffix}")
        
        click.echo(f"Loaded {len(samples)} samples from {source}")
        all_samples.extend(samples)
    
    return all_samples


def validate_sample(sample: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate a single sample for fine-tuning.
    
    Args:
        sample: Sample dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['nl_statement', 'atl_formula']
    
    for field in required_fields:
        if field not in sample:
            return False, f"Missing required field: {field}"
        if not sample[field] or not isinstance(sample[field], str):
            return False, f"Invalid value for field: {field}"
    
    # Check for reasonable lengths
    if len(sample['nl_statement']) < 10:
        return False, "NL statement too short"
    
    if len(sample['atl_formula']) < 3:
        return False, "ATL formula too short"
    
    return True, None


def sample_to_chat_format(
    sample: Dict[str, Any],
    config: FinetuneConfig,
) -> Dict[str, Any]:
    """
    Convert a sample to OpenAI chat fine-tuning format.
    
    Args:
        sample: Sample dictionary with nl_statement and atl_formula
        config: Fine-tuning configuration
        
    Returns:
        Dictionary in OpenAI chat format
    """
    # Build user message
    user_content = f"Translate the following natural language requirement to ATL:\n\n{sample['nl_statement']}"
    
    if config.include_domain and 'domain' in sample:
        user_content += f"\n\nDomain: {sample['domain']}"
    
    if config.include_agents and 'agents' in sample:
        agents = sample['agents']
        if isinstance(agents, list):
            user_content += f"\n\nAgents: {', '.join(agents)}"
    
    # Build messages
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample['atl_formula']},
    ]
    
    return {"messages": messages}


def filter_samples(
    samples: List[Dict[str, Any]],
    config: FinetuneConfig,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filter samples based on configuration criteria.
    
    Args:
        samples: List of sample dictionaries
        config: Fine-tuning configuration with filtering options
        
    Returns:
        Tuple of (filtered_samples, filter_stats)
    """
    filter_stats = {
        "total": len(samples),
        "filtered_confidence": 0,
        "filtered_status": 0,
        "filtered_domain": 0,
        "passed": 0,
    }
    
    filtered = []
    
    for sample in samples:
        # Filter by verification confidence
        confidence = sample.get('verification_confidence', 1.0)
        if confidence < config.min_confidence:
            filter_stats["filtered_confidence"] += 1
            continue
        
        # Filter by verification status
        status = sample.get('verification_status', 'verified')
        if config.verification_status_filter != 'all':
            if status != config.verification_status_filter:
                filter_stats["filtered_status"] += 1
                continue
        
        # Filter by domain
        if config.domain_filter:
            domain = sample.get('domain', '')
            if domain not in config.domain_filter:
                filter_stats["filtered_domain"] += 1
                continue
        
        filtered.append(sample)
        filter_stats["passed"] += 1
    
    return filtered, filter_stats


def prepare_training_data(
    samples: List[Dict[str, Any]],
    config: FinetuneConfig,
) -> tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """
    Prepare training and validation data.
    
    Args:
        samples: List of sample dictionaries
        config: Fine-tuning configuration
        
    Returns:
        Tuple of (training_data, validation_data, stats)
    """
    import random
    
    # Apply filters first
    filtered_samples, filter_stats = filter_samples(samples, config)
    
    valid_samples = []
    invalid_count = 0
    
    for sample in filtered_samples:
        is_valid, error = validate_sample(sample)
        if is_valid:
            valid_samples.append(sample)
        else:
            invalid_count += 1
    
    if not valid_samples:
        raise ValueError("No valid samples found for training")
    
    # Shuffle samples
    random.shuffle(valid_samples)
    
    # Split into training and validation
    val_size = int(len(valid_samples) * config.validation_split)
    val_size = max(1, val_size)  # At least 1 validation sample
    
    val_samples = valid_samples[:val_size]
    train_samples = valid_samples[val_size:]
    
    # Convert to chat format
    train_data = [sample_to_chat_format(s, config) for s in train_samples]
    val_data = [sample_to_chat_format(s, config) for s in val_samples]
    
    stats = {
        "total_samples": len(samples),
        "filtered_samples": filter_stats,
        "valid_samples": len(valid_samples),
        "invalid_samples": invalid_count,
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
    }
    
    return train_data, val_data, stats


def write_jsonl(data: List[Dict], filepath: Path) -> None:
    """Write data to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# =============================================================================
# OpenAI Fine-tuning API
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


def upload_training_file(client, filepath: Path, purpose: str = "fine-tune") -> str:
    """
    Upload a training file to OpenAI.
    
    Args:
        client: OpenAI client
        filepath: Path to the JSONL file
        purpose: Purpose of the file (fine-tune)
        
    Returns:
        File ID
    """
    with open(filepath, 'rb') as f:
        response = client.files.create(file=f, purpose=purpose)
    return response.id


def create_finetune_job(
    client,
    training_file_id: str,
    validation_file_id: Optional[str],
    config: FinetuneConfig,
) -> Dict[str, Any]:
    """
    Create a fine-tuning job.
    
    Args:
        client: OpenAI client
        training_file_id: ID of the training file
        validation_file_id: ID of the validation file (optional)
        config: Fine-tuning configuration
        
    Returns:
        Job details dictionary
    """
    hyperparameters = {}
    if config.n_epochs is not None:
        hyperparameters["n_epochs"] = config.n_epochs
    if config.batch_size is not None:
        hyperparameters["batch_size"] = config.batch_size
    if config.learning_rate_multiplier is not None:
        hyperparameters["learning_rate_multiplier"] = config.learning_rate_multiplier
    
    kwargs = {
        "training_file": training_file_id,
        "model": config.model,
    }
    
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    
    if hyperparameters:
        kwargs["hyperparameters"] = hyperparameters
    
    if config.suffix:
        kwargs["suffix"] = config.suffix
    
    if config.seed is not None:
        kwargs["seed"] = config.seed
    
    job = client.fine_tuning.jobs.create(**kwargs)
    
    return {
        "id": job.id,
        "model": job.model,
        "status": job.status,
        "created_at": job.created_at,
        "training_file": job.training_file,
        "validation_file": job.validation_file,
        "hyperparameters": job.hyperparameters.model_dump() if job.hyperparameters else None,
    }


def list_finetune_jobs(client, limit: int = 10) -> List[Dict[str, Any]]:
    """
    List fine-tuning jobs.
    
    Args:
        client: OpenAI client
        limit: Maximum number of jobs to return
        
    Returns:
        List of job details
    """
    jobs = client.fine_tuning.jobs.list(limit=limit)
    
    return [
        {
            "id": job.id,
            "model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "status": job.status,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
        }
        for job in jobs.data
    ]


def get_finetune_job(client, job_id: str) -> Dict[str, Any]:
    """
    Get details of a fine-tuning job.
    
    Args:
        client: OpenAI client
        job_id: Job ID
        
    Returns:
        Job details dictionary
    """
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    return {
        "id": job.id,
        "model": job.model,
        "fine_tuned_model": job.fine_tuned_model,
        "status": job.status,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "trained_tokens": job.trained_tokens,
        "error": job.error.model_dump() if job.error else None,
        "hyperparameters": job.hyperparameters.model_dump() if job.hyperparameters else None,
    }


def cancel_finetune_job(client, job_id: str) -> Dict[str, Any]:
    """
    Cancel a fine-tuning job.
    
    Args:
        client: OpenAI client
        job_id: Job ID
        
    Returns:
        Cancelled job details
    """
    job = client.fine_tuning.jobs.cancel(job_id)
    
    return {
        "id": job.id,
        "status": job.status,
    }


def get_finetune_events(client, job_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get events for a fine-tuning job.
    
    Args:
        client: OpenAI client
        job_id: Job ID
        limit: Maximum number of events
        
    Returns:
        List of events
    """
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit)
    
    return [
        {
            "created_at": event.created_at,
            "level": event.level,
            "message": event.message,
        }
        for event in events.data
    ]


# =============================================================================
# CLI Commands
# =============================================================================

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """NL2ATL Fine-tuning CLI for OpenAI models."""
    pass


@cli.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='YAML configuration file. If provided, other options override config values.',
)
@click.option(
    '--source', '-s',
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help='Source JSONL/JSON file(s) containing training data. Can be specified multiple times.',
)
@click.option(
    '--model', '-m',
    type=click.Choice(SUPPORTED_MODELS),
    default=None,
    help='Base model to fine-tune.',
)
@click.option(
    '--epochs', '-e',
    type=int,
    default=None,
    help='Number of training epochs (default: auto).',
)
@click.option(
    '--batch-size', '-b',
    type=int,
    default=None,
    help='Training batch size (default: auto).',
)
@click.option(
    '--learning-rate', '-l',
    type=float,
    default=None,
    help='Learning rate multiplier (default: auto).',
)
@click.option(
    '--suffix',
    type=str,
    default=None,
    help='Custom suffix for the fine-tuned model name.',
)
@click.option(
    '--validation-split',
    type=float,
    default=None,
    help='Fraction of data to use for validation (default: 0.1).',
)
@click.option(
    '--min-confidence',
    type=float,
    default=None,
    help='Minimum verification confidence threshold (0.0-1.0).',
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Seed for reproducible training.',
)
@click.option(
    '--include-domain/--no-domain',
    default=None,
    help='Include domain context in prompts.',
)
@click.option(
    '--include-agents/--no-agents',
    default=None,
    help='Include agents list in prompts.',
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Prepare data and validate but do not submit job.',
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default=None,
    help='Directory to save prepared training files.',
)
def train(
    config: Optional[Path],
    source: tuple[Path, ...],
    model: Optional[str],
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    suffix: Optional[str],
    validation_split: Optional[float],
    min_confidence: Optional[float],
    seed: Optional[int],
    include_domain: Optional[bool],
    include_agents: Optional[bool],
    dry_run: bool,
    output_dir: Optional[Path],
):
    """
    Create a fine-tuning job with the specified data sources.
    
    Examples:
    
        # Using config file
        python finetune.py train -c config/finetune_config.yaml
        
        # Single source
        python finetune.py train -s data/verified.jsonl
        
        # Multiple sources
        python finetune.py train -s data/verified_100.jsonl -s data/verified_20.jsonl
        
        # Custom configuration
        python finetune.py train -s data/verified.jsonl -m gpt-4o-2024-08-06 -e 3 --suffix nl2atl-v1
        
        # Override config file options
        python finetune.py train -c config/finetune_config.yaml --epochs 5 --suffix custom
    """
    # Load base config from file or create default
    if config:
        click.echo(f"Loading configuration from: {config}")
        finetune_config = FinetuneConfig.from_yaml(config)
    else:
        finetune_config = FinetuneConfig()
    
    # Override with CLI options if provided
    if source:
        finetune_config.sources = list(source)
    if model is not None:
        finetune_config.model = model
    if epochs is not None:
        finetune_config.n_epochs = epochs
    if batch_size is not None:
        finetune_config.batch_size = batch_size
    if learning_rate is not None:
        finetune_config.learning_rate_multiplier = learning_rate
    if suffix is not None:
        finetune_config.suffix = suffix
    if validation_split is not None:
        finetune_config.validation_split = validation_split
    if min_confidence is not None:
        finetune_config.min_confidence = min_confidence
    if seed is not None:
        finetune_config.seed = seed
    if include_domain is not None:
        finetune_config.include_domain = include_domain
    if include_agents is not None:
        finetune_config.include_agents = include_agents
    
    # Validate that we have sources
    if not finetune_config.sources:
        raise click.UsageError("No data sources specified. Use --source or provide a config file with sources.")
    
    click.echo(f"\n{'='*60}")
    click.echo("NL2ATL Fine-tuning")
    click.echo(f"{'='*60}\n")
    
    click.echo(f"Configuration:")
    click.echo(f"  Model: {finetune_config.model}")
    click.echo(f"  Sources: {len(finetune_config.sources)} file(s)")
    click.echo(f"  Min Confidence: {finetune_config.min_confidence}")
    click.echo(f"  Validation Split: {finetune_config.validation_split}")
    if finetune_config.seed is not None:
        click.echo(f"  Seed: {finetune_config.seed}")
    
    # Load samples
    click.echo("\nLoading samples...")
    samples = load_samples(finetune_config.sources)
    click.echo(f"Total samples loaded: {len(samples)}\n")
    
    # Prepare training data
    click.echo("Preparing training data...")
    train_data, val_data, stats = prepare_training_data(samples, finetune_config)
    
    click.echo(f"\nData Statistics:")
    click.echo(f"  Total samples: {stats['total_samples']}")
    if 'filtered_samples' in stats:
        fs = stats['filtered_samples']
        click.echo(f"  Filtered by confidence: {fs['filtered_confidence']}")
        click.echo(f"  Filtered by status: {fs['filtered_status']}")
        click.echo(f"  Filtered by domain: {fs['filtered_domain']}")
        click.echo(f"  Samples after filtering: {fs['passed']}")
    click.echo(f"  Valid samples: {stats['valid_samples']}")
    click.echo(f"  Invalid samples: {stats['invalid_samples']}")
    click.echo(f"  Training samples: {stats['training_samples']}")
    click.echo(f"  Validation samples: {stats['validation_samples']}")
    
    # Save prepared files if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        train_file = output_dir / "train.jsonl"
        val_file = output_dir / "validation.jsonl"
        write_jsonl(train_data, train_file)
        write_jsonl(val_data, val_file)
        click.echo(f"\nSaved training data to: {train_file}")
        click.echo(f"Saved validation data to: {val_file}")
    
    if dry_run:
        click.echo("\n[DRY RUN] Skipping job submission.")
        click.echo("\nSample training example:")
        click.echo(json.dumps(train_data[0], indent=2, ensure_ascii=False))
        return
    
    # Upload files and create job
    click.echo("\nConnecting to OpenAI...")
    client = get_openai_client()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Write temporary files
        train_file = tmpdir / "train.jsonl"
        val_file = tmpdir / "validation.jsonl"
        write_jsonl(train_data, train_file)
        write_jsonl(val_data, val_file)
        
        # Upload files
        click.echo("Uploading training file...")
        train_file_id = upload_training_file(client, train_file)
        click.echo(f"  Training file ID: {train_file_id}")
        
        click.echo("Uploading validation file...")
        val_file_id = upload_training_file(client, val_file)
        click.echo(f"  Validation file ID: {val_file_id}")
        
        # Create job
        click.echo("\nCreating fine-tuning job...")
        job = create_finetune_job(client, train_file_id, val_file_id, finetune_config)
        
        click.echo(f"\n{'='*60}")
        click.echo("Fine-tuning Job Created")
        click.echo(f"{'='*60}")
        click.echo(f"  Job ID: {job['id']}")
        click.echo(f"  Model: {job['model']}")
        click.echo(f"  Status: {job['status']}")
        click.echo(f"\nTo check status: python finetune.py status --job-id {job['id']}")


@cli.command()
@click.option(
    '--source', '-s',
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Source JSONL/JSON file(s) to validate.',
)
@click.option(
    '--include-domain/--no-domain',
    default=True,
    help='Include domain context in prompts.',
)
@click.option(
    '--include-agents/--no-agents',
    default=True,
    help='Include agents list in prompts.',
)
def validate(
    source: tuple[Path, ...],
    include_domain: bool,
    include_agents: bool,
):
    """
    Validate dataset files without submitting a fine-tuning job.
    
    This command checks that all samples have required fields and
    displays statistics about the dataset.
    """
    config = FinetuneConfig(
        sources=list(source),
        include_domain=include_domain,
        include_agents=include_agents,
    )
    
    click.echo(f"\n{'='*60}")
    click.echo("Dataset Validation")
    click.echo(f"{'='*60}\n")
    
    # Load samples
    samples = load_samples(config.sources)
    
    # Validate each sample
    valid_count = 0
    invalid_samples = []
    
    for i, sample in enumerate(samples):
        is_valid, error = validate_sample(sample)
        if is_valid:
            valid_count += 1
        else:
            invalid_samples.append((i, sample.get('id', 'unknown'), error))
    
    click.echo(f"\nValidation Results:")
    click.echo(f"  Total samples: {len(samples)}")
    click.echo(f"  Valid samples: {valid_count}")
    click.echo(f"  Invalid samples: {len(invalid_samples)}")
    
    if invalid_samples:
        click.echo(f"\nInvalid Samples (first 10):")
        for idx, sample_id, error in invalid_samples[:10]:
            click.echo(f"  [{idx}] {sample_id}: {error}")
    
    # Show sample formats
    if samples:
        click.echo("\nSample converted to chat format:")
        chat_sample = sample_to_chat_format(samples[0], config)
        click.echo(json.dumps(chat_sample, indent=2, ensure_ascii=False))
    
    # Estimate cost
    if valid_count > 0:
        avg_tokens = 200  # Rough estimate per sample
        total_tokens = valid_count * avg_tokens * 3  # 3 epochs default
        estimated_cost = (total_tokens / 1000) * 0.003  # gpt-4o-mini training cost
        click.echo(f"\nEstimated training cost (3 epochs, gpt-4o-mini): ~${estimated_cost:.2f}")


@cli.command('list')
@click.option(
    '--limit', '-n',
    type=int,
    default=10,
    help='Maximum number of jobs to list.',
)
def list_jobs(limit: int):
    """List recent fine-tuning jobs."""
    client = get_openai_client()
    jobs = list_finetune_jobs(client, limit)
    
    if not jobs:
        click.echo("No fine-tuning jobs found.")
        return
    
    click.echo(f"\n{'='*80}")
    click.echo(f"{'Job ID':<25} {'Model':<25} {'Status':<15} {'Fine-tuned Model'}")
    click.echo(f"{'='*80}")
    
    for job in jobs:
        fine_tuned = job['fine_tuned_model'] or '-'
        click.echo(f"{job['id']:<25} {job['model']:<25} {job['status']:<15} {fine_tuned}")


@cli.command()
@click.option(
    '--job-id', '-j',
    type=str,
    required=True,
    help='Fine-tuning job ID.',
)
@click.option(
    '--events/--no-events',
    default=True,
    help='Show recent events.',
)
@click.option(
    '--watch', '-w',
    is_flag=True,
    help='Watch job status until completion.',
)
def status(job_id: str, events: bool, watch: bool):
    """Check the status of a fine-tuning job."""
    client = get_openai_client()
    
    def show_status():
        job = get_finetune_job(client, job_id)
        
        click.echo(f"\n{'='*60}")
        click.echo(f"Fine-tuning Job: {job['id']}")
        click.echo(f"{'='*60}")
        click.echo(f"  Model: {job['model']}")
        click.echo(f"  Status: {job['status']}")
        click.echo(f"  Fine-tuned Model: {job['fine_tuned_model'] or 'N/A'}")
        click.echo(f"  Trained Tokens: {job['trained_tokens'] or 'N/A'}")
        
        if job['error']:
            click.echo(f"  Error: {job['error']}")
        
        if job['hyperparameters']:
            click.echo(f"  Hyperparameters: {json.dumps(job['hyperparameters'])}")
        
        if events:
            job_events = get_finetune_events(client, job_id)
            if job_events:
                click.echo(f"\nRecent Events:")
                for event in reversed(job_events[:10]):
                    ts = datetime.fromtimestamp(event['created_at']).strftime('%Y-%m-%d %H:%M:%S')
                    click.echo(f"  [{ts}] {event['level']}: {event['message']}")
        
        return job['status']
    
    if watch:
        terminal_states = {'succeeded', 'failed', 'cancelled'}
        while True:
            status = show_status()
            if status in terminal_states:
                break
            click.echo(f"\nRefreshing in 30 seconds... (Ctrl+C to stop)")
            time.sleep(30)
    else:
        show_status()


@cli.command()
@click.option(
    '--job-id', '-j',
    type=str,
    required=True,
    help='Fine-tuning job ID to cancel.',
)
@click.confirmation_option(prompt='Are you sure you want to cancel this job?')
def cancel(job_id: str):
    """Cancel a fine-tuning job."""
    client = get_openai_client()
    result = cancel_finetune_job(client, job_id)
    click.echo(f"Job {result['id']} cancelled. Status: {result['status']}")


@cli.command()
@click.option(
    '--model', '-m',
    type=str,
    required=True,
    help='Fine-tuned model ID to test.',
)
@click.option(
    '--statement', '-s',
    type=str,
    required=True,
    help='Natural language statement to translate.',
)
@click.option(
    '--domain', '-d',
    type=str,
    default=None,
    help='Domain context.',
)
@click.option(
    '--agents', '-a',
    type=str,
    default=None,
    help='Comma-separated list of agents.',
)
def test(model: str, statement: str, domain: Optional[str], agents: Optional[str]):
    """
    Test a fine-tuned model with a sample statement.
    
    Example:
        python finetune.py test -m ft:gpt-4o-mini:... -s "The robot can eventually reach the goal"
    """
    client = get_openai_client()
    
    config = FinetuneConfig()
    
    # Build user message
    user_content = f"Translate the following natural language requirement to ATL:\n\n{statement}"
    
    if domain:
        user_content += f"\n\nDomain: {domain}"
    
    if agents:
        user_content += f"\n\nAgents: {agents}"
    
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    click.echo(f"\nTesting model: {model}")
    click.echo(f"Statement: {statement}")
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    
    result = response.choices[0].message.content.strip()
    click.echo(f"\nATL Formula: {result}")
    click.echo(f"\nTokens used: {response.usage.total_tokens}")


@cli.command()
def models():
    """List supported base models for fine-tuning."""
    click.echo("\nSupported base models for fine-tuning:")
    click.echo("="*50)
    for model in SUPPORTED_MODELS:
        click.echo(f"  • {model}")
    click.echo("\nNote: Model availability may vary based on your OpenAI account tier.")


if __name__ == "__main__":
    cli()
