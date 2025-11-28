"""
Unified Sample Storage & Parsing Module
========================================

This module provides:
1. Multi-format sample parsing (handles all team member formats)
2. Syntax normalization (Unicode ↔ ASCII)
3. Domain classification for diversity analysis
4. Persistent JSON storage with deduplication
5. Statistics and coverage reporting

Sample Formats Detected:
------------------------
- Format 1 (sample 1): "NL statement" = <<Agent>> ATL_formula
- Format 2 (sample 2): "NL statement" -> ⟨⟨Agent⟩⟩ ATL_formula
- Format 3 (sample 3): NL statement (newline) <<agent>> ATL_formula

Usage:
------
    from sample_store import SampleStore, load_all_samples
    
    store = load_all_samples(Path("samples"), Path("data/unified_samples.json"))
    print(store.get_stats())
    print(store.get_domain_coverage())
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any


# =============================================================================
# Domain Classification
# =============================================================================

class Domain(str, Enum):
    """Domain categories for sample classification."""
    AUTONOMOUS_SYSTEMS = "autonomous_systems"
    DISTRIBUTED_SYSTEMS = "distributed_systems"
    SAFETY_CRITICAL = "safety_critical"
    ROBOTICS = "robotics"
    AEROSPACE = "aerospace"
    INDUSTRIAL_CONTROL = "industrial_control"
    TRANSPORTATION = "transportation"
    ACCESS_CONTROL = "access_control"
    DATA_MANAGEMENT = "data_management"
    NETWORKING = "networking"
    GENERIC = "generic"


# Domain classification keywords
DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.AUTONOMOUS_SYSTEMS: [
        "drone", "autopilot", "navigation", "autonomous", "self-driving",
        "uav", "robot", "controller"
    ],
    Domain.DISTRIBUTED_SYSTEMS: [
        "replica", "consensus", "cluster", "distributed", "node", "leader",
        "replication", "orchestrator", "scheduler", "membership", "checkpoint"
    ],
    Domain.SAFETY_CRITICAL: [
        "safety", "hazard", "emergency", "alarm", "monitor", "unsafe",
        "collision", "fail", "error", "critical"
    ],
    Domain.ROBOTICS: [
        "robot", "arm", "gripper", "actuator", "manipulator", "servo"
    ],
    Domain.AEROSPACE: [
        "aircraft", "flight", "turbulence", "stabilizer", "altitude",
        "landing", "takeoff"
    ],
    Domain.INDUSTRIAL_CONTROL: [
        "cooling", "power", "temperature", "controller", "sensor",
        "valve", "pump", "motor", "actuator"
    ],
    Domain.TRANSPORTATION: [
        "vehicle", "braking", "speed", "traffic", "lane", "intersection"
    ],
    Domain.ACCESS_CONTROL: [
        "authentication", "authorization", "permission", "access", "user",
        "session", "token", "login", "credential", "identity", "privilege"
    ],
    Domain.DATA_MANAGEMENT: [
        "file", "data", "database", "record", "log", "archive", "backup",
        "transaction", "commit", "rollback", "checksum"
    ],
    Domain.NETWORKING: [
        "connection", "firewall", "handshake", "message", "telemetry",
        "network", "socket", "request", "response"
    ],
}


# =============================================================================
# Syntax Normalization
# =============================================================================

class SyntaxNormalizer:
    """Normalize ATL syntax variations to a canonical form."""
    
    # Unicode to ASCII mappings
    UNICODE_TO_ASCII: dict[str, str] = {
        "⇒": "->",
        "→": "->",
        "=>": "->",
        "¬": "!",
        "~": "!",
        "∧": "&",
        "/\\": "&",
        "∨": "|",
        "\\/": "|",
        "⟨⟨": "<<",
        "⟩⟩": ">>",
        "◇": "F",
        "□": "G",
        "○": "X",
    }
    
    # ASCII to Unicode mappings (for pretty printing)
    ASCII_TO_UNICODE: dict[str, str] = {
        "->": "→",
        "!": "¬",
        "&": "∧",
        "|": "∨",
        "<<": "⟨⟨",
        ">>": "⟩⟩",
    }
    
    @classmethod
    def to_ascii(cls, formula: str) -> str:
        """Convert formula to ASCII-only representation."""
        result = formula
        for unicode_op, ascii_op in cls.UNICODE_TO_ASCII.items():
            result = result.replace(unicode_op, ascii_op)
        return result.strip()
    
    @classmethod
    def to_unicode(cls, formula: str) -> str:
        """Convert formula to Unicode representation."""
        result = formula
        for ascii_op, unicode_op in cls.ASCII_TO_UNICODE.items():
            result = result.replace(ascii_op, unicode_op)
        return result.strip()
    
    @classmethod
    def extract_agents(cls, formula: str) -> list[str]:
        """Extract coalition agents from <<...>> or ⟨⟨...⟩⟩ notation."""
        # Normalize to ASCII first
        normalized = cls.to_ascii(formula)
        pattern = r'<<([^>]+)>>'
        matches = re.findall(pattern, normalized)
        agents = []
        for match in matches:
            agents.extend([a.strip() for a in match.split(',')])
        return list(set(agents))
    
    @classmethod
    def extract_operators(cls, formula: str) -> list[str]:
        """Extract temporal operators used in formula."""
        normalized = cls.to_ascii(formula)
        operators = []
        # Match standalone temporal operators (not part of identifiers)
        for op in ['G', 'F', 'X', 'U']:
            # Look for operator followed by space, paren, or end
            if re.search(rf'\b{op}\b|{op}\s*\(|{op}\s+', normalized):
                operators.append(op)
        if '->' in normalized:
            operators.append('->')
        return operators
    
    @classmethod
    def extract_atoms(cls, formula: str) -> list[str]:
        """Extract atomic propositions from formula."""
        normalized = cls.to_ascii(formula)
        # Remove coalition operators
        normalized = re.sub(r'<<[^>]+>>', '', normalized)
        # Remove temporal operators and boolean connectives
        normalized = re.sub(r'\b(G|F|X|U)\b', ' ', normalized)
        normalized = re.sub(r'[!&|()>\-<]', ' ', normalized)
        # Extract identifiers
        atoms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', normalized)
        # Filter out common words
        exclude = {'true', 'false', 'and', 'or', 'not', 'implies'}
        return list(set(a for a in atoms if a.lower() not in exclude))


# =============================================================================
# Sample Data Model
# =============================================================================

@dataclass
class ATLSample:
    """
    Unified representation of an NL→ATL sample.
    
    This schema supports both:
    - Parsed samples from existing files (source_file set)
    - Generated samples from templates (generation dict set)
    """
    # Core fields (required)
    id: str
    nl_statement: str
    atl_formula: str  # Stored in ASCII format for consistency
    atl_unicode: str  # Unicode pretty-print version
    
    # Classification
    domain: str = "generic"
    
    # Provenance - one of these should be set
    source_file: Optional[str] = None  # For parsed samples
    template_id: Optional[str] = None  # For generated samples
    
    # Extracted components
    agents: list[str] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)
    atoms: list[str] = field(default_factory=list)
    
    # Generation context (for generated samples)
    generation: Optional[dict] = None  # {provider, temperature, paraphrase_of}
    
    # Verification status
    syntax_valid: Optional[bool] = None
    semantic_verified: Optional[bool] = None
    confidence: Optional[float] = None
    verification_notes: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ATLSample':
        """Create from dictionary."""
        # Handle both old and new field names for compatibility
        if 'nl_text' in data and 'nl_statement' not in data:
            data['nl_statement'] = data.pop('nl_text')
        if 'source' in data and 'source_file' not in data:
            data['source_file'] = data.pop('source')
        # Filter to known fields
        known_fields = {
            'id', 'nl_statement', 'atl_formula', 'atl_unicode', 'domain',
            'source_file', 'template_id', 'agents', 'operators', 'atoms',
            'generation', 'syntax_valid', 'semantic_verified', 'confidence',
            'verification_notes', 'created_at'
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
    
    def get_complexity_score(self) -> int:
        """Calculate formula complexity based on operators and nesting."""
        score = len(self.operators) + len(self.atoms)
        # Add weight for nested patterns
        if '->' in self.atl_formula and 'F' in self.atl_formula:
            score += 2  # Response pattern
        if 'G' in self.atl_formula and 'F' in self.atl_formula:
            score += 1  # Nested temporal
        return score


# =============================================================================
# Multi-Format Sample Parser
# =============================================================================

class SampleParser:
    """
    Parse sample files in multiple formats into ATLSample objects.
    
    Supported formats:
    - Format 1: "NL statement" = <<Agent>> ATL_formula
    - Format 2: "NL statement" -> ⟨⟨Agent⟩⟩ ATL_formula  
    - Format 3: NL statement (newline) <<agent>> ATL_formula
    """
    
    @staticmethod
    def generate_id(nl: str, atl: str) -> str:
        """Generate unique ID from content hash."""
        content = f"{nl.strip().lower()}|{atl.strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    @staticmethod
    def classify_domain(nl: str, atl: str) -> Domain:
        """Classify sample into a domain based on keywords."""
        text = (nl + " " + atl).lower()
        scores: dict[Domain, int] = {d: 0 for d in Domain}
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    scores[domain] += 1
        
        best_domain = max(scores, key=lambda d: scores[d])
        return best_domain if scores[best_domain] > 0 else Domain.GENERIC
    
    @classmethod
    def parse_format1(cls, content: str, source_file: str) -> list[ATLSample]:
        """
        Parse Format 1: "NL statement" (newline) <<Agent>> ATL_formula
        Used in sample 1.txt and sample 2.txt - NL in quotes, ATL on next line
        """
        samples = []
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if line is a quoted NL statement
            if line.startswith('"') and line.endswith('"'):
                nl = line[1:-1].strip()  # Remove quotes
                
                # Look for ATL formula on next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('<<') or next_line.startswith('⟨⟨'):
                        atl = next_line
                        
                        atl_ascii = SyntaxNormalizer.to_ascii(atl)
                        atl_unicode = SyntaxNormalizer.to_unicode(atl_ascii)
                        
                        sample = ATLSample(
                            id=cls.generate_id(nl, atl_ascii),
                            nl_statement=nl,
                            atl_formula=atl_ascii,
                            atl_unicode=atl_unicode,
                            domain=cls.classify_domain(nl, atl).value,
                            source_file=source_file,
                            agents=SyntaxNormalizer.extract_agents(atl),
                            operators=SyntaxNormalizer.extract_operators(atl),
                            atoms=SyntaxNormalizer.extract_atoms(atl),
                        )
                        samples.append(sample)
                        i += 2
                        continue
            
            i += 1
        
        return samples
    
    @classmethod
    def parse_format2(cls, content: str, source_file: str) -> list[ATLSample]:
        """
        Parse Format 2: "NL statement" -> ⟨⟨Agent⟩⟩ ATL_formula (same line)
        Legacy format - kept for backwards compatibility
        """
        samples = []
        # Pattern: "..." -> ATL_formula
        pattern = r'"([^"]+)"\s*->\s*(.+?)(?=\n"|\n\n|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for nl, atl in matches:
            nl = nl.strip()
            atl = atl.strip()
            if not nl or not atl:
                continue
                
            atl_ascii = SyntaxNormalizer.to_ascii(atl)
            atl_unicode = SyntaxNormalizer.to_unicode(atl_ascii)
            
            sample = ATLSample(
                id=cls.generate_id(nl, atl_ascii),
                nl_statement=nl,
                atl_formula=atl_ascii,
                atl_unicode=atl_unicode,
                domain=cls.classify_domain(nl, atl).value,
                source_file=source_file,
                agents=SyntaxNormalizer.extract_agents(atl),
                operators=SyntaxNormalizer.extract_operators(atl),
                atoms=SyntaxNormalizer.extract_atoms(atl),
            )
            samples.append(sample)
        
        return samples
    
    @classmethod
    def parse_format3(cls, content: str, source_file: str) -> list[ATLSample]:
        """
        Parse Format 3: NL statement (newline) <<agent>> ATL_formula
        Used in sample 3.txt - no quotes around NL
        """
        samples = []
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comment lines
            if not line or line.startswith('#') or line.startswith('❌'):
                i += 1
                continue
            
            # Check if this line starts with << (ATL formula) - skip
            if line.startswith('<<') or line.startswith('⟨⟨'):
                i += 1
                continue
            
            # Check if next line is an ATL formula
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('<<') or next_line.startswith('⟨⟨'):
                    nl = line
                    atl = next_line
                    
                    # Skip interpretation lines
                    if nl.startswith('Interpreting'):
                        i += 1
                        continue
                    
                    atl_ascii = SyntaxNormalizer.to_ascii(atl)
                    atl_unicode = SyntaxNormalizer.to_unicode(atl_ascii)
                    
                    sample = ATLSample(
                        id=cls.generate_id(nl, atl_ascii),
                        nl_statement=nl,
                        atl_formula=atl_ascii,
                        atl_unicode=atl_unicode,
                        domain=cls.classify_domain(nl, atl).value,
                        source_file=source_file,
                        agents=SyntaxNormalizer.extract_agents(atl),
                        operators=SyntaxNormalizer.extract_operators(atl),
                        atoms=SyntaxNormalizer.extract_atoms(atl),
                    )
                    samples.append(sample)
                    i += 2
                    continue
            
            i += 1
        
        return samples
    
    @classmethod
    def detect_format(cls, content: str) -> int:
        """Detect which format the content uses."""
        # Check for quoted NL followed by ATL on next line (Format 1 - new style)
        if re.search(r'"[^"]+"\s*\n\s*<<', content) or re.search(r'"[^"]+"\s*\n\s*⟨⟨', content):
            return 1
        # Format 2: "..." -> ATL on same line (legacy)
        if re.search(r'"[^"]+"\s*->\s*[⟨<]', content):
            return 2
        # Format 3: No quotes, line-by-line
        if re.search(r'^[A-Z][^"]+\n\s*<<', content, re.MULTILINE):
            return 3
        # Default to format 3 (most permissive)
        return 3
    
    @classmethod
    def parse_file(cls, filepath: Path) -> list[ATLSample]:
        """Parse a sample file, auto-detecting format."""
        content = filepath.read_text(encoding='utf-8')
        source_file = filepath.name
        
        format_num = cls.detect_format(content)
        
        if format_num == 1:
            return cls.parse_format1(content, source_file)
        elif format_num == 2:
            return cls.parse_format2(content, source_file)
        else:
            return cls.parse_format3(content, source_file)


# =============================================================================
# Sample Store with Persistence
# =============================================================================

@dataclass
class StoreStats:
    """Statistics about the sample store."""
    total_samples: int
    unique_domains: int
    domain_distribution: dict[str, int]
    verified_count: int
    syntax_valid_count: int
    unique_agents: int
    unique_atoms: int
    avg_complexity: float
    source_file_distribution: dict[str, int]
    operator_distribution: dict[str, int]
    
    def to_dict(self) -> dict:
        return asdict(self)


class SampleStore:
    """
    Persistent storage for unified NL-ATL samples.
    
    Features:
    - Automatic deduplication via content hashing
    - Domain-based retrieval
    - Statistics and coverage reporting
    - JSON persistence
    """
    
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.samples: dict[str, ATLSample] = {}
        self._load()
    
    def _load(self) -> None:
        """Load samples from JSON file."""
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding='utf-8'))
                for sample_dict in data.get('samples', []):
                    sample = ATLSample.from_dict(sample_dict)
                    self.samples[sample.id] = sample
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load store from {self.store_path}: {e}")
    
    def save(self) -> None:
        """Save samples to JSON file."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'samples': [s.to_dict() for s in self.samples.values()],
            'stats': self.get_stats().to_dict()
        }
        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    def add(self, sample: ATLSample) -> bool:
        """Add sample if unique. Returns True if added."""
        if sample.id not in self.samples:
            self.samples[sample.id] = sample
            return True
        return False
    
    def add_batch(self, samples: list[ATLSample]) -> tuple[int, int]:
        """Add multiple samples. Returns (added, duplicates)."""
        added = 0
        duplicates = 0
        for sample in samples:
            if self.add(sample):
                added += 1
            else:
                duplicates += 1
        return added, duplicates
    
    def get_by_id(self, sample_id: str) -> Optional[ATLSample]:
        """Get a sample by ID."""
        return self.samples.get(sample_id)
    
    def get_by_domain(self, domain: str | Domain) -> list[ATLSample]:
        """Get all samples for a specific domain."""
        domain_str = domain.value if isinstance(domain, Domain) else domain
        return [s for s in self.samples.values() if s.domain == domain_str]
    
    def get_all(self) -> list[ATLSample]:
        """Get all samples."""
        return list(self.samples.values())
    
    def get_unverified(self) -> list[ATLSample]:
        """Get samples that haven't been verified."""
        return [s for s in self.samples.values() if s.syntax_valid is None]
    
    def update_sample(self, sample: ATLSample) -> None:
        """Update an existing sample."""
        if sample.id in self.samples:
            self.samples[sample.id] = sample
    
    def get_stats(self) -> StoreStats:
        """Calculate statistics about the store."""
        samples = list(self.samples.values())
        
        if not samples:
            return StoreStats(
                total_samples=0,
                unique_domains=0,
                domain_distribution={},
                verified_count=0,
                syntax_valid_count=0,
                unique_agents=0,
                unique_atoms=0,
                avg_complexity=0.0,
                source_file_distribution={},
                operator_distribution={},
            )
        
        # Domain distribution
        domain_dist: dict[str, int] = {}
        for s in samples:
            domain_dist[s.domain] = domain_dist.get(s.domain, 0) + 1
        
        # Source file distribution
        source_dist: dict[str, int] = {}
        for s in samples:
            source_dist[s.source_file] = source_dist.get(s.source_file, 0) + 1
        
        # Operator distribution
        op_dist: dict[str, int] = {}
        for s in samples:
            for op in s.operators:
                op_dist[op] = op_dist.get(op, 0) + 1
        
        # Unique agents and atoms
        all_agents = set(a for s in samples for a in s.agents)
        all_atoms = set(a for s in samples for a in s.atoms)
        
        # Complexity
        complexities = [s.get_complexity_score() for s in samples]
        avg_complexity = sum(complexities) / len(complexities)
        
        return StoreStats(
            total_samples=len(samples),
            unique_domains=len(domain_dist),
            domain_distribution=domain_dist,
            verified_count=sum(1 for s in samples if s.semantic_verified is True),
            syntax_valid_count=sum(1 for s in samples if s.syntax_valid is True),
            unique_agents=len(all_agents),
            unique_atoms=len(all_atoms),
            avg_complexity=round(avg_complexity, 2),
            source_file_distribution=source_dist,
            operator_distribution=op_dist,
        )
    
    def get_domain_coverage_report(self) -> str:
        """Generate a domain coverage report."""
        stats = self.get_stats()
        
        lines = [
            "=" * 50,
            "DOMAIN COVERAGE REPORT",
            "=" * 50,
            f"Total Samples: {stats.total_samples}",
            f"Unique Domains: {stats.unique_domains}",
            "",
            "Domain Distribution:",
        ]
        
        # Sort by count descending
        sorted_domains = sorted(
            stats.domain_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for domain, count in sorted_domains:
            pct = (count / stats.total_samples * 100) if stats.total_samples > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  {domain:25} {bar} {count:3} ({pct:5.1f}%)")
        
        lines.extend([
            "",
            "Source File Distribution:",
        ])
        
        for source, count in stats.source_file_distribution.items():
            lines.append(f"  {source:25} {count:3} samples")
        
        lines.extend([
            "",
            f"Unique Agents: {stats.unique_agents}",
            f"Unique Atoms: {stats.unique_atoms}",
            f"Avg Complexity: {stats.avg_complexity}",
            "",
            f"Verified (Semantic): {stats.verified_count}",
            f"Syntax Valid: {stats.syntax_valid_count}",
            "=" * 50,
        ])
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_all_samples(samples_dir: Path, store_path: Path) -> SampleStore:
    """
    Load all sample files from a directory into a unified store.
    
    Args:
        samples_dir: Directory containing sample .txt files
        store_path: Path for persistent JSON storage
        
    Returns:
        SampleStore with all samples loaded
    """
    store = SampleStore(store_path)
    
    total_added = 0
    total_duplicates = 0
    
    for sample_file in sorted(samples_dir.glob("*.txt")):
        try:
            samples = SampleParser.parse_file(sample_file)
            added, duplicates = store.add_batch(samples)
            total_added += added
            total_duplicates += duplicates
            print(f"Parsed {sample_file.name}: {added} new, {duplicates} duplicates")
        except Exception as e:
            print(f"Error parsing {sample_file}: {e}")
    
    store.save()
    print(f"\nTotal: {total_added} samples added, {total_duplicates} duplicates skipped")
    
    return store


# =============================================================================
# CLI for Sample Store
# =============================================================================

if __name__ == "__main__":
    import sys
    
    samples_dir = Path("samples")
    store_path = Path("data/unified_samples.json")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "load":
            store = load_all_samples(samples_dir, store_path)
            print("\n" + store.get_domain_coverage_report())
            
        elif command == "stats":
            store = SampleStore(store_path)
            print(store.get_domain_coverage_report())
            
        elif command == "list":
            store = SampleStore(store_path)
            for sample in store.get_all():
                print(f"[{sample.id}] {sample.domain}")
                print(f"  NL: {sample.nl_statement[:60]}...")
                print(f"  ATL: {sample.atl_formula}")
                print()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python sample_store.py [load|stats|list]")
    else:
        # Default: load samples
        store = load_all_samples(samples_dir, store_path)
        print("\n" + store.get_domain_coverage_report())
