"""
Metrics & Monitoring Module
============================

Provides quantifiable measurements for assessing project approach:
1. Quality Metrics - Syntax validity, semantic accuracy, cross-model agreement
2. Diversity Metrics - Domain coverage, formula complexity distribution
3. Performance Metrics - Latency, throughput, cost estimation
4. Progress Tracking - Time-series data for trend analysis

Usage:
------
    from metrics import MetricsCollector, MetricsDashboard
    
    collector = MetricsCollector()
    collector.record_verification(result)
    collector.record_generation(sample, latency_ms, tokens_used)
    
    dashboard = MetricsDashboard(collector)
    dashboard.print_summary()
    dashboard.export_csv("metrics.csv")
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class QualityMetrics:
    """Quality-related metrics."""
    total_samples: int = 0
    syntax_valid_count: int = 0
    syntax_valid_rate: float = 0.0
    semantic_verified_count: int = 0
    semantic_verified_rate: float = 0.0
    semantic_rejected_count: int = 0
    needs_review_count: int = 0
    error_count: int = 0
    
    # Cross-model metrics
    cross_model_agreement_count: int = 0
    cross_model_agreement_rate: float = 0.0
    
    # Confidence distribution
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    confidence_std: float = 0.0


@dataclass
class DiversityMetrics:
    """Diversity and coverage metrics."""
    unique_domains: int = 0
    domain_distribution: dict[str, int] = field(default_factory=dict)
    domain_entropy: float = 0.0  # Higher = more diverse
    
    unique_agents: int = 0
    unique_atoms: int = 0
    unique_operators: int = 0
    
    # Complexity distribution
    avg_complexity: float = 0.0
    complexity_distribution: dict[str, int] = field(default_factory=dict)  # "low", "medium", "high"
    
    # Source distribution
    source_file_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance and efficiency metrics."""
    total_time_ms: float = 0.0
    avg_time_per_sample_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    
    # Token usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_tokens_per_sample: float = 0.0
    
    # Cost estimation (based on typical pricing)
    estimated_cost_usd: float = 0.0
    
    # Throughput
    samples_per_minute: float = 0.0


@dataclass
class ProgressMetrics:
    """Progress tracking metrics."""
    timestamp: str = ""
    samples_processed: int = 0
    samples_remaining: int = 0
    completion_percentage: float = 0.0
    
    # Trend data
    verification_rate_trend: list[float] = field(default_factory=list)
    avg_confidence_trend: list[float] = field(default_factory=list)


@dataclass
class MetricsSnapshot:
    """Complete snapshot of all metrics at a point in time."""
    snapshot_id: str
    timestamp: str
    quality: QualityMetrics
    diversity: DiversityMetrics
    performance: PerformanceMetrics
    progress: ProgressMetrics
    
    def to_dict(self) -> dict:
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp,
            'quality': asdict(self.quality),
            'diversity': asdict(self.diversity),
            'performance': asdict(self.performance),
            'progress': asdict(self.progress),
        }
    
    def save(self, path: Path) -> None:
        """Save snapshot to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Metrics Collector
# =============================================================================

@dataclass
class SampleMetric:
    """Metrics for a single sample."""
    sample_id: str
    domain: str
    syntax_valid: bool
    semantic_verified: Optional[bool]
    confidence: float
    cross_model_agreement: Optional[bool]
    complexity: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    timestamp: str


class MetricsCollector:
    """
    Collects and aggregates metrics from pipeline runs.
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or Path("data/metrics_history.json")
        self.sample_metrics: list[SampleMetric] = []
        self.snapshots: list[MetricsSnapshot] = []
        self._load()
    
    def _load(self) -> None:
        """Load historical metrics."""
        if self.persistence_path.exists():
            try:
                data = json.loads(self.persistence_path.read_text())
                # Load sample metrics
                for sm in data.get('sample_metrics', []):
                    self.sample_metrics.append(SampleMetric(**sm))
            except (json.JSONDecodeError, KeyError):
                pass
    
    def save(self) -> None:
        """Persist metrics to disk."""
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'sample_metrics': [asdict(sm) for sm in self.sample_metrics],
            'snapshots': [s.to_dict() for s in self.snapshots],
        }
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_verification(
        self,
        sample_id: str,
        domain: str,
        syntax_valid: bool,
        semantic_verified: Optional[bool],
        confidence: float,
        cross_model_agreement: Optional[bool],
        complexity: int,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record metrics for a verified sample."""
        metric = SampleMetric(
            sample_id=sample_id,
            domain=domain,
            syntax_valid=syntax_valid,
            semantic_verified=semantic_verified,
            confidence=confidence,
            cross_model_agreement=cross_model_agreement,
            complexity=complexity,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timestamp=datetime.now().isoformat(),
        )
        self.sample_metrics.append(metric)
    
    def record_from_report(self, report: 'VerificationReport') -> None:
        """Record metrics from a verification report."""
        for result in report.results:
            semantic_verified = None
            confidence = 0.0
            
            if result.semantic_check:
                semantic_verified = result.semantic_check.matches
                confidence = result.semantic_check.confidence
            
            # Estimate complexity from formula
            complexity = len(result.original_atl.split())
            
            self.record_verification(
                sample_id=result.sample_id,
                domain="unknown",  # Would need sample info
                syntax_valid=result.syntax_check.valid,
                semantic_verified=semantic_verified,
                confidence=confidence,
                cross_model_agreement=result.cross_model_agreement,
                complexity=complexity,
                latency_ms=result.verification_time_ms,
            )
    
    def compute_quality_metrics(self) -> QualityMetrics:
        """Compute quality metrics from collected data."""
        if not self.sample_metrics:
            return QualityMetrics()
        
        total = len(self.sample_metrics)
        syntax_valid = sum(1 for m in self.sample_metrics if m.syntax_valid)
        semantic_verified = sum(1 for m in self.sample_metrics if m.semantic_verified is True)
        semantic_rejected = sum(1 for m in self.sample_metrics if m.semantic_verified is False)
        cross_agree = sum(1 for m in self.sample_metrics if m.cross_model_agreement is True)
        
        confidences = [m.confidence for m in self.sample_metrics if m.confidence > 0]
        
        return QualityMetrics(
            total_samples=total,
            syntax_valid_count=syntax_valid,
            syntax_valid_rate=syntax_valid / total if total > 0 else 0,
            semantic_verified_count=semantic_verified,
            semantic_verified_rate=semantic_verified / total if total > 0 else 0,
            semantic_rejected_count=semantic_rejected,
            needs_review_count=total - semantic_verified - semantic_rejected,
            error_count=0,
            cross_model_agreement_count=cross_agree,
            cross_model_agreement_rate=cross_agree / total if total > 0 else 0,
            avg_confidence=statistics.mean(confidences) if confidences else 0,
            min_confidence=min(confidences) if confidences else 0,
            max_confidence=max(confidences) if confidences else 0,
            confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0,
        )
    
    def compute_diversity_metrics(self) -> DiversityMetrics:
        """Compute diversity metrics from collected data."""
        if not self.sample_metrics:
            return DiversityMetrics()
        
        # Domain distribution
        domain_counts: dict[str, int] = defaultdict(int)
        for m in self.sample_metrics:
            domain_counts[m.domain] += 1
        
        # Entropy calculation
        import math
        total = len(self.sample_metrics)
        entropy = 0.0
        for count in domain_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Complexity distribution
        complexities = [m.complexity for m in self.sample_metrics]
        avg_complexity = statistics.mean(complexities) if complexities else 0
        
        complexity_dist = {"low": 0, "medium": 0, "high": 0}
        for c in complexities:
            if c <= 3:
                complexity_dist["low"] += 1
            elif c <= 6:
                complexity_dist["medium"] += 1
            else:
                complexity_dist["high"] += 1
        
        return DiversityMetrics(
            unique_domains=len(domain_counts),
            domain_distribution=dict(domain_counts),
            domain_entropy=round(entropy, 3),
            avg_complexity=round(avg_complexity, 2),
            complexity_distribution=complexity_dist,
        )
    
    def compute_performance_metrics(self) -> PerformanceMetrics:
        """Compute performance metrics from collected data."""
        if not self.sample_metrics:
            return PerformanceMetrics()
        
        latencies = [m.latency_ms for m in self.sample_metrics]
        total_time = sum(latencies)
        
        prompt_tokens = sum(m.prompt_tokens for m in self.sample_metrics)
        completion_tokens = sum(m.completion_tokens for m in self.sample_metrics)
        total_tokens = prompt_tokens + completion_tokens
        
        # Rough cost estimation (GPT-4 pricing as example)
        # $0.03/1K prompt, $0.06/1K completion
        cost = (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)
        
        # Throughput
        if total_time > 0:
            samples_per_min = len(self.sample_metrics) / (total_time / 60000)
        else:
            samples_per_min = 0
        
        return PerformanceMetrics(
            total_time_ms=total_time,
            avg_time_per_sample_ms=statistics.mean(latencies) if latencies else 0,
            min_time_ms=min(latencies) if latencies else 0,
            max_time_ms=max(latencies) if latencies else 0,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            avg_tokens_per_sample=total_tokens / len(self.sample_metrics) if self.sample_metrics else 0,
            estimated_cost_usd=round(cost, 4),
            samples_per_minute=round(samples_per_min, 2),
        )
    
    def take_snapshot(self, snapshot_id: Optional[str] = None) -> MetricsSnapshot:
        """Take a snapshot of current metrics."""
        import uuid
        
        if snapshot_id is None:
            snapshot_id = str(uuid.uuid4())[:8]
        
        snapshot = MetricsSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now().isoformat(),
            quality=self.compute_quality_metrics(),
            diversity=self.compute_diversity_metrics(),
            performance=self.compute_performance_metrics(),
            progress=ProgressMetrics(
                timestamp=datetime.now().isoformat(),
                samples_processed=len(self.sample_metrics),
            ),
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def clear(self) -> None:
        """Clear all collected metrics."""
        self.sample_metrics = []
        self.snapshots = []


# =============================================================================
# Metrics Dashboard
# =============================================================================

class MetricsDashboard:
    """
    Interactive dashboard for viewing and exporting metrics.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of all metrics."""
        quality = self.collector.compute_quality_metrics()
        diversity = self.collector.compute_diversity_metrics()
        performance = self.collector.compute_performance_metrics()
        
        print("=" * 70)
        print("                    METRICS DASHBOARD")
        print("=" * 70)
        
        # Quality Section
        print("\n📊 QUALITY METRICS")
        print("-" * 50)
        print(f"  Total Samples:           {quality.total_samples}")
        print(f"  Syntax Valid:            {quality.syntax_valid_count} ({quality.syntax_valid_rate*100:.1f}%)")
        print(f"  Semantic Verified:       {quality.semantic_verified_count} ({quality.semantic_verified_rate*100:.1f}%)")
        print(f"  Semantic Rejected:       {quality.semantic_rejected_count}")
        print(f"  Needs Review:            {quality.needs_review_count}")
        print(f"  Cross-Model Agreement:   {quality.cross_model_agreement_count} ({quality.cross_model_agreement_rate*100:.1f}%)")
        print()
        print(f"  Confidence - Avg: {quality.avg_confidence:.2f}, Min: {quality.min_confidence:.2f}, Max: {quality.max_confidence:.2f}")
        
        # Diversity Section
        print("\n🌈 DIVERSITY METRICS")
        print("-" * 50)
        print(f"  Unique Domains:          {diversity.unique_domains}")
        print(f"  Domain Entropy:          {diversity.domain_entropy:.3f} (higher = more diverse)")
        print(f"  Average Complexity:      {diversity.avg_complexity}")
        
        if diversity.domain_distribution:
            print("\n  Domain Distribution:")
            sorted_domains = sorted(
                diversity.domain_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )
            max_count = max(diversity.domain_distribution.values()) if diversity.domain_distribution else 1
            for domain, count in sorted_domains[:5]:  # Top 5
                bar_len = int(count / max_count * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"    {domain:25} {bar} {count}")
        
        if diversity.complexity_distribution:
            print("\n  Complexity Distribution:")
            for level, count in diversity.complexity_distribution.items():
                print(f"    {level:10} {count}")
        
        # Performance Section
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 50)
        print(f"  Total Time:              {performance.total_time_ms/1000:.2f}s")
        print(f"  Avg Time/Sample:         {performance.avg_time_per_sample_ms:.0f}ms")
        print(f"  Throughput:              {performance.samples_per_minute:.1f} samples/min")
        print()
        print(f"  Total Tokens:            {performance.total_prompt_tokens + performance.total_completion_tokens}")
        print(f"  Estimated Cost:          ${performance.estimated_cost_usd:.4f}")
        
        print("\n" + "=" * 70)
    
    def print_quality_bars(self) -> None:
        """Print visual quality indicators."""
        quality = self.collector.compute_quality_metrics()
        
        def bar(value: float, width: int = 30) -> str:
            filled = int(value * width)
            return "█" * filled + "░" * (width - filled)
        
        print("\n📈 QUALITY INDICATORS")
        print("-" * 50)
        print(f"Syntax Valid:      {bar(quality.syntax_valid_rate)} {quality.syntax_valid_rate*100:.1f}%")
        print(f"Semantic Valid:    {bar(quality.semantic_verified_rate)} {quality.semantic_verified_rate*100:.1f}%")
        print(f"Cross-Model Agree: {bar(quality.cross_model_agreement_rate)} {quality.cross_model_agreement_rate*100:.1f}%")
        print(f"Avg Confidence:    {bar(quality.avg_confidence)} {quality.avg_confidence*100:.1f}%")
    
    def export_csv(self, path: Path) -> None:
        """Export metrics to CSV for further analysis."""
        import csv
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'sample_id', 'domain', 'syntax_valid', 'semantic_verified',
                'confidence', 'cross_model_agreement', 'complexity',
                'latency_ms', 'prompt_tokens', 'completion_tokens', 'timestamp'
            ])
            
            # Data
            for m in self.collector.sample_metrics:
                writer.writerow([
                    m.sample_id, m.domain, m.syntax_valid, m.semantic_verified,
                    m.confidence, m.cross_model_agreement, m.complexity,
                    m.latency_ms, m.prompt_tokens, m.completion_tokens, m.timestamp
                ])
        
        print(f"Exported {len(self.collector.sample_metrics)} records to {path}")
    
    def export_json(self, path: Path) -> None:
        """Export full metrics snapshot to JSON."""
        snapshot = self.collector.take_snapshot()
        snapshot.save(path)
        print(f"Exported snapshot to {path}")
    
    def get_trend_data(self) -> dict:
        """Get trend data for visualization."""
        if len(self.collector.snapshots) < 2:
            return {"message": "Not enough data for trends (need 2+ snapshots)"}
        
        return {
            "timestamps": [s.timestamp for s in self.collector.snapshots],
            "verification_rates": [s.quality.semantic_verified_rate for s in self.collector.snapshots],
            "avg_confidences": [s.quality.avg_confidence for s in self.collector.snapshots],
            "throughputs": [s.performance.samples_per_minute for s in self.collector.snapshots],
        }


# =============================================================================
# Comparison Reports
# =============================================================================

class ComparisonReport:
    """
    Generate before/after comparison reports.
    """
    
    @staticmethod
    def compare_samples(
        before: list[dict],
        after: list[dict],
    ) -> dict:
        """
        Compare samples before and after verification.
        
        Returns a detailed comparison including:
        - Changed samples
        - Corrections applied
        - Quality improvements
        """
        before_by_id = {s.get('id') or s.get('sample_id'): s for s in before}
        after_by_id = {s.get('id') or s.get('sample_id'): s for s in after}
        
        changes = []
        corrections = []
        
        for sample_id, after_sample in after_by_id.items():
            if sample_id in before_by_id:
                before_sample = before_by_id[sample_id]
                
                # Check for ATL changes
                before_atl = before_sample.get('atl_formula', before_sample.get('atl', ''))
                after_atl = after_sample.get('corrected_atl') or after_sample.get('atl_formula', '')
                
                if before_atl != after_atl and after_sample.get('corrected_atl'):
                    corrections.append({
                        'sample_id': sample_id,
                        'nl': before_sample.get('nl_statement', before_sample.get('nl', '')),
                        'before_atl': before_atl,
                        'after_atl': after_atl,
                        'reason': after_sample.get('semantic_check', {}).get('explanation', 'N/A'),
                    })
                
                # Check for status changes
                before_verified = before_sample.get('verified', before_sample.get('syntax_valid'))
                after_status = after_sample.get('status', 'unknown')
                
                if before_verified != (after_status == 'verified'):
                    changes.append({
                        'sample_id': sample_id,
                        'before_status': 'verified' if before_verified else 'unverified',
                        'after_status': after_status,
                    })
        
        return {
            'total_before': len(before),
            'total_after': len(after),
            'corrections_applied': len(corrections),
            'status_changes': len(changes),
            'corrections': corrections,
            'changes': changes,
        }
    
    @staticmethod
    def generate_diff_report(comparison: dict, output_path: Optional[Path] = None) -> str:
        """Generate a human-readable diff report."""
        lines = [
            "=" * 60,
            "BEFORE/AFTER COMPARISON REPORT",
            "=" * 60,
            f"Total Samples (Before): {comparison['total_before']}",
            f"Total Samples (After):  {comparison['total_after']}",
            f"Corrections Applied:    {comparison['corrections_applied']}",
            f"Status Changes:         {comparison['status_changes']}",
            "",
        ]
        
        if comparison['corrections']:
            lines.append("CORRECTIONS:")
            lines.append("-" * 50)
            for c in comparison['corrections']:
                lines.append(f"\n[{c['sample_id']}]")
                lines.append(f"  NL: {c['nl'][:60]}...")
                lines.append(f"  BEFORE: {c['before_atl']}")
                lines.append(f"  AFTER:  {c['after_atl']}")
                lines.append(f"  REASON: {c['reason'][:80]}...")
        
        if comparison['changes']:
            lines.append("\n\nSTATUS CHANGES:")
            lines.append("-" * 50)
            for ch in comparison['changes']:
                lines.append(f"  [{ch['sample_id']}] {ch['before_status']} → {ch['after_status']}")
        
        lines.append("\n" + "=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
        
        return report


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    collector = MetricsCollector()
    dashboard = MetricsDashboard(collector)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "summary":
            dashboard.print_summary()
        
        elif command == "bars":
            dashboard.print_quality_bars()
        
        elif command == "export-csv":
            output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/metrics.csv")
            dashboard.export_csv(output)
        
        elif command == "export-json":
            output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/metrics_snapshot.json")
            dashboard.export_json(output)
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python metrics.py [summary|bars|export-csv|export-json] [output_path]")
    else:
        dashboard.print_summary()
