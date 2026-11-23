"""
Executive summary aggregation and statistical analytics module.
"""
import math
import statistics
from typing import Dict, List, Optional
from src.config import IRIS_CLASSES, IRIS_DISPLAY_NAMES
from src.models import IrisRecord, SummaryMetrics


class IrisAnalytics:
    """Computes comprehensive aggregations, descriptive statistics, and operational trends."""

    @classmethod
    def compute_summary(cls, records: List[IrisRecord]) -> SummaryMetrics:
        """Computes all summary metrics across the current dataset."""
        total = len(records)
        if total == 0:
            return SummaryMetrics(
                total_records=0,
                species_counts={sp: 0 for sp in IRIS_CLASSES},
                species_percentages={sp: 0.0 for sp in IRIS_CLASSES},
                avg_confidence=0.0,
                feature_statistics={
                    feat: {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                    for feat in ["sepal_length", "sepal_width", "petal_length", "petal_width"]
                },
                warning_count=0,
                latest_timestamp=None,
                oldest_timestamp=None
            )

        # Species counts & percentages
        species_counts = {sp: 0 for sp in IRIS_CLASSES}
        for r in records:
            sp = r.predicted_species.lower()
            if sp in species_counts:
                species_counts[sp] += 1
            else:
                species_counts[sp] = 1

        species_percentages = {
            sp: round((cnt / total) * 100, 2)
            for sp, cnt in species_counts.items()
        }

        # Confidence stats
        confidences = [r.confidence for r in records]
        avg_confidence = round(statistics.mean(confidences), 4)

        # Feature statistics
        feature_stats = {}
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for feat in features:
            values = [getattr(r, feat) for r in records]
            if values:
                mean_v = statistics.mean(values)
                median_v = statistics.median(values)
                std_v = statistics.stdev(values) if len(values) > 1 else 0.0
                min_v = min(values)
                max_v = max(values)
                feature_stats[feat] = {
                    "mean": round(mean_v, 2),
                    "median": round(median_v, 2),
                    "std": round(std_v, 2),
                    "min": round(min_v, 2),
                    "max": round(max_v, 2)
                }

        # Warnings count
        warning_count = sum(1 for r in records if len(r.warnings) > 0)

        # Timestamps
        sorted_by_ts = sorted(records, key=lambda r: r.timestamp)
        oldest_ts = sorted_by_ts[0].timestamp if sorted_by_ts else None
        latest_ts = sorted_by_ts[-1].timestamp if sorted_by_ts else None

        return SummaryMetrics(
            total_records=total,
            species_counts=species_counts,
            species_percentages=species_percentages,
            avg_confidence=avg_confidence,
            feature_statistics=feature_stats,
            warning_count=warning_count,
            latest_timestamp=latest_ts,
            oldest_timestamp=oldest_ts
        )

    @classmethod
    def get_operational_status(cls, summary: SummaryMetrics) -> Dict[str, str]:
        """Evaluates operational health status indicators based on aggregate thresholds."""
        if summary.total_records == 0:
            return {
                "status": "AWAITING_DATA",
                "label": "🟡 Awaiting Data",
                "message": "No classification records stored yet."
            }

        warning_ratio = summary.warning_count / summary.total_records
        if summary.avg_confidence >= 0.85 and warning_ratio < 0.15:
            return {
                "status": "OPTIMAL",
                "label": "🟢 Optimal Operational State",
                "message": "High classification confidence with low warning rate."
            }
        elif summary.avg_confidence >= 0.70 and warning_ratio < 0.35:
            return {
                "status": "MODERATE",
                "label": "🟡 Moderate Operational State",
                "message": "Acceptable performance; minor proportion of boundary cases detected."
            }
        else:
            return {
                "status": "ATTENTION_REQUIRED",
                "label": "🔴 Attention Required",
                "message": "Elevated anomaly/warning frequency or low average confidence."
            }
