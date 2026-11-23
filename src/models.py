"""
Data models and structures for Iris Dataset Classifier Demo.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
import uuid


@dataclass
class ValidationResult:
    """Encapsulates validation outcome, errors, and warning flags."""
    is_valid: bool
    cleaned_value: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class IrisRecord:
    """Represents a single classified Iris observation record."""
    id: str
    timestamp: str
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    predicted_species: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    source: str = "manual"
    notes: str = ""

    @classmethod
    def create_new(
        cls,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        predicted_species: str,
        confidence: float,
        probabilities: Optional[Dict[str, float]] = None,
        warnings: Optional[List[str]] = None,
        source: str = "manual",
        notes: str = "",
        record_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> "IrisRecord":
        """Factory method to construct a validated record with automatic ID & timestamp."""
        r_id = record_id or f"IRIS-{uuid.uuid4().hex[:8].upper()}"
        ts = timestamp or datetime.now().isoformat()
        probs = probabilities or {}
        warns = warnings or []
        return cls(
            id=r_id,
            timestamp=ts,
            sepal_length=round(float(sepal_length), 2),
            sepal_width=round(float(sepal_width), 2),
            petal_length=round(float(petal_length), 2),
            petal_width=round(float(petal_width), 2),
            predicted_species=predicted_species,
            confidence=round(float(confidence), 4),
            probabilities={k: round(float(v), 4) for k, v in probs.items()},
            warnings=warns,
            source=source,
            notes=notes.strip()
        )

    def to_dict(self) -> Dict:
        """Converts record to structured dictionary (for JSON persistence)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "IrisRecord":
        """Recreates an IrisRecord instance from a dictionary."""
        return cls(
            id=str(data.get("id", f"IRIS-{uuid.uuid4().hex[:8].upper()}")),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
            sepal_length=float(data["sepal_length"]),
            sepal_width=float(data["sepal_width"]),
            petal_length=float(data["petal_length"]),
            petal_width=float(data["petal_width"]),
            predicted_species=str(data.get("predicted_species", "unknown")),
            confidence=float(data.get("confidence", 0.0)),
            probabilities=dict(data.get("probabilities", {})),
            warnings=list(data.get("warnings", [])),
            source=str(data.get("source", "manual")),
            notes=str(data.get("notes", ""))
        )

    def to_flat_dict(self) -> Dict:
        """Converts record to flat dictionary for CSV serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "sepal_length": self.sepal_length,
            "sepal_width": self.sepal_width,
            "petal_length": self.petal_length,
            "petal_width": self.petal_width,
            "predicted_species": self.predicted_species,
            "confidence": self.confidence,
            "prob_setosa": self.probabilities.get("setosa", 0.0),
            "prob_versicolor": self.probabilities.get("versicolor", 0.0),
            "prob_virginica": self.probabilities.get("virginica", 0.0),
            "warnings": "; ".join(self.warnings),
            "source": self.source,
            "notes": self.notes
        }

    @classmethod
    def from_flat_dict(cls, row: Dict) -> "IrisRecord":
        """Reconstructs record from a flat CSV row dictionary."""
        probs = {
            "setosa": float(row.get("prob_setosa", 0.0)),
            "versicolor": float(row.get("prob_versicolor", 0.0)),
            "virginica": float(row.get("prob_virginica", 0.0))
        }
        raw_warns = row.get("warnings", "")
        warnings = [w.strip() for w in raw_warns.split("; ") if w.strip()] if raw_warns else []
        return cls(
            id=str(row.get("id", f"IRIS-{uuid.uuid4().hex[:8].upper()}")),
            timestamp=str(row.get("timestamp", datetime.now().isoformat())),
            sepal_length=float(row["sepal_length"]),
            sepal_width=float(row["sepal_width"]),
            petal_length=float(row["petal_length"]),
            petal_width=float(row["petal_width"]),
            predicted_species=str(row.get("predicted_species", "unknown")),
            confidence=float(row.get("confidence", 0.0)),
            probabilities=probs,
            warnings=warnings,
            source=str(row.get("source", "csv")),
            notes=str(row.get("notes", ""))
        )


@dataclass
class SummaryMetrics:
    """Aggregated operational and statistical summary metrics."""
    total_records: int
    species_counts: Dict[str, int]
    species_percentages: Dict[str, float]
    avg_confidence: float
    feature_statistics: Dict[str, Dict[str, float]]
    warning_count: int
    latest_timestamp: Optional[str]
    oldest_timestamp: Optional[str]
