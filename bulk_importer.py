"""
Bulk dataset ingestion module with automatic schema validation and row-by-row error logging.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from src.validator import IrisValidator
from src.classifier import get_classifier
from src.models import IrisRecord
from src.storage import IrisStorage


@dataclass
class BulkImportSummary:
    """Summary metrics of a bulk ingestion batch."""
    total_rows: int
    imported_count: int
    skipped_count: int
    errors: List[Dict[str, Union[int, str]]] = field(default_factory=list)
    imported_records: List[IrisRecord] = field(default_factory=list)


class BulkDatasetImporter:
    """Handles parsing, validation, classification, and persistence of bulk CSV files."""

    @staticmethod
    def _normalize_column_name(col: str) -> str:
        """Normalizes column names to standard internal keys."""
        cleaned = col.strip().lower().replace(" ", "_").replace("-", "_")
        mapping = {
            "sepallength": "sepal_length",
            "sepallengthcm": "sepal_length",
            "sepal_length_cm": "sepal_length",
            "sepalwidth": "sepal_width",
            "sepalwidthcm": "sepal_width",
            "sepal_width_cm": "sepal_width",
            "petallength": "petal_length",
            "petallengthcm": "petal_length",
            "petal_length_cm": "petal_length",
            "petalwidth": "petal_width",
            "petalwidthcm": "petal_width",
            "petal_width_cm": "petal_width",
            "class": "species",
            "target": "species",
            "variety": "species"
        }
        return mapping.get(cleaned, cleaned)

    @classmethod
    def import_csv(
        cls,
        csv_file_path: Union[str, Path],
        persist_to_storage: bool = True,
        source_label: str = "bulk_import"
    ) -> BulkImportSummary:
        """
        Parses and ingests an external CSV file with strict row-by-row error logging.
        """
        path = Path(csv_file_path)
        if not path.exists():
            return BulkImportSummary(
                total_rows=0,
                imported_count=0,
                skipped_count=0,
                errors=[{"row": 0, "error": f"File not found: {path}"}]
            )

        classifier = get_classifier()
        valid_records: List[IrisRecord] = []
        errors: List[Dict[str, Union[int, str]]] = []
        row_index = 0

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return BulkImportSummary(
                        total_rows=0,
                        imported_count=0,
                        skipped_count=0,
                        errors=[{"row": 1, "error": "CSV file is completely empty."}]
                    )

                norm_header = [cls._normalize_column_name(c) for c in header]
                required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

                # Verify all 4 required features exist in the header
                missing_cols = [c for c in required if c not in norm_header]
                if missing_cols:
                    return BulkImportSummary(
                        total_rows=0,
                        imported_count=0,
                        skipped_count=0,
                        errors=[{
                            "row": 1,
                            "error": f"Missing required column headers: {', '.join(missing_cols)}"
                        }]
                    )

                idx_map = {col: norm_header.index(col) for col in required}
                notes_idx = norm_header.index("notes") if "notes" in norm_header else -1
                species_idx = norm_header.index("species") if "species" in norm_header else (
                    norm_header.index("predicted_species") if "predicted_species" in norm_header else -1
                )

                for line_num, row in enumerate(reader, start=2):
                    row_index += 1
                    # Skip blank lines
                    if not row or all(c.strip() == "" for c in row):
                        continue

                    # Check column length
                    if len(row) < max(idx_map.values()) + 1:
                        errors.append({
                            "row": line_num,
                            "error": f"Row has insufficient columns (expected at least {len(required)}, found {len(row)})."
                        })
                        continue

                    # Extract raw values
                    raw_sl = row[idx_map["sepal_length"]]
                    raw_sw = row[idx_map["sepal_width"]]
                    raw_pl = row[idx_map["petal_length"]]
                    raw_pw = row[idx_map["petal_width"]]

                    # Validate each feature defensively
                    val_errors = []
                    val_warns = []
                    parsed_vals = {}

                    for feat_name, raw_val in [
                        ("sepal_length", raw_sl),
                        ("sepal_width", raw_sw),
                        ("petal_length", raw_pl),
                        ("petal_width", raw_pw)
                    ]:
                        res = IrisValidator.validate_feature(feat_name, raw_val)
                        if not res.is_valid:
                            val_errors.extend(res.errors)
                        else:
                            parsed_vals[feat_name] = res.cleaned_value
                            val_warns.extend(res.warnings)

                    if val_errors:
                        errors.append({
                            "row": line_num,
                            "error": "; ".join(val_errors)
                        })
                        continue

                    # Validate cross-feature anomalies
                    is_valid, set_errors, set_warns = IrisValidator.validate_sample_set(
                        parsed_vals["sepal_length"],
                        parsed_vals["sepal_width"],
                        parsed_vals["petal_length"],
                        parsed_vals["petal_width"]
                    )

                    if not is_valid:
                        errors.append({
                            "row": line_num,
                            "error": "; ".join(set_errors)
                        })
                        continue

                    combined_warnings = list(set(val_warns + set_warns))

                    # Run ML prediction
                    pred_species, conf, probs, _ = classifier.predict(
                        parsed_vals["sepal_length"],
                        parsed_vals["sepal_width"],
                        parsed_vals["petal_length"],
                        parsed_vals["petal_width"]
                    )

                    # Append prediction warnings
                    combined_warnings = IrisValidator.evaluate_prediction_warnings(conf, combined_warnings)

                    # Extract optional notes or specified species
                    row_notes = row[notes_idx].strip() if notes_idx != -1 and notes_idx < len(row) else ""
                    if species_idx != -1 and species_idx < len(row):
                        given_sp = row[species_idx].strip().lower().replace("iris-", "")
                        if given_sp and given_sp != pred_species.lower():
                            combined_warnings.append(
                                f"Label Mismatch: Ingested label '{given_sp}' differs from ML prediction '{pred_species}'."
                            )

                    # Create record
                    record = IrisRecord.create_new(
                        sepal_length=parsed_vals["sepal_length"],
                        sepal_width=parsed_vals["sepal_width"],
                        petal_length=parsed_vals["petal_length"],
                        petal_width=parsed_vals["petal_width"],
                        predicted_species=pred_species,
                        confidence=conf,
                        probabilities=probs,
                        warnings=combined_warnings,
                        source=source_label,
                        notes=row_notes
                    )
                    valid_records.append(record)

        except Exception as e:
            errors.append({"row": 0, "error": f"Fatal file read error: {str(e)}"})

        # Persist valid records to local storage if requested
        if persist_to_storage and valid_records:
            storage = IrisStorage()
            storage.add_records_bulk(valid_records)

        return BulkImportSummary(
            total_rows=row_index,
            imported_count=len(valid_records),
            skipped_count=len(errors),
            errors=errors,
            imported_records=valid_records
        )
