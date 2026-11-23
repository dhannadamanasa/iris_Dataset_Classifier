"""
Fail-safe persistence engine for JSON and CSV storage with automated corruption recovery.
"""
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from src.config import RECORDS_JSON, RECORDS_CSV, DATA_DIR
from src.models import IrisRecord


class IrisStorage:
    """Manages dual JSON and CSV local persistence with automatic schema healing."""

    def __init__(self, json_path: Path = RECORDS_JSON, csv_path: Path = RECORDS_CSV):
        self.json_path = Path(json_path)
        self.csv_path = Path(csv_path)
        self.ensure_initialized()

    def ensure_initialized(self) -> None:
        """Ensures storage directory and baseline storage files exist with valid schemas."""
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.json_path.exists():
            self._write_empty_json()
        else:
            # Validate JSON integrity; recover if corrupted
            self._verify_json_integrity()

        if not self.csv_path.exists():
            self._write_empty_csv()

    def _write_empty_json(self) -> None:
        """Writes an empty JSON template structure."""
        initial_data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "total_records": 0,
            "records": []
        }
        temp_file = self.json_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2)
        shutil.move(temp_file, self.json_path)

    def _write_empty_csv(self) -> None:
        """Writes empty CSV with proper header columns."""
        fieldnames = [
            "id", "timestamp", "sepal_length", "sepal_width",
            "petal_length", "petal_width", "predicted_species",
            "confidence", "prob_setosa", "prob_versicolor", "prob_virginica",
            "warnings", "source", "notes"
        ]
        temp_file = self.csv_path.with_suffix(".tmp")
        with open(temp_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        shutil.move(temp_file, self.csv_path)

    def _verify_json_integrity(self) -> bool:
        """
        Detects unformatted or corrupted JSON files.
        If corrupted, backs up corrupted file and initializes clean template.
        """
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict) or "records" not in data:
                    raise ValueError("JSON missing root 'records' list.")
            return True
        except (json.JSONDecodeError, ValueError, OSError) as e:
            # Backup corrupted file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.json_path.with_name(f"records_corrupted_{timestamp}.json.bak")
            try:
                shutil.copy(self.json_path, backup_path)
            except Exception:
                pass
            # Self-heal with clean template
            self._write_empty_json()
            return False

    def load_all_records(self) -> List[IrisRecord]:
        """
        Reads all records from JSON, with automatic fallback to CSV if JSON fails.
        """
        self.ensure_initialized()
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                raw_records = payload.get("records", [])
                return [IrisRecord.from_dict(item) for item in raw_records]
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # Fallback to CSV if JSON had an issue
            return self._load_from_csv()

    def _load_from_csv(self) -> List[IrisRecord]:
        """Reads records from CSV."""
        if not self.csv_path.exists():
            return []
        records = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        records.append(IrisRecord.from_flat_dict(row))
                    except Exception:
                        continue
        except Exception:
            return []
        return records

    def save_all_records(self, records: List[IrisRecord]) -> bool:
        """
        Atomically writes full record set to both JSON and CSV files.
        """
        try:
            # 1. Write JSON atomically
            json_payload = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "total_records": len(records),
                "records": [r.to_dict() for r in records]
            }
            temp_json = self.json_path.with_suffix(".tmp")
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2)
            shutil.move(temp_json, self.json_path)

            # 2. Write CSV atomically
            fieldnames = [
                "id", "timestamp", "sepal_length", "sepal_width",
                "petal_length", "petal_width", "predicted_species",
                "confidence", "prob_setosa", "prob_versicolor", "prob_virginica",
                "warnings", "source", "notes"
            ]
            temp_csv = self.csv_path.with_suffix(".tmp")
            with open(temp_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    writer.writerow(r.to_flat_dict())
            shutil.move(temp_csv, self.csv_path)

            return True
        except Exception as e:
            return False

    def add_record(self, record: IrisRecord) -> bool:
        """Appends a new record to both stores."""
        records = self.load_all_records()
        records.append(record)
        return self.save_all_records(records)

    def add_records_bulk(self, new_records: List[IrisRecord]) -> int:
        """Appends multiple records in a single batch atomic transaction."""
        records = self.load_all_records()
        existing_ids = {r.id for r in records}
        added_count = 0
        for nr in new_records:
            if nr.id not in existing_ids:
                records.append(nr)
                existing_ids.add(nr.id)
                added_count += 1
        self.save_all_records(records)
        return added_count

    def clear_all_records(self) -> bool:
        """Clears all records, leaving clean empty templates."""
        return self.save_all_records([])
