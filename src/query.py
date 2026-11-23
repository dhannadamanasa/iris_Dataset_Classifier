"""
Record query, keyword search, multi-criteria filtering, and pagination module.
"""
from typing import Dict, List, Optional, Tuple
from src.models import IrisRecord


class IrisQueryEngine:
    """Provides advanced multi-attribute filtering, keyword search, and pagination."""

    @staticmethod
    def search_and_filter(
        records: List[IrisRecord],
        keyword: Optional[str] = None,
        species: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        min_sepal_length: Optional[float] = None,
        max_sepal_length: Optional[float] = None,
        min_petal_length: Optional[float] = None,
        max_petal_length: Optional[float] = None,
        only_with_warnings: bool = False,
        sort_by: str = "timestamp",
        reverse: bool = True
    ) -> List[IrisRecord]:
        """
        Filters and sorts Iris records based on multiple criteria.
        """
        results = records

        # 1. Keyword search (case-insensitive across ID, notes, predicted_species, timestamp)
        if keyword:
            kw = keyword.strip().lower()
            results = [
                r for r in results
                if kw in r.id.lower()
                or kw in r.notes.lower()
                or kw in r.predicted_species.lower()
                or kw in r.timestamp.lower()
                or any(kw in w.lower() for w in r.warnings)
            ]

        # 2. Species filter
        if species and species.strip().lower() != "all":
            sp = species.strip().lower()
            results = [r for r in results if r.predicted_species.lower() == sp]

        # 3. Confidence range filter
        if min_confidence is not None:
            results = [r for r in results if r.confidence >= min_confidence]
        if max_confidence is not None:
            results = [r for r in results if r.confidence <= max_confidence]

        # 4. Feature boundary filters
        if min_sepal_length is not None:
            results = [r for r in results if r.sepal_length >= min_sepal_length]
        if max_sepal_length is not None:
            results = [r for r in results if r.sepal_length <= max_sepal_length]

        if min_petal_length is not None:
            results = [r for r in results if r.petal_length >= min_petal_length]
        if max_petal_length is not None:
            results = [r for r in results if r.petal_length <= max_petal_length]

        # 5. Warning indicator filter
        if only_with_warnings:
            results = [r for r in results if len(r.warnings) > 0]

        # 6. Sorting
        valid_sort_keys = {
            "timestamp": lambda r: r.timestamp,
            "confidence": lambda r: r.confidence,
            "id": lambda r: r.id,
            "species": lambda r: r.predicted_species,
            "sepal_length": lambda r: r.sepal_length,
            "petal_length": lambda r: r.petal_length
        }
        sort_key_fn = valid_sort_keys.get(sort_by, lambda r: r.timestamp)
        results = sorted(results, key=sort_key_fn, reverse=reverse)

        return results

    @staticmethod
    def paginate(
        records: List[IrisRecord],
        page: int = 1,
        page_size: int = 10
    ) -> Tuple[List[IrisRecord], int, int]:
        """
        Splits records into pages.
        Returns: (page_records, current_page, total_pages)
        """
        if not records:
            return [], 1, 1
        page_size = max(1, page_size)
        total_pages = max(1, (len(records) + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return records[start_idx:end_idx], page, total_pages
