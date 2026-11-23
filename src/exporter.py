"""
Multi-format summary export module.
Generates structured Markdown reports, plain-text audit logs, and standardized CSV exports.
"""
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from src.config import SUMMARY_REPORT_MD, AUDIT_LOG_TXT, EXPORT_CSV, EXPORTS_DIR, IRIS_DISPLAY_NAMES
from src.models import IrisRecord, SummaryMetrics
from src.analytics import IrisAnalytics


class IrisExporter:
    """Handles multi-format reporting for external audit and review."""

    @staticmethod
    def _ensure_dir() -> Path:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return EXPORTS_DIR

    @classmethod
    def export_markdown_report(
        cls,
        records: List[IrisRecord],
        summary: SummaryMetrics,
        chart_paths: Optional[Dict[str, Path]] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generates a comprehensive executive summary in GitHub Flavored Markdown.
        """
        cls._ensure_dir()
        target_path = output_path or SUMMARY_REPORT_MD
        status_info = IrisAnalytics.get_operational_status(summary)

        md_lines = [
            "# Iris Dataset Classifier - Executive Summary Report",
            f"**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Operational State:** {status_info['label']}  ",
            f"*{status_info['message']}*",
            "",
            "---",
            "",
            "## 1. Key Performance Indicators",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Total Records Classified** | {summary.total_records} |",
            f"| **Average Model Confidence** | {round(summary.avg_confidence * 100, 2)}% |",
            f"| **Records with Domain Warnings** | {summary.warning_count} ({round((summary.warning_count / max(1, summary.total_records)) * 100, 1)}%) |",
            f"| **Observation Window (Oldest)** | {summary.oldest_timestamp or 'N/A'} |",
            f"| **Observation Window (Latest)** | {summary.latest_timestamp or 'N/A'} |",
            "",
            "## 2. Species Classification Breakdown",
            "",
            "| Species | Common / Display Name | Sample Count | Share (%) |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for sp, cnt in summary.species_counts.items():
            pct = summary.species_percentages.get(sp, 0.0)
            disp = IRIS_DISPLAY_NAMES.get(sp, sp.capitalize())
            md_lines.append(f"| `{sp}` | {disp} | {cnt} | {pct}% |")

        md_lines.extend([
            "",
            "## 3. Morphological Feature Statistics (cm)",
            "",
            "| Feature | Mean | Median | Std Dev | Min | Max |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        feat_display = {
            "sepal_length": "Sepal Length",
            "sepal_width": "Sepal Width",
            "petal_length": "Petal Length",
            "petal_width": "Petal Width"
        }

        for f_key, stats in summary.feature_statistics.items():
            name = feat_display.get(f_key, f_key)
            md_lines.append(
                f"| **{name}** | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |"
            )

        # Warnings and Anomalies Section
        md_lines.extend([
            "",
            "## 4. Anomaly & Warning Log (Recent Flags)",
            ""
        ])

        warn_records = [r for r in records if r.warnings]
        if not warn_records:
            md_lines.append("*No anomalies or warning flags recorded in current dataset.*")
        else:
            md_lines.extend([
                "| Record ID | Timestamp | Species | Confidence | Warnings Triggered |",
                "| :--- | :--- | :--- | :--- | :--- |"
            ])
            for r in warn_records[:15]:  # Show top 15
                w_str = "<br>".join(r.warnings)
                md_lines.append(
                    f"| `{r.id}` | {r.timestamp[:19]} | {r.predicted_species} | {round(r.confidence * 100, 1)}% | {w_str} |"
                )

        # Charts Section
        if chart_paths:
            md_lines.extend([
                "",
                "## 5. Visual Trends & Distributions",
                ""
            ])
            for name, p in chart_paths.items():
                title = name.replace("_", " ").title()
                md_lines.append(f"### {title}")
                md_lines.append(f"![{title}]({p.as_posix()})")
                md_lines.append("")

        with open(target_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        return target_path

    @classmethod
    def export_audit_log(
        cls,
        records: List[IrisRecord],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generates a chronological plain-text audit log of all classification operations.
        """
        cls._ensure_dir()
        target_path = output_path or AUDIT_LOG_TXT

        lines = [
            "=" * 80,
            "                   IRIS CLASSIFIER - AUDIT & ACTIVITY LOG",
            f"                Export Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"                Total Transaction Entries: {len(records)}",
            "=" * 80,
            ""
        ]

        for i, r in enumerate(sorted(records, key=lambda x: x.timestamp), 1):
            warn_tag = f" [WARNINGS: {len(r.warnings)}]" if r.warnings else ""
            lines.append(f"ENTRY #{i:03d} | ID: {r.id} | Timestamp: {r.timestamp}{warn_tag}")
            lines.append(f"  Inputs: Sepal = {r.sepal_length} x {r.sepal_width} cm | Petal = {r.petal_length} x {r.petal_width} cm")
            lines.append(f"  Outcome: Predicted {r.predicted_species.upper()} (Confidence: {round(r.confidence * 100, 2)}%) [Source: {r.source}]")
            if r.probabilities:
                prob_str = ", ".join(f"{k}: {round(v*100, 1)}%" for k, v in r.probabilities.items())
                lines.append(f"  Class Probabilities: [{prob_str}]")
            if r.warnings:
                for w in r.warnings:
                    lines.append(f"  ! FLAG: {w}")
            if r.notes:
                lines.append(f"  Notes: {r.notes}")
            lines.append("-" * 80)

        with open(target_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return target_path

    @classmethod
    def export_standard_csv(
        cls,
        records: List[IrisRecord],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generates standardized CSV file containing full tabular history.
        """
        cls._ensure_dir()
        target_path = output_path or EXPORT_CSV

        fieldnames = [
            "id", "timestamp", "sepal_length", "sepal_width",
            "petal_length", "petal_width", "predicted_species",
            "confidence", "prob_setosa", "prob_versicolor", "prob_virginica",
            "warnings", "source", "notes"
        ]

        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_flat_dict())

        return target_path
