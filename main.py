"""
Iris Dataset Classifier Demo
Main interactive menu-driven terminal application.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional

from src.config import (
    Colors, FEATURE_CONSTRAINTS, IRIS_DISPLAY_NAMES,
    RECORDS_JSON, RECORDS_CSV, SUMMARY_REPORT_MD, AUDIT_LOG_TXT, EXPORT_CSV
)
from src.models import IrisRecord, SummaryMetrics
from src.validator import IrisValidator, prompt_validated_float
from src.classifier import get_classifier
from src.storage import IrisStorage
from src.query import IrisQueryEngine
from src.analytics import IrisAnalytics
from src.visualizer import IrisVisualizer
from src.exporter import IrisExporter
from bulk_importer import BulkDatasetImporter


def clear_screen():
    """Clears console if supported."""
    # Keep output visible in non-interactive / redirection shells
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    """Prints clean, branded welcome banner."""
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}")
    print("=" * 76)
    print("  🌸  IRIS DATASET CLASSIFIER DEMO  |  AI & SMART APPLICATIONS  🌸")
    print("      Robust Scikit-Learn Engine • Defensive Validation • Dual Store")
    print("=" * 76)
    print(f"{Colors.RESET}")


def format_table(records: List[IrisRecord], title: str = "CLASSIFICATION RECORDS") -> str:
    """Formats records into a fixed-width, readable terminal table with clean borders."""
    if not records:
        return f"\n{Colors.YELLOW}No records found.{Colors.RESET}\n"

    # Columns: ID (16), Timestamp (16), Sepal (10), Petal (10), Species (12), Conf (8), Warn (6)
    header = (
        f"| {'RECORD ID':<16} | {'DATE/TIME':<16} | {'SEPAL (L x W)':<13} | "
        f"{'PETAL (L x W)':<13} | {'PREDICTED':<12} | {'CONF':<7} | {'FLAG':<5} |"
    )
    divider = "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+" + "-" * 14 + "+" + "-" * 9 + "+" + "-" * 7 + "+"

    lines = [
        f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}>>> {title} ({len(records)} entries){Colors.RESET}",
        divider,
        f"{Colors.BOLD}{header}{Colors.RESET}",
        divider
    ]

    for r in records:
        sepal_str = f"{r.sepal_length:.1f}x{r.sepal_width:.1f}"
        petal_str = f"{r.petal_length:.1f}x{r.petal_width:.1f}"
        conf_str = f"{r.confidence * 100:.1f}%"
        flag_str = "⚠️" if r.warnings else "✓"
        ts_short = r.timestamp[:16].replace("T", " ")
        
        # Color-code species
        sp_color = Colors.CYAN
        if r.predicted_species == "setosa":
            sp_color = Colors.GREEN
        elif r.predicted_species == "versicolor":
            sp_color = Colors.YELLOW
        elif r.predicted_species == "virginica":
            sp_color = Colors.MAGENTA

        row_str = (
            f"| {r.id:<16} | {ts_short:<16} | {sepal_str:<13} | "
            f"{petal_str:<13} | {sp_color}{r.predicted_species:<12}{Colors.RESET} | {conf_str:<7} | {flag_str:<5} |"
        )
        lines.append(row_str)

    lines.append(divider)
    return "\n".join(lines)


def handle_record_view_paginated(storage: IrisStorage):
    """Displays stored classification records with interactive pagination."""
    records = storage.load_all_records()
    if not records:
        print(f"\n{Colors.YELLOW}No records stored in local database yet.{Colors.RESET}")
        return

    page_size = 10
    current_page = 1
    total_pages = max(1, (len(records) + page_size - 1) // page_size)

    while True:
        page_records, current_page, total_pages = IrisQueryEngine.paginate(records, current_page, page_size)
        print(format_table(page_records, f"STORED RECORDS (Page {current_page} of {total_pages})"))
        print(f"{Colors.DIM}Total: {len(records)} records | Page {current_page}/{total_pages}{Colors.RESET}")
        print(f"{Colors.CYAN}[N] Next Page  |  [P] Previous Page  |  [G] Go to Page  |  [B] Back to Menu{Colors.RESET}")
        
        choice = input(f"{Colors.BOLD}Select action: {Colors.RESET}").strip().upper()
        if choice == "N":
            if current_page < total_pages:
                current_page += 1
            else:
                print(f"{Colors.YELLOW}Already on the last page.{Colors.RESET}")
        elif choice == "P":
            if current_page > 1:
                current_page -= 1
            else:
                print(f"{Colors.YELLOW}Already on the first page.{Colors.RESET}")
        elif choice == "G":
            try:
                p_num = int(input(f"Enter page number (1-{total_pages}): ").strip())
                if 1 <= p_num <= total_pages:
                    current_page = p_num
                else:
                    print(f"{Colors.RED}Invalid page number.{Colors.RESET}")
            except ValueError:
                print(f"{Colors.RED}Please enter an integer.{Colors.RESET}")
        elif choice == "B" or choice == "":
            break


def handle_new_classification(storage: IrisStorage):
    """Interactive workflow to record, validate, classify, and persist a new sample."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_GREEN}--- CLASSIFY NEW IRIS SPECIMEN ---{Colors.RESET}")
    print(f"{Colors.DIM}Enter the botanical dimensions in centimeters. Real-time validation active.{Colors.RESET}\n")

    classifier = get_classifier()

    # Step-by-step defensive input gathering
    sl = prompt_validated_float("sepal_length")
    sw = prompt_validated_float("sepal_width")
    pl = prompt_validated_float("petal_length")
    pw = prompt_validated_float("petal_width")

    # Cross-feature morphological check
    is_valid, errors, warnings = IrisValidator.validate_sample_set(sl, sw, pl, pw)
    if not is_valid:
        print(f"\n{Colors.RED}Validation Error: {'; '.join(errors)}{Colors.RESET}")
        print("Record aborted. Please try again with valid morphology.")
        return

    # Optional observation notes
    notes = input(f"{Colors.CYAN}Optional notes or collector comment (press Enter to skip): {Colors.RESET}").strip()

    # Run ML Engine
    pred_species, confidence, probs, engine_name = classifier.predict(sl, sw, pl, pw)

    # Threshold warnings evaluation
    warnings = IrisValidator.evaluate_prediction_warnings(confidence, warnings)

    # Create structured record
    record = IrisRecord.create_new(
        sepal_length=sl,
        sepal_width=sw,
        petal_length=pl,
        petal_width=pw,
        predicted_species=pred_species,
        confidence=confidence,
        probabilities=probs,
        warnings=warnings,
        source="interactive_cli",
        notes=notes
    )

    # Persist
    storage.add_record(record)

    # Display Rich Result Card
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_GREEN}================ CLASSIFICATION RESULT ================{Colors.RESET}")
    print(f"  Record ID         : {Colors.BOLD}{record.id}{Colors.RESET}")
    print(f"  Predicted Species : {Colors.BOLD}{Colors.BRIGHT_CYAN}{IRIS_DISPLAY_NAMES.get(pred_species, pred_species)}{Colors.RESET}")
    print(f"  Model Confidence  : {Colors.BOLD}{confidence * 100:.2f}%{Colors.RESET}")
    print(f"  Engine Mode       : {Colors.DIM}{engine_name}{Colors.RESET}")
    print("  Class Probabilities:")
    for sp, prob in probs.items():
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    - {IRIS_DISPLAY_NAMES.get(sp, sp):<18}: {prob * 100:5.1f}% |{bar}|")

    if warnings:
        print(f"\n  {Colors.BOLD}{Colors.YELLOW}Automated Warning Flags:{Colors.RESET}")
        for w in warnings:
            print(f"    ⚠️  {Colors.YELLOW}{w}{Colors.RESET}")
    else:
        print(f"\n  {Colors.GREEN}✓ No anomalous limit breaches or botanical warnings detected.{Colors.RESET}")

    print(f"  {Colors.DIM}Record successfully saved to JSON and CSV stores.{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}======================================================={Colors.RESET}\n")


def handle_search_and_query(storage: IrisStorage):
    """Interactive multi-criteria search and filtering."""
    records = storage.load_all_records()
    if not records:
        print(f"\n{Colors.YELLOW}No records stored to query.{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}--- RECORD QUERY & SEARCH ---{Colors.RESET}")
    print("Leave options blank to omit specific criteria:")

    kw = input("Search keyword (ID, note, timestamp): ").strip()
    sp = input("Species filter (setosa / versicolor / virginica / all): ").strip().lower()
    if sp not in ["setosa", "versicolor", "virginica"]:
        sp = None

    conf_input = input("Minimum confidence percentage (e.g. 80 for 80%): ").strip()
    min_conf = (float(conf_input) / 100.0) if conf_input.replace(".", "").isdigit() else None

    warn_only_input = input("Filter only records with warnings? (y/N): ").strip().lower()
    warn_only = warn_only_input == "y"

    results = IrisQueryEngine.search_and_filter(
        records,
        keyword=kw if kw else None,
        species=sp,
        min_confidence=min_conf,
        only_with_warnings=warn_only
    )

    print(format_table(results, f"QUERY RESULTS (Found {len(results)} matches)"))


def handle_executive_summary(storage: IrisStorage):
    """Computes and renders executive metrics, KPI cards, and operational status."""
    records = storage.load_all_records()
    summary = IrisAnalytics.compute_summary(records)
    status_info = IrisAnalytics.get_operational_status(summary)

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}================= EXECUTIVE SUMMARY REPORT ================={Colors.RESET}")
    print(f"  Operational State: {Colors.BOLD}{status_info['label']}{Colors.RESET}")
    print(f"  {Colors.DIM}{status_info['message']}{Colors.RESET}\n")

    print(f"  Total Classifications  : {Colors.BOLD}{summary.total_records}{Colors.RESET}")
    print(f"  Average Confidence     : {Colors.BOLD}{summary.avg_confidence * 100:.2f}%{Colors.RESET}")
    print(f"  Flagged Warnings       : {Colors.BOLD}{summary.warning_count}{Colors.RESET} ({summary.warning_count / max(1, summary.total_records) * 100:.1f}%)")
    print(f"  Earliest Record        : {summary.oldest_timestamp or 'N/A'}")
    print(f"  Latest Record          : {summary.latest_timestamp or 'N/A'}")

    print(f"\n  {Colors.BOLD}Species Distribution Breakdown:{Colors.RESET}")
    for sp, cnt in summary.species_counts.items():
        pct = summary.species_percentages.get(sp, 0.0)
        disp = IRIS_DISPLAY_NAMES.get(sp, sp.capitalize())
        print(f"    • {disp:<18}: {cnt:3d} specimens ({pct:5.1f}%)")

    print(f"\n  {Colors.BOLD}Feature Statistical Summary (cm):{Colors.RESET}")
    print(f"  {'Feature':<16} | {'Mean':<6} | {'Median':<6} | {'Std':<6} | {'Min':<6} | {'Max':<6}")
    print("  " + "-" * 55)
    for feat, stats in summary.feature_statistics.items():
        f_name = feat.replace("_", " ").title()
        print(f"  {f_name:<16} | {stats['mean']:<6.2f} | {stats['median']:<6.2f} | {stats['std']:<6.2f} | {stats['min']:<6.2f} | {stats['max']:<6.2f}")

    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}============================================================{Colors.RESET}\n")


def handle_generate_charts(storage: IrisStorage):
    """Generates all Matplotlib visual plots and displays confirmation paths."""
    records = storage.load_all_records()
    if not records:
        print(f"\n{Colors.YELLOW}No records available to generate charts.{Colors.RESET}")
        return

    print(f"\n{Colors.CYAN}Generating high-resolution Matplotlib charts...{Colors.RESET}")
    summary = IrisAnalytics.compute_summary(records)
    charts = IrisVisualizer.generate_all_charts(records, summary)

    print(f"{Colors.GREEN}✓ Successfully generated visual distribution charts:{Colors.RESET}")
    for name, path in charts.items():
        print(f"  • {name:<24}: {path}")


def handle_export_reports(storage: IrisStorage):
    """Exports structured Markdown, plain-text audit logs, and standardized CSV."""
    records = storage.load_all_records()
    if not records:
        print(f"\n{Colors.YELLOW}No records available to export.{Colors.RESET}")
        return

    print(f"\n{Colors.CYAN}Generating multi-format export reports...{Colors.RESET}")
    summary = IrisAnalytics.compute_summary(records)
    charts = IrisVisualizer.generate_all_charts(records, summary)

    md_path = IrisExporter.export_markdown_report(records, summary, charts)
    log_path = IrisExporter.export_audit_log(records)
    csv_path = IrisExporter.export_standard_csv(records)

    print(f"{Colors.GREEN}✓ Reports successfully exported:{Colors.RESET}")
    print(f"  • Markdown Executive Report : {md_path}")
    print(f"  • Plain-Text Audit Log      : {log_path}")
    print(f"  • Standardized CSV Dataset  : {csv_path}")


def handle_bulk_import(storage: IrisStorage):
    """Guided CSV bulk import workflow."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}--- BULK DATASET CSV INGESTION ---{Colors.RESET}")
    file_path = input("Enter path to CSV file (press Enter to cancel): ").strip().strip("\"'")
    if not file_path:
        return

    path = Path(file_path)
    if not path.exists():
        print(f"{Colors.RED}Error: File not found at '{file_path}'{Colors.RESET}")
        return

    print(f"{Colors.CYAN}Ingesting and validating dataset rows...{Colors.RESET}")
    summary = BulkDatasetImporter.import_csv(path, persist_to_storage=True)

    print(f"\n{Colors.BOLD}Bulk Import Summary:{Colors.RESET}")
    print(f"  Total Scanned Rows   : {summary.total_rows}")
    print(f"  Successfully Imported: {Colors.GREEN}{summary.imported_count}{Colors.RESET}")
    print(f"  Skipped / Failed Rows: {Colors.RED if summary.skipped_count else Colors.GREEN}{summary.skipped_count}{Colors.RESET}")

    if summary.errors:
        print(f"\n{Colors.YELLOW}Detailed Row-by-Row Error Log:{Colors.RESET}")
        for err in summary.errors[:10]:  # Show first 10
            print(f"  Row {err['row']}: {err['error']}")
        if len(summary.errors) > 10:
            print(f"  ... and {len(summary.errors) - 10} more errors.")


def handle_system_diagnostics(storage: IrisStorage):
    """Displays storage diagnostics, model status, and fallback mechanisms."""
    classifier = get_classifier()
    metrics = classifier.get_model_metrics()
    records = storage.load_all_records()

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}================ SYSTEM DIAGNOSTICS & HEALTH ================{Colors.RESET}")
    print(f"  ML Engine Architecture  : {metrics['engine']}")
    print(f"  Test Set Accuracy       : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  Train Set Accuracy      : {metrics['train_accuracy'] * 100:.2f}%")
    print(f"  JSON Storage Path       : {RECORDS_JSON} (Exists: {RECORDS_JSON.exists()})")
    print(f"  CSV Storage Path        : {RECORDS_CSV} (Exists: {RECORDS_CSV.exists()})")
    print(f"  Total Active Records    : {len(records)}")
    print(f"  Dual Store Integrity    : {Colors.GREEN}VERIFIED & HEALTHY{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}============================================================{Colors.RESET}\n")


def handle_reset_samples(storage: IrisStorage):
    """Restores baseline sample records."""
    confirm = input(f"{Colors.YELLOW}Restore 45 baseline sample records? (y/N): {Colors.RESET}").strip().lower()
    if confirm == "y":
        from generate_samples import generate_sample_data
        generate_sample_data()
        print(f"{Colors.GREEN}✓ Sample records successfully restored.{Colors.RESET}")


def main_menu():
    """Main interactive navigation dispatch loop."""
    storage = IrisStorage()
    # Check if empty; if so, populate baseline samples
    if len(storage.load_all_records()) == 0:
        from generate_samples import generate_sample_data
        generate_sample_data()

    while True:
        print_banner()
        print(f"{Colors.BOLD}Main Navigation Menu:{Colors.RESET}")
        print(f"  {Colors.BRIGHT_GREEN}[1]{Colors.RESET} Classify New Iris Specimen (Interactive Guided Input)")
        print(f"  {Colors.BRIGHT_CYAN}[2]{Colors.RESET} View Stored Classification Records (Paginated Table)")
        print(f"  {Colors.BRIGHT_CYAN}[3]{Colors.RESET} Query & Filter Records (Species, Confidence, Range)")
        print(f"  {Colors.BRIGHT_CYAN}[4]{Colors.RESET} Executive Summary & Operational Analytics")
        print(f"  {Colors.BRIGHT_YELLOW}[5]{Colors.RESET} Generate & Save Visual Distribution Plots (Matplotlib)")
        print(f"  {Colors.BRIGHT_YELLOW}[6]{Colors.RESET} Export Multi-Format Reports (Markdown, Text Log, CSV)")
        print(f"  {Colors.BRIGHT_MAGENTA}[7]{Colors.RESET} Bulk Ingest Dataset from CSV (with Row-by-Row Log)")
        print(f"  {Colors.BRIGHT_WHITE}[8]{Colors.RESET} System Diagnostics & Storage Health Check")
        print(f"  {Colors.BRIGHT_WHITE}[9]{Colors.RESET} Re-seed / Reset Sample Dataset")
        print(f"  {Colors.RED}[0] Exit Application{Colors.RESET}")
        print("-" * 76)

        try:
            choice = input(f"{Colors.BOLD}Select an option (0-9): {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.YELLOW}Exiting Iris Classifier. Goodbye!{Colors.RESET}")
            break

        if choice == "1":
            handle_new_classification(storage)
        elif choice == "2":
            handle_record_view_paginated(storage)
        elif choice == "3":
            handle_search_and_query(storage)
        elif choice == "4":
            handle_executive_summary(storage)
        elif choice == "5":
            handle_generate_charts(storage)
        elif choice == "6":
            handle_export_reports(storage)
        elif choice == "7":
            handle_bulk_import(storage)
        elif choice == "8":
            handle_system_diagnostics(storage)
        elif choice == "9":
            handle_reset_samples(storage)
        elif choice == "0":
            print(f"\n{Colors.BRIGHT_GREEN}Thank you for using Iris Dataset Classifier Demo. Have a great day!{Colors.RESET}\n")
            break
        else:
            print(f"\n{Colors.RED}Invalid selection. Please choose an option from 0 to 9.{Colors.RESET}\n")

        input(f"\n{Colors.DIM}Press Enter to return to main menu...{Colors.RESET}")


if __name__ == "__main__":
    main_menu()
