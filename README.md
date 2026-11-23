# 🌸 Iris Dataset Classifier Demo

**Project #117: Iris Dataset Classifier Demo**  
*Domain:* AI, APIs & Smart Applications | Hackathon Mentor Guide  
*Author:* Antigravity AI Implementation

---

## 📌 1. Project Context & Objectives
In academic and everyday environments, stakeholders frequently experience friction when attempting to follow a basic scikit-learn workflow on the Iris dataset and display a prediction manually or with fragmented tools. 

**Iris Dataset Classifier Demo** resolves this challenge by delivering a focused, accessible, and beginner-friendly Python application engineered to streamline data handling, validate entries defensively, provide fail-safe local persistence (JSON/CSV), compute executive summaries, and visualize distributions.

---

## ✨ 2. Features Implemented

### Core Features (Must-Have)
- **Interactive Terminal Interface (`main.py`)**: Menu-driven command prompt facilitating seamless user interactions, ANSI styling, formatted fixed-width tables, and pagination breaks.
- **Defensive Input Validation (`src/validator.py`)**: Real-time sanitization, regex checks, and botanical boundary validations (`4.0-8.0cm` Sepal Length, etc.), preventing crashes from blank, non-numeric, or negative inputs.
- **Core Business Logic Engine (`src/classifier.py`)**: Scikit-learn machine learning workflow (`LogisticRegression` + `StandardScaler`) trained on Fisher's Iris dataset with class probability distributions and confidence scores. Includes a **self-contained pure-Python centroid fallback** ensuring 100% offline uptime.
- **Local Data Persistence (`src/storage.py`)**: Automated dual synchronization to both JSON (`data/records.json`) and CSV (`data/records.csv`). Encapsulates file read operations in `try-except` blocks to handle missing or corrupted files, automatically creating clean initial templates and backing up corrupted data.
- **Record Query & Search (`src/query.py`)**: Multi-attribute filtering by species, confidence threshold, measurement ranges, and full-text keyword search with pagination.
- **Executive Summary Generation (`src/analytics.py`)**: Aggregation module computing totals, species percentage distributions, feature means/medians/standard deviations/min/max, and operational state indicators.
- **Visual Data Plotting (`src/visualizer.py`)**: Matplotlib integration generating clear trend charts and visual distributions (species donuts, feature histograms, scatter plots, and confidence trends) saved to `exports/charts/`.

### Stretch Features (Good-to-Have)
- **Multi-Format Summary Export (`src/exporter.py`)**: Generates structured Markdown (`exports/summary_report.md`), plain-text chronological audit logs (`exports/audit_log.txt`), and standardized CSV datasets (`exports/export_records.csv`).
- **Threshold Warning Indicators (`src/validator.py`)**: Automated warning flags highlighting borderline model confidence (< 65%), botanical limit outliers, and anatomical anomalies (e.g. Petal Length > Sepal Length).
- **Modern Web GUI - Streamlit (`app_streamlit.py`)**: Interactive web dashboard ("Streamline") featuring real-time prediction sliders, dynamic charts, database explorer, bulk CSV drag-and-drop, and report downloads.
- **Zero-Dependency Desktop GUI - Tkinter (`gui_desktop.py`)**: Built-in standard library GUI with multi-tab layout for instant local execution without installing third-party UI libraries.
- **Bulk Dataset Import (`bulk_importer.py`)**: Automated ingestion of external CSV datasets with detailed row-by-row error logs.
- **Realistic 45-Sample Seed Dataset (`data/sample_dataset.csv`, `data/sample_dataset.json`)**: Realistic dataset with normal, boundary, and edge conditions.

---

## 🏗️ 3. Project Structure

```
iris_Dataset_Classifier/
│
├── data/
│   ├── records.json              # Primary JSON local persistence store
│   ├── records.csv               # Synchronized CSV local persistence store
│   ├── sample_dataset.csv        # 45 realistic test records (normal, boundary, edge)
│   └── sample_dataset.json       # JSON format of sample dataset
│
├── exports/
│   ├── charts/                   # Matplotlib plots (species, scatter, histograms, trends)
│   ├── summary_report.md         # Markdown executive summary
│   ├── audit_log.txt             # Plain-text audit & action log
│   └── export_records.csv        # Standardized CSV export
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Botanical domain constraints, thresholds, paths, ANSI colors
│   ├── models.py                 # Dataclasses (IrisRecord, ValidationResult, SummaryMetrics)
│   ├── validator.py              # Defensive sanitization, range checks, threshold warnings
│   ├── classifier.py             # Scikit-learn ML engine + pure-Python mathematical fallback
│   ├── storage.py                # Dual JSON/CSV persistence & auto-healing corruption recovery
│   ├── query.py                  # Multi-criteria filtering, keyword search & pagination
│   ├── analytics.py              # Statistical aggregations & executive summary generation
│   ├── visualizer.py             # Matplotlib visual distribution & trend chart generator
│   └── exporter.py               # Markdown, plain-text audit log, and CSV exporter
│
├── main.py                       # Core interactive Terminal Interface (ANSI formatted)
├── app_streamlit.py              # Modern Streamlit Web Application ("Streamline")
├── gui_desktop.py                # Zero-dependency Tkinter desktop GUI
├── bulk_importer.py              # Bulk dataset ingestion engine with row-by-row logger
├── generate_samples.py           # Sample dataset generator script
├── requirements.txt              # Project dependencies
├── tests/
│   └── test_classifier.py        # Automated test suite (100% passing)
└── README.md                     # Project documentation & user guide
```

---

## 🚀 4. Quickstart Guide

### Prerequisites
Python 3.10+ (tested on Python 3.14). Install dependencies via pip:
```bash
pip install -r requirements.txt
```

---

### Option A: Interactive Terminal Interface (CLI)
To run the terminal application with ANSI formatting, paginated tables, and guided workflows:
```bash
python main.py
```

**Main Menu Options:**
- `[1]` Classify New Iris Specimen (Interactive guided input with real-time validation)
- `[2]` View Stored Classification Records (Paginated table)
- `[3]` Query & Filter Records (Species, confidence, range)
- `[4]` Executive Summary & Operational Analytics
- `[5]` Generate & Save Visual Distribution Plots (Matplotlib)
- `[6]` Export Multi-Format Reports (Markdown, Text Log, CSV)
- `[7]` Bulk Ingest Dataset from CSV (with row-by-row error log)
- `[8]` System Diagnostics & Storage Health Check
- `[9]` Re-seed / Reset Sample Dataset (45 samples)
- `[0]` Exit

---

### Option B: Modern Streamlit Web GUI ("Streamline")
To run the modern web interface:
```bash
streamlit run app_streamlit.py
```
*Features:*
- Real-time prediction sliders with live probability bar charts and confidence gauges.
- Interactive searchable record explorer with category filters.
- Live KPI cards and executive summary statistics.
- Visual chart gallery embedded dynamically.
- Drag-and-drop CSV dataset ingestion with instant error log.
- 1-click downloads for Markdown, Audit Log, and CSV files.

---

### Option C: Zero-Dependency Desktop GUI (Tkinter)
To run the desktop GUI without needing a browser or web server:
```bash
python gui_desktop.py
```

---

### Option D: Bulk Dataset Ingestion via CLI
To ingest any external CSV file and generate a row-by-row error audit:
```bash
python -c "from bulk_importer import BulkDatasetImporter; print(BulkDatasetImporter.import_csv('data/sample_dataset.csv'))"
```

---

## 🧪 5. Running Automated Tests
Run the comprehensive unit test suite:
```bash
python -m unittest discover tests -v
```
All 9 test cases covering validation, machine learning, fallback centroid engine, persistence, corruption recovery, query engine, analytics, and visual plotting will execute and pass.

---

## 🌟 6. Hackathon Demo Walkthrough (Section 9 Compliance)

1. **Launch**: Execute `python main.py` to display the branded welcome screen.
2. **Defensive Validation Demo**:
   - Select `[1]` to classify a new specimen.
   - Intentionally input invalid values (e.g. empty string, `"abc"`, negative numbers `-2.5`, or extreme values `99.0`). Observe how the validator catches errors and prompts for re-entry without crashing.
   - Enter standard values (e.g. `Sepal: 5.1 x 3.5`, `Petal: 1.4 x 0.2`). Observe instant classification: *Iris setosa* (98.1% confidence), full class probability breakdown, and automatic saving.
3. **Analytics & Executive Summary**:
   - Select `[4]` to view KPI cards, species percentages, descriptive statistics (mean, median, std dev), and the system operational health indicator.
4. **Visual Chart Generation & Report Export**:
   - Select `[5]` and `[6]` to generate high-resolution Matplotlib charts and multi-format reports (`summary_report.md`, `audit_log.txt`, `export_records.csv`).
5. **Persistence Verification**:
   - Exit the app (`[0]`) and re-launch `python main.py`. Select `[2]` to verify that all records survived program termination.
