"""
Configuration and constants for Iris Dataset Classifier Demo.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = BASE_DIR / "exports"
CHARTS_DIR = EXPORTS_DIR / "charts"

# Persistence files
RECORDS_JSON = DATA_DIR / "records.json"
RECORDS_CSV = DATA_DIR / "records.csv"
SAMPLE_DATASET_CSV = DATA_DIR / "sample_dataset.csv"
SAMPLE_DATASET_JSON = DATA_DIR / "sample_dataset.json"

# Export files
SUMMARY_REPORT_MD = EXPORTS_DIR / "summary_report.md"
AUDIT_LOG_TXT = EXPORTS_DIR / "audit_log.txt"
EXPORT_CSV = EXPORTS_DIR / "export_records.csv"

# Ensure directories exist
for folder in [DATA_DIR, EXPORTS_DIR, CHARTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Botanical domain constraints (in centimeters)
# Typical ranges based on Edgar Anderson's Iris dataset:
# Sepal Length: 4.3 to 7.9
# Sepal Width:  2.0 to 4.4
# Petal Length: 1.0 to 6.9
# Petal Width:  0.1 to 2.5
FEATURE_CONSTRAINTS = {
    "sepal_length": {
        "name": "Sepal Length (cm)",
        "min": 4.0,
        "max": 8.0,
        "extreme_min": 2.5,
        "extreme_max": 10.0,
        "unit": "cm"
    },
    "sepal_width": {
        "name": "Sepal Width (cm)",
        "min": 2.0,
        "max": 5.0,
        "extreme_min": 1.5,
        "extreme_max": 6.0,
        "unit": "cm"
    },
    "petal_length": {
        "name": "Petal Length (cm)",
        "min": 1.0,
        "max": 7.5,
        "extreme_min": 0.5,
        "extreme_max": 9.0,
        "unit": "cm"
    },
    "petal_width": {
        "name": "Petal Width (cm)",
        "min": 0.1,
        "max": 3.0,
        "extreme_min": 0.05,
        "extreme_max": 4.0,
        "unit": "cm"
    }
}

# Target Iris Classes
IRIS_CLASSES = [
    "setosa",
    "versicolor",
    "virginica"
]

IRIS_DISPLAY_NAMES = {
    "setosa": "Iris setosa",
    "versicolor": "Iris versicolor",
    "virginica": "Iris virginica"
}

# Alert Thresholds
CONFIDENCE_WARNING_THRESHOLD = 0.65  # Flag if confidence is below 65%
ANOMALY_PETAL_SEPAL_RATIO = 1.05     # Petal length rarely exceeds sepal length in Iris

# ANSI Colors for Terminal Styling
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright Foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"
