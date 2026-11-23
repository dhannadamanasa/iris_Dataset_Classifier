"""
Script to generate realistic test dataset (45 records) containing normal,
boundary, and edge-case samples for demo verification.
"""
from datetime import datetime, timedelta
from pathlib import Path
from src.classifier import get_classifier
from src.validator import IrisValidator
from src.models import IrisRecord
from src.storage import IrisStorage
from src.config import SAMPLE_DATASET_CSV, SAMPLE_DATASET_JSON, RECORDS_JSON, RECORDS_CSV


def generate_sample_data():
    classifier = get_classifier()
    
    # 45 varied realistic test cases across all 3 classes:
    # includes normal, boundary, and edge conditions
    raw_samples = [
        # Setosa - Standard & Typical
        (5.1, 3.5, 1.4, 0.2, "Standard specimen from Fisher study", "setosa"),
        (4.9, 3.0, 1.4, 0.2, "Typical small blossom", "setosa"),
        (4.7, 3.2, 1.3, 0.2, "Compact petal geometry", "setosa"),
        (4.6, 3.1, 1.5, 0.2, "Early spring collection", "setosa"),
        (5.0, 3.6, 1.4, 0.2, "Healthy garden observation", "setosa"),
        (5.4, 3.9, 1.7, 0.4, "Robust flower, broad sepals", "setosa"),
        (4.6, 3.4, 1.4, 0.3, "Alpine meadow sample", "setosa"),
        (5.0, 3.4, 1.5, 0.2, "Greenhouse cultivated", "setosa"),
        (4.4, 2.9, 1.4, 0.2, "Boundary: Lower bound sepal length", "setosa"),
        (4.9, 3.1, 1.5, 0.1, "Boundary: Extreme lower petal width", "setosa"),
        (5.4, 3.7, 1.5, 0.2, "Woodland clearing specimen", "setosa"),
        (4.8, 3.4, 1.6, 0.2, "Moist soil substrate", "setosa"),
        (4.8, 3.0, 1.4, 0.1, "Shaded forest floor", "setosa"),
        (5.7, 4.4, 1.5, 0.4, "Boundary: Upper sepal width outlier", "setosa"),
        (5.2, 4.1, 1.5, 0.1, "Edge: High sepal width to length ratio", "setosa"),

        # Versicolor - Standard & Boundary
        (7.0, 3.2, 4.7, 1.4, "Typical robust versicolor", "versicolor"),
        (6.4, 3.2, 4.5, 1.5, "Standard riverbank variety", "versicolor"),
        (6.9, 3.1, 4.9, 1.5, "Tall stem specimen", "versicolor"),
        (5.5, 2.3, 4.0, 1.3, "Boundary: Low sepal width", "versicolor"),
        (6.5, 2.8, 4.6, 1.5, "Meadow marsh sample", "versicolor"),
        (5.7, 2.8, 4.5, 1.3, "Suburban wetland park", "versicolor"),
        (6.3, 3.3, 4.7, 1.6, "Healthy adult bloom", "versicolor"),
        (4.9, 2.4, 3.3, 1.0, "Boundary: Dwarf versicolor specimen", "versicolor"),
        (6.6, 2.9, 4.6, 1.3, "Late summer field record", "versicolor"),
        (5.2, 2.7, 3.9, 1.4, "Boundary: Near setosa-versicolor transition", "versicolor"),
        (5.0, 2.0, 3.5, 1.0, "Boundary: Hard minimum sepal width", "versicolor"),
        (5.9, 3.0, 4.2, 1.5, "Botanical conservatory sample", "versicolor"),
        (6.0, 2.2, 4.0, 1.0, "Dry meadow boundary specimen", "versicolor"),
        (6.1, 2.9, 4.7, 1.4, "Lakeshore collection", "versicolor"),
        (5.6, 2.9, 3.6, 1.3, "Borderline confidence test case", "versicolor"),

        # Virginica - Standard, Boundary & Edge
        (6.3, 3.3, 6.0, 2.5, "Standard mature virginica", "virginica"),
        (5.8, 2.7, 5.1, 1.9, "Borderline versicolor/virginica transition", "virginica"),
        (7.1, 3.0, 5.9, 2.1, "Large wetland wild bloom", "virginica"),
        (6.3, 2.9, 5.6, 1.8, "Forest trail border", "virginica"),
        (6.5, 3.0, 5.8, 2.2, "Rich loamy soil sample", "virginica"),
        (7.6, 3.0, 6.6, 2.1, "Boundary: High sepal and petal lengths", "virginica"),
        (4.9, 2.5, 4.5, 1.7, "Boundary: Unusually compact virginica", "virginica"),
        (7.3, 2.9, 6.3, 1.8, "Coastal plain giant bloom", "virginica"),
        (6.7, 2.5, 5.8, 1.8, "Boundary: Low sepal width for virginica", "virginica"),
        (7.2, 3.6, 6.1, 2.5, "Vigorous specimen in full bloom", "virginica"),
        (6.5, 3.2, 5.1, 2.0, "Late autumn survivor", "virginica"),
        (6.4, 2.7, 5.3, 1.9, "Pond margin specimen", "virginica"),
        (6.8, 3.0, 5.5, 2.1, "Nature reserve collection", "virginica"),
        (7.7, 3.8, 6.7, 2.2, "Boundary: Near maximum standard dimensions", "virginica"),
        (6.0, 3.0, 4.8, 1.8, "Edge: Ambiguous boundary between versicolor and virginica", "virginica"),
    ]

    base_time = datetime.now() - timedelta(days=15)
    records = []

    for i, (sl, sw, pl, pw, note, _) in enumerate(raw_samples):
        # Validate sample set & collect warnings
        is_valid, errors, warnings = IrisValidator.validate_sample_set(sl, sw, pl, pw)
        pred_sp, conf, probs, _ = classifier.predict(sl, sw, pl, pw)
        warnings = IrisValidator.evaluate_prediction_warnings(conf, warnings)

        record_time = (base_time + timedelta(hours=i * 8)).isoformat()
        rec = IrisRecord.create_new(
            sepal_length=sl,
            sepal_width=sw,
            petal_length=pl,
            petal_width=pw,
            predicted_species=pred_sp,
            confidence=conf,
            probabilities=probs,
            warnings=warnings,
            source="sample_dataset",
            notes=note,
            record_id=f"IRIS-SAMPLE-{i+1:03d}",
            timestamp=record_time
        )
        records.append(rec)

    # Save to sample_dataset files
    sample_storage = IrisStorage(SAMPLE_DATASET_JSON, SAMPLE_DATASET_CSV)
    sample_storage.save_all_records(records)
    print(f"Generated {len(records)} sample records in {SAMPLE_DATASET_CSV} and {SAMPLE_DATASET_JSON}")

    # Also seed active persistence storage (records.json / records.csv)
    active_storage = IrisStorage(RECORDS_JSON, RECORDS_CSV)
    active_storage.save_all_records(records)
    print(f"Initialized active store with {len(records)} records in {RECORDS_JSON} and {RECORDS_CSV}")


if __name__ == "__main__":
    generate_sample_data()
