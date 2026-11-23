"""
Comprehensive automated test suite for Iris Dataset Classifier Demo.
Covers validation, machine learning, fallback engine, dual persistence,
corruption recovery, queries, analytics, visual plotting, and bulk ingestion.
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.config import IRIS_CLASSES
from src.models import IrisRecord, SummaryMetrics
from src.validator import IrisValidator
from src.classifier import IrisMLClassifier, get_classifier
from src.storage import IrisStorage
from src.query import IrisQueryEngine
from src.analytics import IrisAnalytics
from src.visualizer import IrisVisualizer
from src.exporter import IrisExporter
from bulk_importer import BulkDatasetImporter


class TestIrisClassifierSystem(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.test_json = self.test_dir / "test_records.json"
        self.test_csv = self.test_dir / "test_records.csv"
        self.storage = IrisStorage(self.test_json, self.test_csv)
        self.classifier = get_classifier()

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- 1. Defensive Input Validation Tests ---
    def test_validation_normal_values(self):
        res = IrisValidator.validate_feature("sepal_length", "5.1")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.cleaned_value, 5.1)
        self.assertEqual(len(res.errors), 0)

    def test_validation_blank_and_non_numeric(self):
        res_blank = IrisValidator.validate_feature("sepal_length", "   ")
        self.assertFalse(res_blank.is_valid)
        self.assertIn("blank", res_blank.errors[0])

        res_alpha = IrisValidator.validate_feature("petal_width", "invalid_abc")
        self.assertFalse(res_alpha.is_valid)

        res_neg = IrisValidator.validate_feature("petal_width", "-1.5")
        self.assertFalse(res_neg.is_valid)

    def test_validation_biological_limits_and_warnings(self):
        # Extreme value breaching biological limits
        res_extreme = IrisValidator.validate_feature("sepal_length", "25.0")
        self.assertFalse(res_extreme.is_valid)
        self.assertIn("breaches biological hard limits", res_extreme.errors[0])

        # Value within biological limits but outside typical range triggers warning
        res_warn = IrisValidator.validate_feature("sepal_length", "8.5")
        self.assertTrue(res_warn.is_valid)
        self.assertTrue(len(res_warn.warnings) > 0)

    def test_cross_feature_morphological_checks(self):
        # Anomaly: Petal length greater than sepal length
        is_valid, errors, warnings = IrisValidator.validate_sample_set(
            sepal_length=4.5,
            sepal_width=3.0,
            petal_length=6.5,
            petal_width=1.5
        )
        self.assertTrue(is_valid)
        self.assertTrue(any("Morphological Anomaly" in w for w in warnings))

    # --- 2. Machine Learning Engine & Fallback Tests ---
    def test_ml_prediction(self):
        # Test canonical Setosa sample
        pred_sp, conf, probs, engine = self.classifier.predict(5.1, 3.5, 1.4, 0.2)
        self.assertEqual(pred_sp, "setosa")
        self.assertGreater(conf, 0.80)
        self.assertIn("setosa", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

        # Test canonical Virginica sample
        pred_sp2, conf2, probs2, _ = self.classifier.predict(6.7, 3.1, 5.6, 2.4)
        self.assertEqual(pred_sp2, "virginica")
        self.assertGreater(conf2, 0.70)

    def test_pure_python_fallback_engine(self):
        fallback_clf = IrisMLClassifier()
        # Force fallback centroid mode
        fallback_clf.engine_mode = "Pure-Python Fallback (Test)"
        fallback_clf._init_fallback_centroids()

        pred_sp, conf, probs, engine = fallback_clf.predict(5.0, 3.4, 1.5, 0.2)
        self.assertEqual(pred_sp, "setosa")
        self.assertGreater(conf, 0.50)
        self.assertIn("Pure-Python Fallback", engine)

    # --- 3. Dual JSON/CSV Storage & Persistence Tests ---
    def test_storage_add_and_load(self):
        rec = IrisRecord.create_new(
            sepal_length=5.1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2,
            predicted_species="setosa",
            confidence=0.98,
            probabilities={"setosa": 0.98, "versicolor": 0.02, "virginica": 0.0},
            notes="Test record"
        )
        self.storage.add_record(rec)

        loaded = self.storage.load_all_records()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, rec.id)
        self.assertEqual(loaded[0].predicted_species, "setosa")
        self.assertEqual(loaded[0].notes, "Test record")

        # Verify CSV is also populated
        self.assertTrue(self.test_csv.exists())
        with open(self.test_csv, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("setosa", content)
            self.assertIn(rec.id, content)

    def test_storage_corruption_recovery(self):
        # Intentionally write invalid corrupted JSON
        with open(self.test_json, "w", encoding="utf-8") as f:
            f.write("CORRUPTED_NON_JSON_DATA{{{###")

        # Attempting to load should detect corruption, create a backup, and heal with clean template
        records = self.storage.load_all_records()
        self.assertEqual(records, [])
        # Check that corrupted backup was created
        backups = list(self.test_dir.glob("records_corrupted_*.bak"))
        self.assertTrue(len(backups) > 0)

    # --- 4. Query and Search Engine Tests ---
    def test_query_filtering(self):
        r1 = IrisRecord.create_new(5.1, 3.5, 1.4, 0.2, "setosa", 0.95, notes="Spring bloom")
        r2 = IrisRecord.create_new(6.5, 3.0, 5.5, 1.8, "virginica", 0.88, notes="Autumn marsh")
        r3 = IrisRecord.create_new(5.8, 2.7, 4.1, 1.2, "versicolor", 0.60, warnings=["Low confidence flag"])
        
        all_recs = [r1, r2, r3]

        # Filter by species
        res_sp = IrisQueryEngine.search_and_filter(all_recs, species="virginica")
        self.assertEqual(len(res_sp), 1)
        self.assertEqual(res_sp[0].id, r2.id)

        # Filter by keyword
        res_kw = IrisQueryEngine.search_and_filter(all_recs, keyword="spring")
        self.assertEqual(len(res_kw), 1)
        self.assertEqual(res_kw[0].id, r1.id)

        # Filter by confidence
        res_conf = IrisQueryEngine.search_and_filter(all_recs, min_confidence=0.85)
        self.assertEqual(len(res_conf), 2)

        # Filter by warnings
        res_warn = IrisQueryEngine.search_and_filter(all_recs, only_with_warnings=True)
        self.assertEqual(len(res_warn), 1)
        self.assertEqual(res_warn[0].id, r3.id)

    # --- 5. Executive Summary & Analytics Tests ---
    def test_analytics_summary(self):
        r1 = IrisRecord.create_new(5.0, 3.5, 1.5, 0.2, "setosa", 0.90)
        r2 = IrisRecord.create_new(6.0, 3.0, 4.5, 1.5, "versicolor", 0.80)
        summary = IrisAnalytics.compute_summary([r1, r2])

        self.assertEqual(summary.total_records, 2)
        self.assertEqual(summary.species_counts["setosa"], 1)
        self.assertEqual(summary.species_counts["versicolor"], 1)
        self.assertEqual(summary.species_percentages["setosa"], 50.0)
        self.assertAlmostEqual(summary.avg_confidence, 0.85, places=2)
        self.assertEqual(summary.feature_statistics["sepal_length"]["mean"], 5.5)

        status = IrisAnalytics.get_operational_status(summary)
        self.assertIn("status", status)

    # --- 6. Visualizer & Exporter Tests ---
    def test_visualizer_and_export_generation(self):
        r1 = IrisRecord.create_new(5.1, 3.5, 1.4, 0.2, "setosa", 0.95)
        r2 = IrisRecord.create_new(6.5, 3.0, 5.5, 1.8, "virginica", 0.88)
        records = [r1, r2]
        summary = IrisAnalytics.compute_summary(records)

        # Visual plots
        charts = IrisVisualizer.generate_all_charts(records, summary)
        for name, p in charts.items():
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 0)

        # Multi-format exports
        md_path = self.test_dir / "test_report.md"
        log_path = self.test_dir / "test_log.txt"
        csv_path = self.test_dir / "test_export.csv"

        IrisExporter.export_markdown_report(records, summary, charts, output_path=md_path)
        IrisExporter.export_audit_log(records, output_path=log_path)
        IrisExporter.export_standard_csv(records, output_path=csv_path)

        self.assertTrue(md_path.exists())
        self.assertTrue(log_path.exists())
        self.assertTrue(csv_path.exists())

    # --- 7. Bulk Dataset Ingestion Tests ---
    def test_bulk_csv_importer(self):
        # Create a test CSV with 2 valid rows and 1 invalid row
        csv_content = (
            "sepal_length,sepal_width,petal_length,petal_width,notes\n"
            "5.1,3.5,1.4,0.2,Valid setosa\n"
            "6.2,2.9,4.3,1.3,Valid versicolor\n"
            "invalid_text,3.0,1.2,0.2,Corrupted row\n"
            "5.0,-2.0,1.5,0.2,Negative width row\n"
        )
        test_import_csv = self.test_dir / "test_bulk.csv"
        test_import_csv.write_text(csv_content, encoding="utf-8")

        summary = BulkDatasetImporter.import_csv(test_import_csv, persist_to_storage=False)
        self.assertEqual(summary.imported_count, 2)
        self.assertEqual(summary.skipped_count, 2)
        self.assertEqual(len(summary.errors), 2)


if __name__ == "__main__":
    unittest.main()
