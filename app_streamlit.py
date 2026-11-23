"""
Iris Dataset Classifier Demo - Modern Streamlit Web Application
Provides interactive real-time prediction, visual data exploration,
executive analytics, and bulk CSV ingestion.
"""
import io
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import (
    FEATURE_CONSTRAINTS, IRIS_CLASSES, IRIS_DISPLAY_NAMES,
    RECORDS_JSON, RECORDS_CSV
)
from src.models import IrisRecord
from src.validator import IrisValidator
from src.classifier import get_classifier
from src.storage import IrisStorage
from src.query import IrisQueryEngine
from src.analytics import IrisAnalytics
from src.visualizer import IrisVisualizer
from src.exporter import IrisExporter
from bulk_importer import BulkDatasetImporter

# Page Configuration
st.set_page_config(
    page_title="Iris Classifier Demo | AI & Smart Applications",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .badge-setosa {
        background-color: #c6f6d5;
        color: #22543d;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-versicolor {
        background-color: #feebc8;
        color: #744210;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-virginica {
        background-color: #e9d8fd;
        color: #44337a;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def init_state():
    """Initializes session state and persistence."""
    if "storage" not in st.session_state:
        st.session_state.storage = IrisStorage()
        # Seed if empty
        if len(st.session_state.storage.load_all_records()) == 0:
            from generate_samples import generate_sample_data
            generate_sample_data()


init_state()
storage: IrisStorage = st.session_state.storage
classifier = get_classifier()
all_records = storage.load_all_records()
summary = IrisAnalytics.compute_summary(all_records)
status_info = IrisAnalytics.get_operational_status(summary)

# Sidebar
st.sidebar.image("https://images.unsplash.com/photo-1596704017254-9b121068fb31?w=300&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("🌸 Iris Classifier")
st.sidebar.markdown(f"**Operational State:**\n{status_info['label']}")
st.sidebar.markdown(f"*{status_info['message']}*")
st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ System Info")
metrics = classifier.get_model_metrics()
st.sidebar.text(f"Engine: {metrics['engine'][:26]}...")
st.sidebar.text(f"Test Accuracy: {metrics['test_accuracy'] * 100:.1f}%")
st.sidebar.text(f"Total Stored: {summary.total_records} records")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset to 45 Sample Records"):
    from generate_samples import generate_sample_data
    generate_sample_data()
    st.sidebar.success("Sample records restored!")
    st.rerun()

# Header
st.markdown('<div class="main-header">🌸 Iris Dataset Classifier Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Robust Scikit-Learn Engine • Defensive Validation • Dual JSON/CSV Store • Rich Analytics</div>', unsafe_allow_html=True)

# Top KPIs
col_k1, col_k2, col_k3, col_k4 = st.columns(4)
with col_k1:
    st.metric("Total Records", summary.total_records)
with col_k2:
    st.metric("Avg Confidence", f"{summary.avg_confidence * 100:.1f}%")
with col_k3:
    st.metric("Warnings Flagged", summary.warning_count)
with col_k4:
    dominant = max(summary.species_counts, key=summary.species_counts.get) if summary.total_records > 0 else "N/A"
    st.metric("Dominant Class", IRIS_DISPLAY_NAMES.get(dominant, dominant))

st.markdown("---")

# Navigation Tabs
tab_predict, tab_history, tab_analytics, tab_plots, tab_bulk, tab_export = st.tabs([
    "🎯 Real-Time Predictor",
    "📋 Record Explorer",
    "📊 Executive Summary",
    "📈 Visual Charts",
    "📥 Bulk CSV Ingestion",
    "📤 Multi-Format Exports"
])

# ----------------- TAB 1: PREDICTOR -----------------
with tab_predict:
    st.subheader("Interactive Iris Specimen Classifier")
    st.write("Adjust the morphological measurement sliders below to compute real-time classification.")

    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        sl = st.slider("Sepal Length (cm)", min_value=2.5, max_value=9.0, value=5.8, step=0.1, help="Typical range: 4.0 - 8.0 cm")
        sw = st.slider("Sepal Width (cm)", min_value=1.5, max_value=5.5, value=3.0, step=0.1, help="Typical range: 2.0 - 5.0 cm")
    with col_inp2:
        pl = st.slider("Petal Length (cm)", min_value=0.5, max_value=8.5, value=4.2, step=0.1, help="Typical range: 1.0 - 7.5 cm")
        pw = st.slider("Petal Width (cm)", min_value=0.05, max_value=3.5, value=1.3, step=0.05, help="Typical range: 0.1 - 3.0 cm")

    # Defensive validation
    is_valid, val_errors, val_warnings = IrisValidator.validate_sample_set(sl, sw, pl, pw)

    if not is_valid:
        for err in val_errors:
            st.error(f"❌ {err}")
    else:
        # Run classification
        pred_sp, conf, probs, engine_name = classifier.predict(sl, sw, pl, pw)
        pred_warnings = IrisValidator.evaluate_prediction_warnings(conf, val_warnings)

        st.markdown("### 🏆 Prediction Outcome")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(f"#### **{IRIS_DISPLAY_NAMES.get(pred_sp, pred_sp)}**")
            st.metric("Model Confidence", f"{conf * 100:.2f}%")
            st.progress(conf)

            if pred_warnings:
                st.warning("⚠️ **Botanical Warnings:**")
                for w in pred_warnings:
                    st.caption(f"• {w}")
            else:
                st.success("✓ Normal morphological proportions.")

        with res_col2:
            st.write("**Class Probability Distribution:**")
            prob_df = pd.DataFrame({
                "Species": [IRIS_DISPLAY_NAMES.get(k, k) for k in probs.keys()],
                "Probability (%)": [v * 100 for v in probs.values()]
            })
            st.bar_chart(prob_df.set_index("Species"))

        st.markdown("---")
        with st.expander("💾 Save Observation to Database"):
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                collector_note = st.text_input("Observation Notes (optional)", placeholder="e.g. Specimen found in North garden")
            with col_s2:
                st.write("")
                st.write("")
                if st.button("Save Record", type="primary"):
                    new_rec = IrisRecord.create_new(
                        sepal_length=sl,
                        sepal_width=sw,
                        petal_length=pl,
                        petal_width=pw,
                        predicted_species=pred_sp,
                        confidence=conf,
                        probabilities=probs,
                        warnings=pred_warnings,
                        source="streamlit_gui",
                        notes=collector_note
                    )
                    storage.add_record(new_rec)
                    st.success(f"✓ Observation `{new_rec.id}` saved to local persistence stores!")
                    st.rerun()

# ----------------- TAB 2: HISTORY & EXPLORER -----------------
with tab_history:
    st.subheader("Stored Classification Database")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        search_kw = st.text_input("🔍 Search Keyword", placeholder="ID, note, species...")
    with col_f2:
        species_filter = st.selectbox("Species Filter", ["All"] + IRIS_CLASSES)
    with col_f3:
        min_conf_filter = st.slider("Min Confidence (%)", 0, 100, 0) / 100.0
    with col_f4:
        warnings_only = st.checkbox("Only records with warnings")

    filtered_records = IrisQueryEngine.search_and_filter(
        all_records,
        keyword=search_kw if search_kw else None,
        species=species_filter.lower(),
        min_confidence=min_conf_filter,
        only_with_warnings=warnings_only
    )

    st.write(f"Showing **{len(filtered_records)}** of **{len(all_records)}** records:")

    if filtered_records:
        table_data = []
        for r in filtered_records:
            table_data.append({
                "ID": r.id,
                "Timestamp": r.timestamp[:19].replace("T", " "),
                "Sepal L": r.sepal_length,
                "Sepal W": r.sepal_width,
                "Petal L": r.petal_length,
                "Petal W": r.petal_width,
                "Predicted Species": r.predicted_species,
                "Confidence": f"{r.confidence * 100:.1f}%",
                "Warnings": "; ".join(r.warnings) if r.warnings else "None",
                "Source": r.source,
                "Notes": r.notes
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No records matched the current search criteria.")

# ----------------- TAB 3: EXECUTIVE SUMMARY -----------------
with tab_analytics:
    st.subheader("Executive Operational Summary")
    st.markdown(f"**Status:** {status_info['label']} — *{status_info['message']}*")

    st.markdown("### 1. Species Breakdown")
    cols = st.columns(3)
    for i, sp in enumerate(IRIS_CLASSES):
        cnt = summary.species_counts.get(sp, 0)
        pct = summary.species_percentages.get(sp, 0.0)
        with cols[i]:
            st.metric(IRIS_DISPLAY_NAMES.get(sp, sp), f"{cnt} samples", f"{pct:.1f}% share")

    st.markdown("### 2. Morphological Statistics (cm)")
    stat_rows = []
    for feat, stats in summary.feature_statistics.items():
        stat_rows.append({
            "Feature": feat.replace("_", " ").title(),
            "Mean": f"{stats['mean']:.2f}",
            "Median": f"{stats['median']:.2f}",
            "Std Dev": f"{stats['std']:.2f}",
            "Min": f"{stats['min']:.2f}",
            "Max": f"{stats['max']:.2f}"
        })
    st.table(pd.DataFrame(stat_rows))

# ----------------- TAB 4: VISUAL CHARTS -----------------
with tab_plots:
    st.subheader("Visual Distribution & Trend Charts")
    if st.button("🔄 Regenerate All Visual Charts"):
        IrisVisualizer.generate_all_charts(all_records, summary)
        st.success("Charts regenerated successfully!")

    charts = IrisVisualizer.generate_all_charts(all_records, summary)

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.image(str(charts["species_distribution"]), caption="Species Distribution & Share", use_container_width=True)
        st.image(str(charts["feature_distributions"]), caption="Morphological Feature Histograms", use_container_width=True)
    with col_ch2:
        st.image(str(charts["feature_scatter"]), caption="Sepal vs. Petal Scatter with Decision Regions", use_container_width=True)
        st.image(str(charts["confidence_trend"]), caption="Classification Confidence Trend Across Observations", use_container_width=True)

# ----------------- TAB 5: BULK CSV INGESTION -----------------
with tab_bulk:
    st.subheader("Bulk Dataset Ingestion")
    st.write("Upload a legacy or external CSV dataset with Iris measurements for automatic format validation and batch classification.")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded_file is not None:
        # Save temp file for processor
        temp_csv_path = Path("temp_bulk_upload.csv")
        temp_csv_path.write_bytes(uploaded_file.getvalue())

        summary_bulk = BulkDatasetImporter.import_csv(temp_csv_path, persist_to_storage=False)
        temp_csv_path.unlink(missing_ok=True)

        st.markdown(f"**Total Scanned Rows:** {summary_bulk.total_rows}")
        st.markdown(f"**Valid Rows:** {summary_bulk.imported_count} | **Skipped Rows:** {summary_bulk.skipped_count}")

        if summary_bulk.errors:
            st.error("⚠️ **Validation / Schema Errors Encountered:**")
            for err in summary_bulk.errors[:10]:
                st.caption(f"• Row {err['row']}: {err['error']}")

        if summary_bulk.imported_records:
            st.write("**Preview of Valid Ingested Records:**")
            preview_data = [
                {
                    "ID": r.id,
                    "Sepal": f"{r.sepal_length}x{r.sepal_width}",
                    "Petal": f"{r.petal_length}x{r.petal_width}",
                    "Predicted": r.predicted_species,
                    "Confidence": f"{r.confidence * 100:.1f}%",
                    "Warnings": "; ".join(r.warnings)
                }
                for r in summary_bulk.imported_records[:10]
            ]
            st.dataframe(pd.DataFrame(preview_data))

            if st.button("Commit and Persist Valid Records to Database", type="primary"):
                storage.add_records_bulk(summary_bulk.imported_records)
                st.success(f"Successfully committed {len(summary_bulk.imported_records)} records to database!")
                st.rerun()

# ----------------- TAB 6: EXPORTS -----------------
with tab_export:
    st.subheader("Multi-Format Summary & Audit Exports")
    st.write("Generate and download structured reports in Markdown, Plain-Text Audit Logs, or Standardized CSV.")

    charts_dict = IrisVisualizer.generate_all_charts(all_records, summary)
    md_file = IrisExporter.export_markdown_report(all_records, summary, charts_dict)
    log_file = IrisExporter.export_audit_log(all_records)
    csv_file = IrisExporter.export_standard_csv(all_records)

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown("#### 📄 Markdown Report")
        st.caption("Executive summary with KPI tables, species share, and linked charts.")
        with open(md_file, "r", encoding="utf-8") as f:
            st.download_button("Download summary_report.md", f.read(), file_name="summary_report.md", mime="text/markdown")

    with col_e2:
        st.markdown("#### 📝 Audit Log")
        st.caption("Plain-text chronological activity log with parameter audit trails.")
        with open(log_file, "r", encoding="utf-8") as f:
            st.download_button("Download audit_log.txt", f.read(), file_name="audit_log.txt", mime="text/plain")

    with col_e3:
        st.markdown("#### 📊 Standardized CSV")
        st.caption("Complete tabular dataset export formatted for external data pipelines.")
        with open(csv_file, "r", encoding="utf-8") as f:
            st.download_button("Download export_records.csv", f.read(), file_name="export_records.csv", mime="text/csv")
