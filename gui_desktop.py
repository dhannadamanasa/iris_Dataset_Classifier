"""
Iris Dataset Classifier Demo - Desktop GUI Application
Built using standard library Tkinter (zero external dependencies).
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from src.config import IRIS_CLASSES, IRIS_DISPLAY_NAMES
from src.models import IrisRecord
from src.validator import IrisValidator
from src.classifier import get_classifier
from src.storage import IrisStorage
from src.query import IrisQueryEngine
from src.analytics import IrisAnalytics
from src.visualizer import IrisVisualizer
from src.exporter import IrisExporter
from bulk_importer import BulkDatasetImporter


class IrisDesktopApp(tk.Tk):
    """Zero-dependency Tkinter Desktop GUI for Iris Classification."""

    def __init__(self):
        super().__init__()
        self.title("🌸 Iris Dataset Classifier Demo | Desktop GUI")
        self.geometry("950x700")
        self.minsize(800, 600)

        self.storage = IrisStorage()
        self.classifier = get_classifier()

        # Seed sample data if empty
        if len(self.storage.load_all_records()) == 0:
            from generate_samples import generate_sample_data
            generate_sample_data()

        self._apply_styles()
        self._build_ui()
        self.refresh_records()

    def _apply_styles(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[10, 5])
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1a365d")
        self.style.configure("Card.TFrame", background="#f7fafc", relief="solid", borderwidth=1)

    def _build_ui(self):
        # Header banner
        header_frame = ttk.Frame(self, padding=10)
        header_frame.pack(fill="x")
        title_lbl = ttk.Label(header_frame, text="🌸 Iris Dataset Classifier Demo", style="Header.TLabel")
        title_lbl.pack(anchor="w")
        sub_lbl = ttk.Label(header_frame, text="Scikit-Learn Classification • Defensive Validation • Dual Store", font=("Segoe UI", 9))
        sub_lbl.pack(anchor="w")

        # Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_classify = ttk.Frame(self.notebook, padding=15)
        self.tab_history = ttk.Frame(self.notebook, padding=15)
        self.tab_summary = ttk.Frame(self.notebook, padding=15)
        self.tab_export = ttk.Frame(self.notebook, padding=15)

        self.notebook.add(self.tab_classify, text="  🎯 Classify Specimen  ")
        self.notebook.add(self.tab_history, text="  📋 Stored Records  ")
        self.notebook.add(self.tab_summary, text="  📊 Executive Summary  ")
        self.notebook.add(self.tab_export, text="  📤 Exports & Charts  ")

        self._build_classify_tab()
        self._build_history_tab()
        self._build_summary_tab()
        self._build_export_tab()

    def _build_classify_tab(self):
        frame = self.tab_classify

        # Input fields
        inp_frame = ttk.LabelFrame(frame, text="Specimen Measurements (cm)", padding=15)
        inp_frame.pack(fill="x", pady=5)

        self.var_sl = tk.DoubleVar(value=5.8)
        self.var_sw = tk.DoubleVar(value=3.0)
        self.var_pl = tk.DoubleVar(value=4.2)
        self.var_pw = tk.DoubleVar(value=1.3)

        fields = [
            ("Sepal Length (cm):", self.var_sl, 2.5, 9.0),
            ("Sepal Width (cm):", self.var_sw, 1.5, 5.5),
            ("Petal Length (cm):", self.var_pl, 0.5, 8.5),
            ("Petal Width (cm):", self.var_pw, 0.05, 3.5),
        ]

        for i, (label, var, min_v, max_v) in enumerate(fields):
            row_f = ttk.Frame(inp_frame)
            row_f.pack(fill="x", pady=4)
            ttk.Label(row_f, text=label, width=20).pack(side="left")
            scale = ttk.Scale(row_f, from_=min_v, to=max_v, variable=var, orient="horizontal", command=lambda e: self._on_input_change())
            scale.pack(side="left", fill="x", expand=True, padx=10)
            val_lbl = ttk.Label(row_f, textvariable=var, width=6)
            val_lbl.pack(side="left")

        # Note entry
        note_f = ttk.Frame(inp_frame)
        note_f.pack(fill="x", pady=6)
        ttk.Label(note_f, text="Collector Note:", width=20).pack(side="left")
        self.entry_note = ttk.Entry(note_f)
        self.entry_note.pack(side="left", fill="x", expand=True)

        # Output card
        self.res_frame = ttk.LabelFrame(frame, text="Prediction & Validation", padding=15)
        self.res_frame.pack(fill="both", expand=True, pady=10)

        self.lbl_prediction = ttk.Label(self.res_frame, text="Predicted Species: --", font=("Segoe UI", 12, "bold"))
        self.lbl_prediction.pack(anchor="w", pady=2)

        self.lbl_confidence = ttk.Label(self.res_frame, text="Confidence: --", font=("Segoe UI", 10))
        self.lbl_confidence.pack(anchor="w", pady=2)

        self.lbl_probs = ttk.Label(self.res_frame, text="Probabilities: --", font=("Segoe UI", 9))
        self.lbl_probs.pack(anchor="w", pady=2)

        self.lbl_warnings = ttk.Label(self.res_frame, text="Warnings: None", font=("Segoe UI", 9), foreground="#c53030")
        self.lbl_warnings.pack(anchor="w", pady=4)

        # Action buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="💾 Save to Database", command=self._save_current_prediction).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🔄 Reset Defaults", command=self._reset_inputs).pack(side="left", padx=5)

        self._on_input_change()

    def _on_input_change(self):
        sl = round(self.var_sl.get(), 2)
        sw = round(self.var_sw.get(), 2)
        pl = round(self.var_pl.get(), 2)
        pw = round(self.var_pw.get(), 2)

        is_valid, errors, warnings = IrisValidator.validate_sample_set(sl, sw, pl, pw)
        if not is_valid:
            self.lbl_prediction.config(text=f"Validation Error: {'; '.join(errors)}", foreground="red")
            self.lbl_confidence.config(text="")
            self.lbl_probs.config(text="")
            self.lbl_warnings.config(text="")
            return

        pred_sp, conf, probs, engine = self.classifier.predict(sl, sw, pl, pw)
        warnings = IrisValidator.evaluate_prediction_warnings(conf, warnings)

        self.current_pred = {
            "sl": sl, "sw": sw, "pl": pl, "pw": pw,
            "species": pred_sp, "confidence": conf, "probs": probs, "warnings": warnings
        }

        self.lbl_prediction.config(
            text=f"Predicted Species: {IRIS_DISPLAY_NAMES.get(pred_sp, pred_sp).upper()}",
            foreground="#2b6cb0"
        )
        self.lbl_confidence.config(text=f"Model Confidence: {conf * 100:.2f}%  (Engine: {engine[:30]}...)")
        
        prob_str = " | ".join(f"{k.capitalize()}: {v*100:.1f}%" for k, v in probs.items())
        self.lbl_probs.config(text=f"Class Probabilities: [{prob_str}]")

        if warnings:
            self.lbl_warnings.config(text="⚠️ " + " | ".join(warnings), foreground="#c53030")
        else:
            self.lbl_warnings.config(text="✓ Normal morphological proportions.", foreground="#2f855a")

    def _save_current_prediction(self):
        if not hasattr(self, "current_pred"):
            return
        p = self.current_pred
        note = self.entry_note.get().strip()

        rec = IrisRecord.create_new(
            sepal_length=p["sl"],
            sepal_width=p["sw"],
            petal_length=p["pl"],
            petal_width=p["pw"],
            predicted_species=p["species"],
            confidence=p["confidence"],
            probabilities=p["probs"],
            warnings=p["warnings"],
            source="desktop_gui",
            notes=note
        )
        self.storage.add_record(rec)
        messagebox.showinfo("Saved", f"Record {rec.id} successfully saved to database!")
        self.entry_note.delete(0, tk.END)
        self.refresh_records()

    def _reset_inputs(self):
        self.var_sl.set(5.8)
        self.var_sw.set(3.0)
        self.var_pl.set(4.2)
        self.var_pw.set(1.3)
        self.entry_note.delete(0, tk.END)
        self._on_input_change()

    def _build_history_tab(self):
        frame = self.tab_history

        # Filter bar
        filter_f = ttk.Frame(frame)
        filter_f.pack(fill="x", pady=5)

        ttk.Label(filter_f, text="Search:").pack(side="left")
        self.entry_search = ttk.Entry(filter_f, width=20)
        self.entry_search.pack(side="left", padx=5)
        self.entry_search.bind("<KeyRelease>", lambda e: self.refresh_records())

        ttk.Label(filter_f, text="Species:").pack(side="left", padx=5)
        self.combo_species = ttk.Combobox(filter_f, values=["All"] + IRIS_CLASSES, state="readonly", width=12)
        self.combo_species.set("All")
        self.combo_species.pack(side="left", padx=5)
        self.combo_species.bind("<<ComboboxSelected>>", lambda e: self.refresh_records())

        ttk.Button(filter_f, text="🔄 Refresh", command=self.refresh_records).pack(side="right", padx=5)

        # Treeview Table
        columns = ("id", "timestamp", "sepal", "petal", "species", "confidence", "flags")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=16)

        self.tree.heading("id", text="Record ID")
        self.tree.heading("timestamp", text="Timestamp")
        self.tree.heading("sepal", text="Sepal (LxW)")
        self.tree.heading("petal", text="Petal (LxW)")
        self.tree.heading("species", text="Predicted")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("flags", text="Warnings")

        self.tree.column("id", width=120)
        self.tree.column("timestamp", width=150)
        self.tree.column("sepal", width=100)
        self.tree.column("petal", width=100)
        self.tree.column("species", width=110)
        self.tree.column("confidence", width=90)
        self.tree.column("flags", width=160)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def refresh_records(self):
        records = self.storage.load_all_records()
        kw = getattr(self, "entry_search", None)
        kw_val = kw.get().strip() if kw else ""
        sp = getattr(self, "combo_species", None)
        sp_val = sp.get().strip().lower() if sp else "all"

        filtered = IrisQueryEngine.search_and_filter(
            records,
            keyword=kw_val if kw_val else None,
            species=sp_val if sp_val != "all" else None
        )

        for item in self.tree.get_children():
            self.tree.delete(item)

        for r in filtered:
            warn_str = "⚠️ " + "; ".join(r.warnings) if r.warnings else "✓ Normal"
            self.tree.insert("", "end", values=(
                r.id,
                r.timestamp[:19].replace("T", " "),
                f"{r.sepal_length} x {r.sepal_width}",
                f"{r.petal_length} x {r.petal_width}",
                r.predicted_species,
                f"{r.confidence * 100:.1f}%",
                warn_str
            ))

    def _build_summary_tab(self):
        frame = self.tab_summary
        self.txt_summary = tk.Text(frame, wrap="word", font=("Consolas", 10), padx=10, pady=10)
        self.txt_summary.pack(fill="both", expand=True)

        btn_refresh = ttk.Button(frame, text="Update Executive Summary", command=self._render_summary)
        btn_refresh.pack(pady=5)
        self._render_summary()

    def _render_summary(self):
        records = self.storage.load_all_records()
        summary = IrisAnalytics.compute_summary(records)
        status = IrisAnalytics.get_operational_status(summary)

        lines = [
            "=" * 60,
            "              IRIS CLASSIFIER - EXECUTIVE SUMMARY",
            "=" * 60,
            f"Operational State : {status['label']}",
            f"Status Details    : {status['message']}",
            f"Total Records     : {summary.total_records}",
            f"Avg Confidence    : {summary.avg_confidence * 100:.2f}%",
            f"Flagged Warnings  : {summary.warning_count}",
            "",
            "--- Species Breakdown ---"
        ]
        for sp, cnt in summary.species_counts.items():
            pct = summary.species_percentages.get(sp, 0.0)
            lines.append(f"  • {IRIS_DISPLAY_NAMES.get(sp, sp):<18}: {cnt:3d} ({pct:5.1f}%)")

        lines.extend(["", "--- Morphological Statistics (cm) ---"])
        lines.append(f"  {'Feature':<16} | {'Mean':<6} | {'Median':<6} | {'Std':<6} | {'Min':<6} | {'Max':<6}")
        lines.append("  " + "-" * 52)
        for feat, st in summary.feature_statistics.items():
            lines.append(f"  {feat:<16} | {st['mean']:<6.2f} | {st['median']:<6.2f} | {st['std']:<6.2f} | {st['min']:<6.2f} | {st['max']:<6.2f}")

        self.txt_summary.delete("1.0", tk.END)
        self.txt_summary.insert(tk.END, "\n".join(lines))

    def _build_export_tab(self):
        frame = self.tab_export
        box = ttk.LabelFrame(frame, text="Generate Reports & Charts", padding=20)
        box.pack(fill="x", pady=10)

        ttk.Button(box, text="📈 Generate & Save All Matplotlib Charts", command=self._export_charts).pack(fill="x", pady=5)
        ttk.Button(box, text="📄 Export Markdown Summary Report", command=self._export_md).pack(fill="x", pady=5)
        ttk.Button(box, text="📝 Export Plain-Text Audit Log", command=self._export_log).pack(fill="x", pady=5)
        ttk.Button(box, text="📊 Export Standardized CSV", command=self._export_csv).pack(fill="x", pady=5)
        ttk.Button(box, text="📥 Bulk Import External CSV Dataset", command=self._bulk_import_dialog).pack(fill="x", pady=5)

    def _export_charts(self):
        records = self.storage.load_all_records()
        summary = IrisAnalytics.compute_summary(records)
        charts = IrisVisualizer.generate_all_charts(records, summary)
        messagebox.showinfo("Export Success", f"Charts successfully saved to:\n{list(charts.values())[0].parent}")

    def _export_md(self):
        records = self.storage.load_all_records()
        summary = IrisAnalytics.compute_summary(records)
        charts = IrisVisualizer.generate_all_charts(records, summary)
        path = IrisExporter.export_markdown_report(records, summary, charts)
        messagebox.showinfo("Export Success", f"Markdown summary exported to:\n{path}")

    def _export_log(self):
        records = self.storage.load_all_records()
        path = IrisExporter.export_audit_log(records)
        messagebox.showinfo("Export Success", f"Audit log exported to:\n{path}")

    def _export_csv(self):
        records = self.storage.load_all_records()
        path = IrisExporter.export_standard_csv(records)
        messagebox.showinfo("Export Success", f"Standardized CSV exported to:\n{path}")

    def _bulk_import_dialog(self):
        f = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if f:
            summary = BulkDatasetImporter.import_csv(f, persist_to_storage=True)
            self.refresh_records()
            messagebox.showinfo("Bulk Import Result", f"Imported {summary.imported_count} records ({summary.skipped_count} skipped).")


if __name__ == "__main__":
    app = IrisDesktopApp()
    app.mainloop()
