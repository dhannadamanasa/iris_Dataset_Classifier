"""
Visual Data Plotting Module using Matplotlib.
Generates distribution charts, feature scatter plots, and confidence trends.
"""
from pathlib import Path
from typing import Dict, List, Optional
from src.config import CHARTS_DIR, IRIS_CLASSES, IRIS_DISPLAY_NAMES
from src.models import IrisRecord, SummaryMetrics


class IrisVisualizer:
    """Generates high-resolution Matplotlib visual charts for reporting and dashboards."""

    @staticmethod
    def _ensure_charts_dir() -> Path:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        return CHARTS_DIR

    @classmethod
    def plot_species_distribution(
        cls,
        summary: SummaryMetrics,
        save_path: Optional[Path] = None
    ) -> Path:
        """Generates species distribution bar & donut chart."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend safe for headless/server/CLI
        import matplotlib.pyplot as plt

        charts_dir = cls._ensure_charts_dir()
        out_path = save_path or (charts_dir / "species_distribution.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Iris Classifications - Species Distribution", fontsize=14, fontweight="bold")

        species = [IRIS_DISPLAY_NAMES.get(sp, sp.capitalize()) for sp in IRIS_CLASSES]
        counts = [summary.species_counts.get(sp, 0) for sp in IRIS_CLASSES]
        colors = ["#2b5c8f", "#38a169", "#d69e2e"]

        # Bar Chart
        bars = ax1.bar(species, counts, color=colors, edgecolor="black", alpha=0.85)
        ax1.set_title("Observation Count by Species", fontsize=11)
        ax1.set_ylabel("Count")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")

        # Donut Chart
        if sum(counts) > 0:
            wedges, texts, autotexts = ax2.pie(
                counts,
                labels=species,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                wedgeprops=dict(width=0.4, edgecolor="white")
            )
            for at in autotexts:
                at.set_fontweight("bold")
            ax2.set_title("Species Share Percentage", fontsize=11)
        else:
            ax2.text(0.5, 0.5, "No Data Available", ha="center", va="center")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def plot_feature_scatter(
        cls,
        records: List[IrisRecord],
        save_path: Optional[Path] = None
    ) -> Path:
        """Generates Sepal Length vs Petal Length scatter distribution."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        charts_dir = cls._ensure_charts_dir()
        out_path = save_path or (charts_dir / "feature_scatter.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Sepal Length vs. Petal Length by Predicted Species", fontsize=13, fontweight="bold")

        color_map = {
            "setosa": "#2b5c8f",
            "versicolor": "#38a169",
            "virginica": "#d69e2e"
        }
        marker_map = {
            "setosa": "o",
            "versicolor": "s",
            "virginica": "^"
        }

        if records:
            for sp in IRIS_CLASSES:
                sp_records = [r for r in records if r.predicted_species.lower() == sp]
                if sp_records:
                    x = [r.sepal_length for r in sp_records]
                    y = [r.petal_length for r in sp_records]
                    ax.scatter(
                        x, y,
                        label=IRIS_DISPLAY_NAMES.get(sp, sp),
                        color=color_map.get(sp, "gray"),
                        marker=marker_map.get(sp, "o"),
                        s=60, alpha=0.75, edgecolors="k"
                    )

            # Highlight records with warnings
            warn_records = [r for r in records if r.warnings]
            if warn_records:
                wx = [r.sepal_length for r in warn_records]
                wy = [r.petal_length for r in warn_records]
                ax.scatter(
                    wx, wy,
                    facecolors="none",
                    edgecolors="#e53e3e",
                    s=120,
                    linewidths=1.8,
                    label="Flagged Anomaly / Warning"
                )

        ax.set_xlabel("Sepal Length (cm)", fontsize=11)
        ax.set_ylabel("Petal Length (cm)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def plot_feature_distributions(
        cls,
        records: List[IrisRecord],
        save_path: Optional[Path] = None
    ) -> Path:
        """Generates 2x2 multi-panel histograms for all 4 features."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        charts_dir = cls._ensure_charts_dir()
        out_path = save_path or (charts_dir / "feature_distributions.png")

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Iris Morphological Feature Distributions", fontsize=14, fontweight="bold")

        feature_configs = [
            ("sepal_length", "Sepal Length (cm)", axs[0, 0], "#3182ce"),
            ("sepal_width", "Sepal Width (cm)", axs[0, 1], "#38a169"),
            ("petal_length", "Petal Length (cm)", axs[1, 0], "#d69e2e"),
            ("petal_width", "Petal Width (cm)", axs[1, 1], "#805ad5")
        ]

        for feat_key, feat_title, ax, color in feature_configs:
            if records:
                vals = [getattr(r, feat_key) for r in records]
                ax.hist(vals, bins=12, color=color, edgecolor="black", alpha=0.75)
                ax.set_title(feat_title, fontsize=11)
                ax.set_xlabel("cm")
                ax.set_ylabel("Frequency")
                ax.grid(axis="y", linestyle="--", alpha=0.5)
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def plot_confidence_trend(
        cls,
        records: List[IrisRecord],
        save_path: Optional[Path] = None
    ) -> Path:
        """Generates chronological model confidence trend with threshold indicator."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        charts_dir = cls._ensure_charts_dir()
        out_path = save_path or (charts_dir / "confidence_trend.png")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_title("Classification Confidence Trend Across Observations", fontsize=12, fontweight="bold")

        if records:
            sorted_records = sorted(records, key=lambda r: r.timestamp)
            indices = list(range(1, len(sorted_records) + 1))
            confs = [r.confidence * 100 for r in sorted_records]

            ax.plot(indices, confs, marker="o", markersize=4, color="#3182ce", alpha=0.8, label="Model Confidence (%)")
            ax.axhline(65.0, color="#e53e3e", linestyle="--", linewidth=1.5, label="65% Warning Threshold")
            ax.set_ylim(40, 105)
            ax.set_xlabel("Observation Sequence", fontsize=10)
            ax.set_ylabel("Confidence Score (%)", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower right")
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def generate_all_charts(
        cls,
        records: List[IrisRecord],
        summary: SummaryMetrics
    ) -> Dict[str, Path]:
        """Generates all visual plots and returns a map of chart name to file path."""
        p1 = cls.plot_species_distribution(summary)
        p2 = cls.plot_feature_scatter(records)
        p3 = cls.plot_feature_distributions(records)
        p4 = cls.plot_confidence_trend(records)
        return {
            "species_distribution": p1,
            "feature_scatter": p2,
            "feature_distributions": p3,
            "confidence_trend": p4
        }
