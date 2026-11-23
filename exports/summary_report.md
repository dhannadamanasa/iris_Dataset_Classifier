# Iris Dataset Classifier - Executive Summary Report
**Generated At:** 2026-11-23 17:31:50  
**Operational State:** 🟢 Optimal Operational State  
*High classification confidence with low warning rate.*

---

## 1. Key Performance Indicators

| Metric | Value |
| :--- | :--- |
| **Total Records Classified** | 45 |
| **Average Model Confidence** | 88.43% |
| **Records with Domain Warnings** | 3 (6.7%) |
| **Observation Window (Oldest)** | 2026-11-08T16:49:43.507321 |
| **Observation Window (Latest)** | 2026-11-23T08:49:43.507321 |

## 2. Species Classification Breakdown

| Species | Common / Display Name | Sample Count | Share (%) |
| :--- | :--- | :--- | :--- |
| `setosa` | Iris setosa | 15 | 33.33% |
| `versicolor` | Iris versicolor | 16 | 35.56% |
| `virginica` | Iris virginica | 14 | 31.11% |

## 3. Morphological Feature Statistics (cm)

| Feature | Mean | Median | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sepal Length** | 5.85 | 5.80 | 0.90 | 4.40 | 7.70 |
| **Sepal Width** | 3.08 | 3.00 | 0.48 | 2.00 | 4.40 |
| **Petal Length** | 3.80 | 4.50 | 1.83 | 1.30 | 6.70 |
| **Petal Width** | 1.19 | 1.40 | 0.78 | 0.10 | 2.50 |

## 4. Anomaly & Warning Log (Recent Flags)

| Record ID | Timestamp | Species | Confidence | Warnings Triggered |
| :--- | :--- | :--- | :--- | :--- |
| `IRIS-SAMPLE-022` | 2026-11-15T16:49:43 | versicolor | 62.1% | Low Confidence Flag: Model certainty is 62.1% (below 65% threshold). Specimen lies near classification decision boundary. |
| `IRIS-SAMPLE-037` | 2026-11-20T16:49:43 | versicolor | 57.4% | Low Confidence Flag: Model certainty is 57.4% (below 65% threshold). Specimen lies near classification decision boundary. |
| `IRIS-SAMPLE-045` | 2026-11-23T08:49:43 | virginica | 59.5% | Low Confidence Flag: Model certainty is 59.5% (below 65% threshold). Specimen lies near classification decision boundary. |

## 5. Visual Trends & Distributions

### Species Distribution
![Species Distribution](C:/Users/Admin/Downloads/iris_Dataset_Classifier/exports/charts/species_distribution.png)

### Feature Scatter
![Feature Scatter](C:/Users/Admin/Downloads/iris_Dataset_Classifier/exports/charts/feature_scatter.png)

### Feature Distributions
![Feature Distributions](C:/Users/Admin/Downloads/iris_Dataset_Classifier/exports/charts/feature_distributions.png)

### Confidence Trend
![Confidence Trend](C:/Users/Admin/Downloads/iris_Dataset_Classifier/exports/charts/confidence_trend.png)

