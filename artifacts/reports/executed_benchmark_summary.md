# Executed benchmark summary

## Data provenance

- Source type: **fallback_synthetic**
- Fallback used: **Yes**
- Source path: `data/synthetic/fallback_pharma_sales.csv`
- Analytical grain: **Field rep × product × territory × month**
- Analytical observations: **4,161**
- Synthetic reps: **36**
- Injected anomalies: **250 (6.01%)**

## Weighted conclusions

- Best segmentation model: **K-Means**
- Best anomaly-detection model: **K-Means**
- K-Means precision: **0.656**
- K-Means recall: **0.656**
- K-Means PR-AUC: **0.710**
- K-Means Lift@5%: **11.63×**

Selection uses the configured weighted framework; synthetic anomaly labels are used only for benchmark evaluation and anomaly-model comparison, never as clustering features or unsupervised tuning inputs.

## Guardrail

Flags identify observations for business review. They do not establish fraud, misconduct, or incorrect payment.
