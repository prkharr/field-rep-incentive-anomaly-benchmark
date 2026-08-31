# Temporal methodology

For each actual rep/product-class series, score the current observation against prior observations only. Calendar lags explicitly reindex monthly dates; missing months are not converted into zeros. Rolling residual uses prior six-period median and MAD (minimum 3 observations); EWMA uses alpha=0.3 updated AFTER scoring; seasonal residual looks up the exact month one year earlier. MAD scale has a 10%-of-baseline floor. Seasonal scores are zero until lag-12 is available and carry an availability flag, not fabricated history.

Level-shift signals use positive/negative CUSUM with drift=0.5 and require consecutive same-direction residuals. Separate trend signals compare earlier vs later portions of the PRIOR window. This is a lightweight screening detector, not a statistically calibrated change-point posterior. Earlier validation/test observations may update history for later months (rolling-origin operation); future observations never enter expectations. Same-month peer signals are retrospective after month close, not forecasts.

Poland appears only through 2018; Germany-only 2019 introduces a coverage break at the rep/product-class grain. This can create review-worthy residuals without employee behavior changing. Anomaly review must inspect the market view and coverage provenance first.

Temporal benchmark copies are injected BEFORE historical/peer feature derivation. A labeled event can therefore influence later expectations. Unlabeled follow-on flags are counted as false positives conservatively. Targets and contractual expected payouts remain fixed from the clean DEMO schedule; corrupted payouts do not rewrite the compensation plan.

Only monthly records exist. Quarter-end injections proxy end-of-period spikes; intra-month timing cannot be established. Two-month level shifts are short, and four test months provide limited temporal evidence. Support-zero anomaly types are reported as unavailable, never perfect detection.
