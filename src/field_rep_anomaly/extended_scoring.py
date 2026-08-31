"""Shared ranking, calibration and transparent ensemble contracts."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .evaluation import classification_metrics, ranking_metrics, top_fraction_flags


class PercentileCalibrator:
    """TRAIN empirical CDF with smooth, bounded tails to avoid tied extreme queues.

    Interior values use ECDF plotting positions. Extrapolation within the remaining
    endpoint probability mass preserves ordering beyond training extremes.
    """
    def fit(self, values):
        self.reference_ = np.sort(np.asarray(values, float))
        self.tail_scale_ = max(float(np.std(self.reference_)), float(np.ptp(self.reference_))*.1, 1e-8)
        return self

    def transform(self, values):
        x = np.asarray(values, float)
        r = self.reference_
        n = len(r)
        unique, counts = np.unique(r, return_counts=True)
        positions = (np.cumsum(counts) - counts*.5)/(n+1)
        result = np.interp(x, unique, positions)
        high, low = x > r[-1], x < r[0]
        result[high] = 1-(1-positions[-1])/(1+(x[high]-r[-1])/self.tail_scale_)
        result[low] = positions[0]/(1+(r[0]-x[low])/self.tail_scale_)
        return np.clip(result, 0, 1)


def metrics(truth, raw, fraction=.05):
    flags = top_fraction_flags(np.asarray(raw), fraction)
    a = classification_metrics(np.asarray(truth), np.asarray(raw), flags)
    b = ranking_metrics(np.asarray(truth), np.asarray(raw))
    out = {k: a[k] for k in ['precision','recall']}
    out.update(F1=a['f1'], F2=a['f2'], PR_AUC=a['pr_auc'], ROC_AUC=a['roc_auc'])
    for pct in [1,5,10]:
        for label in ['Precision','Recall','Lift']:
            out[f'{label}@{pct}%'] = b[f'{label.lower()}_at_{pct}pct']
    out['top_decile_capture'] = b['top_decile_capture']
    return out


def selection_utility(m):
    return .4*m['Recall@5%'] + .4*m['PR_AUC'] + .2*m['F2']


def stability(a,b):
    correlation = float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else float(np.array_equal(a,b))
    aa,bb = top_fraction_flags(a,.05), top_fraction_flags(b,.05)
    return correlation, float(np.sum(aa & bb)/max(aa.sum(),1))


def ensemble_scores(percentiles, weights, kind):
    names = list(weights)
    matrix = np.column_stack([percentiles[n] for n in names])
    w = np.array([weights[n] for n in names], float)
    if np.any(w<0) or w.sum()<=0:
        raise ValueError('Ensemble weights must be nonnegative and nonzero')
    if kind in ['equal_percentile', 'rank_average']:
        return matrix.mean(axis=1)  # Percentile ranks already share a TRAIN reference.
    if kind == 'consensus':
        return (matrix>=.95).sum(axis=1) + matrix.mean(axis=1)*.001
    if kind == 'maximum':
        return matrix.max(axis=1)
    return matrix @ (w/w.sum())
