########################################################################################################################
# Apache License 2.0
########################################################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Nina de Lacy
########################################################################################################################

########################################################################################################################
# Overview: Provide a script to return a list of performance metrics for binary classification.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, accuracy_score, auc, roc_curve,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, precision_recall_curve, log_loss, mean_squared_error)
from statsmodels.stats.proportion import proportion_confint
from typing import Optional

########################################################################################################################
# Define a function to compute 95% confidence intervals for certain discriminatory metrics computed subsequently.
# Remark: Applicable only for metrics that can be represented by binomial proportions (i.e., precision, recall,
# specificity, NPV, accuracy).
########################################################################################################################


def wilson_ci(x_: int,   # numerator of the metric formula
              n_: int):  # denominator of the metric formula
    return proportion_confint(x_, n_, alpha=0.05, method='wilson') if n_ > 0 else (np.nan, np.nan)

########################################################################################################################
# Define a custom function to compute expected cost
########################################################################################################################


def ec_fast(p: np.ndarray,
            y: np.ndarray,
            cost_ratio: float):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)

    order = np.argsort(p, kind='mergesort')     # sort by ascending risk (stable to preserve order within ties)
    p_sorted, y_sorted = p[order], y[order]

    # Positive / negative indicators
    pos = (y_sorted == 1).astype(np.int64)
    neg = 1 - pos
    pos_n = int(pos.sum())
    neg_n = int(neg.sum())
    N = pos_n + neg_n
    if N == 0 or pos_n == 0 or neg_n == 0:
        return np.nan, np.nan  # degenerate edge case
    prev = pos_n / N

    # Cumulative counts up to each index i (inclusive)
    cum_pos = np.cumsum(pos)
    cum_neg = np.cumsum(neg)

    # Evaluate thresholds only at the *starts* of runs of unique scores (at positions where a new score value begins)
    is_new = np.empty_like(p_sorted, dtype=bool)
    is_new[0] = True
    is_new[1:] = p_sorted[1:] != p_sorted[:-1]
    idx = np.flatnonzero(is_new)

    # For threshold t = p_sorted[idx[j]], classify >= t as positive.
    # Positives below threshold = FN count; negatives >= threshold = FP count
    # Using counts before idx (hence idx-1) for "below threshold"
    i_prev = np.clip(idx - 1, 0, p_sorted.size - 1)
    fn = np.where(idx == 0, 0, cum_pos[i_prev])  # positives below threshold
    fp = np.where(idx == 0, neg_n, neg_n - cum_neg[i_prev])  # negatives at/above threshold

    fnr = fn / pos_n
    fpr = fp / neg_n

    # Cost weights
    C_FN, C_FP = cost_ratio, 1.0
    ec_vec = fnr * prev * C_FN + fpr * (1.0 - prev) * C_FP
    j = int(np.argmin(ec_vec))
    ec_min = float(ec_vec[j])
    t_star = float(p_sorted[idx[j]])
    return ec_min, t_star

########################################################################################################################
# Define a function to identify the flagged samples based on a precision percentile (e.g., 1%, 2%, 5%)
########################################################################################################################


def flagged_at_top_k_ppv(y_prob: np.ndarray,
                         k: int) -> np.ndarray:
    order: np.ndarray = np.argsort(-y_prob, kind='mergesort')
    m: int = max(1, int(np.round((k/100) * len(y_prob))))
    flagged: np.ndarray = np.zeros(len(y_prob), dtype=bool)
    flagged[order[:m]] = True
    return flagged

########################################################################################################################
# Define a function to specify the weight for false positives in the calculation of (standardized net benefit)
########################################################################################################################


def nb_weight_from_pt(pt: int) -> float:
    assert 0 < pt < 1
    return pt / (1 - pt)

########################################################################################################################
# Define a function to identify the decision threshold based on a specificity percentile (e.g., 99th)
########################################################################################################################


def threshold_at_specificity_k(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               k: int = 99) -> float:
    S0: np.ndarray = y_prob[y_true == 0]
    tau: float = float(np.quantile(S0, k/100, method='linear'))
    # return tau
    return float(np.nextafter(tau, np.inf))     # nudge up to the next smaller float to break ties

########################################################################################################################
# Define a generic function to return a list of performance statistics for binary classification
########################################################################################################################


def binary_metrics(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   y_pred_override: Optional[np.ndarray] = None,
                   threshold: float = 0.5,
                   recall_limit: float = 0.85,
                   cost_ratio: float = 1,
                   nb_weight: float = 1,
                   n_params: int = 1,
                   decimals: int = 5,
                   verbose: bool = False,
                   prefix: Optional[str] = None):
    """
    :param y_true: np.ndarray
           1-dimensional binary ground-truth labels. Must contain both 0 and 1.
    :param y_prob: np.ndarray
           1-dimensional predicted probabilities for the positive class.
    :param y_pred_override: np.ndarray
           Overwrite the prediction by adopting flagged predicted labels (e.g., when using flagged_at_top_k_ppv)
           Default setting: y_pred_override=None
    :param threshold: float
           Probability cutoff for dichotomizing predictions.
           Default setting: threshold=0.5
    :param recall_limit: float
           Minimum recall/sensitivity used to compute the partial AUROC.
           Default setting: recall_limit=0.85
    :param cost_ratio: float
           Relative cost of false negatives versus false positives in expected cost calculations. Values >1 penalize
           false negatives more; values <1 penalize false positives more.
           Default setting: cost_ratio=1
    :param nb_weight: float
           Weight of false positives in the calculation of (standardized) net benefit.
           Default setting: nb_weight=1
    :param n_params: int
           Number of parameters of the predictive model, used as the model complexity component in calculating Akaike
           and Bayesian Information Criteria (AIC and BIC).
           Default setting: n_params=1
    :param decimals: int
           Number of decimal places to round all metrics.
           Default setting: decimals=4
    :param prefix: str or None
           Optional prefix for metric names in the returned dictionary.
           Default setting: prefix=None
    :return:
    A dictionary reporting the following 52 performance metrics in order.
    (^) The metric value is also computed with its 95% confidence intervals using the Wilson statistical method.
    1. #FN: Number of False Negatives
    2. #FP: Number of False Positives
    3. #TN: Number of True Negatives
    4. #TP: Number of True Positives
    5. AIC: Akaike Information Criterion
    6. AUPRC: Area Under the Precision-Recall Curve
    7. AUROC: Area Under the Receiver Operating Characteristic curve
    8-10. Accuracy: Accuracy (^)
    11. AveragePrecision: Average precision
    12. BIC: Bayesian Information Criterion
    13. BalancedAccuracy: Balanced accuracy
    14. BrierScore: Brier score
    15. BrierSkillScore: Brier skill score
    16. CalibrationIntercept: Calibration intercept (ideally 0)
    17. CalibrationSlope: Calibration slope (ideally 1)
    18. CohenKappa: Cohen's kappa used to measure inter-rater agreement
    19. CoxSnellPseudoR2: Cox–Snell pseudo R-squared
    20. DOR: Diagnostic Odds Ratio
    21. DiscriminationSlope: Difference in average predicted probabilities between positive and negative cases
    22. ECE: Expected calibration error
    # ECI: Estimated calibration index
    23. ExpectedCost: Expected misclassification cost under the specified cost_ratio
    24. ExpectedCostThreshold: Probability threshold that minimizes the expected cost
    25. F1: F1 score
    # ICI: Integrated calibration index
    26. LogLikelihood: Log-likelihood
    27. LogLoss: Logarithmic loss (also known as binary cross-entropy loss)
    28. MAPE: Mean Absolute Prediction Error
    29. MCC: Matthews Correlation Coefficient
    30. McFaddenPseudoR2: McFadden pseudo R-squared
    31-33. NPV: Negative Predictive Value (^)
    34. NagelkerkePseudoR2: Nagelkerke pseudo R-squared
    35. NetBenefit: Net benefit
    36. NNB: Number needed to evaluate = 1 / Precision
    37. O:ERatio: Observed-to-Expected ratio
    38. PartialAUROC: Partial AUROC, restricted to the region above the prespecified recall_limit
    39-41. Precision: Precision (also known as PPV) (^)
    42-44. Recall: Recall (also known as sensitivity) (^)
    45-47. Specificity: Specificity (^)
    48. StandardizedNetBenefit: Standardized net benefit adjusted for prevalence
    49. YoudenIndex: Youden's index
    50. FPR_LIST: False positive rate values across thresholds from the ROC curve
    51. TPR_LIST: True positive rate values across thresholds from the ROC curve
    52. ROC_Threshold_LIST: The thresholds considered in the ROC curve
    53. Precision_LIST: Precision values across thresholds from the PR curve
    54. Recall_LIST: Recall values across thresholds from the PR curve
    55. PR_Threshold_LIST: The thresholds considered in the PR curve
    """
    ####################################################################################################################
    # Type and value check
    ####################################################################################################################
    try:
        y_true = np.asarray(y_true)
    except TypeError:
        raise TypeError('y_true must be (convertible to) a numpy array.')
    assert set(np.unique(y_true)) == {0, 1}, 'y_true must contain both 0 and 1.'
    try:
        y_prob = np.asarray(y_prob)
    except TypeError:
        raise TypeError('y_prob must be (convertible to) a numpy array.')
    assert not (y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape[0] != y_prob.shape[0]), \
        'y_true and y_prob must be 1-D arrays of the same length.'
    assert not (np.any((y_prob < 0) | (y_prob > 1) | ~np.isfinite(y_prob))), \
        'y_prob must be finite probabilities in [0, 1]'
    assert isinstance(threshold, float) and 0 < threshold < 1, \
        'threshold must be a float in the open interval (0, 1).'
    assert isinstance(recall_limit, float) and 0 < recall_limit < 1, \
        'recall_limit must be a float in the open interval (0, 1).'
    assert isinstance(decimals, int) and decimals >= 0, 'decimals must be a non-negative integer.'
    if prefix is not None:
        assert isinstance(prefix, str), 'prefix, if not None, must be a string.'
    else:
        prefix = ''

    ####################################################################################################################
    # Create 2 dictionaries to store the performance statistics and other auxiliary statistics respectively
    ####################################################################################################################
    stat_dict: dict[str, float] = dict()
    aux_dict: dict[str, list[float]] = dict()

    ####################################################################################################################
    # Define the predicted label
    ####################################################################################################################
    if y_pred_override is None:
        y_pred: np.ndarray = (y_prob >= threshold).astype(int)
    else:
        y_pred_override = np.asarray(y_pred_override)
        assert y_pred_override.ndim == 1 and y_pred_override.shape[0] == y_true.shape[0], \
            'y_pred_override must be a 1-D array with the same length as y_true.'
        y_pred: np.ndarray = y_pred_override.astype(int)

    ####################################################################################################################
    # 1. AUROC and partial AUROC
    # Instead of using the implementation from Calster et al., we adopt the standard sklearn implementation.
    ####################################################################################################################
    stat_dict[f'{prefix}AUROC'] = np.round(roc_auc_score(y_true, y_prob), decimals)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    aux_dict[f'{prefix}FPR_LIST'] = list(np.round(fpr, decimals))
    aux_dict[f'{prefix}TPR_LIST'] = list(np.round(tpr, decimals))
    aux_dict[f'{prefix}ROC_Threshold_LIST'] = list(np.round(roc_thresholds, decimals))
    if tpr.max() < recall_limit:
        partial_auroc: float = 0
    else:
        idx = np.argmax(tpr >= recall_limit)
        if idx == 0:
            fpr_star = fpr[0]
        else:
            tpr0, tpr1 = tpr[idx - 1], tpr[idx]
            fpr0, fpr1 = fpr[idx - 1], fpr[idx]
            if tpr0 == tpr1:
                fpr_star = fpr1
            else:
                w = (recall_limit - tpr0) / (tpr1 - tpr0)
                fpr_star = fpr0 + w * (fpr1 - fpr0)
        partial_fpr: np.ndarray = np.concatenate(([fpr_star], fpr[idx:]))
        partial_tpr: np.ndarray = np.concatenate(([recall_limit], tpr[idx:]))
        if partial_fpr[-1] < 1.0:
            partial_fpr = np.append(partial_fpr, 1.0)
            partial_tpr = np.append(partial_tpr, 1.0)
        adjusted_tpr: np.ndarray = partial_tpr - recall_limit
        partial_auroc: float = np.trapz(adjusted_tpr, partial_fpr)
    stat_dict[f'{prefix}PartialAUROC'] = np.round(partial_auroc, decimals)
    if verbose:
        print('(Partial) AUROC computed.')

    ####################################################################################################################
    # 2. AUPRC and average precision (which approximates AUPRC)
    ####################################################################################################################
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    auprc = auc(recalls[::-1], precisions[::-1])
    stat_dict[f'{prefix}AUPRC'] = np.round(auprc, decimals)
    stat_dict[f'{prefix}AveragePrecision'] = np.round(average_precision_score(y_true, y_prob), decimals)
    aux_dict[f'{prefix}Precision_LIST'] = list(np.round(precisions, decimals))
    aux_dict[f'{prefix}Recall_LIST'] = list(np.round(recalls, decimals))
    aux_dict[f'{prefix}PR_Threshold_LIST'] = list(np.round(pr_thresholds, decimals))
    if verbose:
        print('AUPRC and average precision computed.')

    ####################################################################################################################
    # 3. Standard classification performance metrics
    # Instead of using the implementation from Calster et al., we adopt the standard sklearn implementation.
    ####################################################################################################################
    # Confusion matrix [from sklearn]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Accuracy, precision, recall, and F1 [from sklearn]
    accuracy: float = accuracy_score(y_true, y_pred)
    precision: float = precision_score(y_true, y_pred, zero_division=np.nan)
    recall: float = recall_score(y_true, y_pred, zero_division=np.nan)
    f1: float = f1_score(y_true, y_pred, zero_division=np.nan)
    nne: float = 1 / precision if precision != 0 else np.nan

    # Specificity, NPV, and Diagnostic Odds Ratio (DOR) [derived from confusion matrix]
    specificity: float = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv: float = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    dor_denominator: float = (1 - specificity) * (1 - recall) if not (np.isnan(specificity) or np.isnan(recall)) \
        else np.nan
    dor: float = (specificity * recall) / dor_denominator if dor_denominator > 0 else np.nan

    # Cohen's Kappa [derived from confusion matrix]
    prevalence: float = (tp + fn) / (tp + tn + fp + fn)
    predicted_positive_rate: float = (tp + fp) / (tp + tn + fp + fn)
    expected_accuracy: float = prevalence * predicted_positive_rate + (1 - prevalence) * (1 - predicted_positive_rate)
    kappa: float = (accuracy - expected_accuracy) / (1 - expected_accuracy) if expected_accuracy != 1 else np.nan

    # Matthews Correlation Coefficient (MCC) [derived from confusion matrix]
    a = tp + fp
    b = tp + fn
    c = tn + fp
    d = tn + fn
    if a * b * c * d == 0:
        mcc: float = np.nan
    else:
        mcc_denominator: float = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc: float = ((tp * tn - fp * fn) / mcc_denominator) if (np.isfinite(mcc_denominator) and mcc_denominator > 0) \
            else np.nan

    # Compute the 95% confidence intervals using the Wilson method
    _acc_lo, _acc_hi = wilson_ci(tp+tn, tp+tn+fp+fn)
    _pre_lo, _pre_hi = wilson_ci(tp, tp+fp)
    _rec_lo, _rec_hi = wilson_ci(tp, tp+fn)
    _spec_lo, _spec_hi = wilson_ci(tn, tn+fp)
    _npv_lo, _npv_hi = wilson_ci(tn, tn+fn)

    # Update the dictionary
    stat_dict[f'{prefix}Accuracy'] = np.round(accuracy, decimals)
    stat_dict[f'{prefix}AccuracyLoCI'] = np.round(_acc_lo, decimals)
    stat_dict[f'{prefix}AccuracyHiCI'] = np.round(_acc_hi, decimals)
    stat_dict[f'{prefix}BalancedAccuracy'] = np.round(0.5 * (specificity + recall), decimals)
    stat_dict[f'{prefix}Precision'] = np.round(precision, decimals)
    stat_dict[f'{prefix}PrecisionLoCI'] = np.round(_pre_lo, decimals)
    stat_dict[f'{prefix}PrecisionHiCI'] = np.round(_pre_hi, decimals)
    stat_dict[f'{prefix}Recall'] = np.round(recall, decimals)
    stat_dict[f'{prefix}RecallLoCI'] = np.round(_rec_lo, decimals)
    stat_dict[f'{prefix}RecallHiCI'] = np.round(_rec_hi, decimals)
    stat_dict[f'{prefix}F1'] = np.round(f1, decimals)
    stat_dict[f'{prefix}Specificity'] = np.round(specificity, decimals)
    stat_dict[f'{prefix}SpecificityLoCI'] = np.round(_spec_lo, decimals)
    stat_dict[f'{prefix}SpecificityHiCI'] = np.round(_spec_hi, decimals)
    stat_dict[f'{prefix}NNE'] = np.round(nne, decimals)
    stat_dict[f'{prefix}NPV'] = np.round(npv, decimals)
    stat_dict[f'{prefix}NPVLoCI'] = np.round(_npv_lo, decimals)
    stat_dict[f'{prefix}NPVHiCI'] = np.round(_npv_hi, decimals)
    stat_dict[f'{prefix}DOR'] = np.round(dor, decimals)
    stat_dict[f'{prefix}YoudenIndex'] = np.round(specificity + recall - 1, decimals)
    stat_dict[f'{prefix}CohenKappa'] = np.round(kappa, decimals)
    stat_dict[f'{prefix}MCC'] = np.round(mcc, decimals)
    stat_dict[f'{prefix}#TP'] = int(tp)
    stat_dict[f'{prefix}#FP'] = int(fp)
    stat_dict[f'{prefix}#TN'] = int(tn)
    stat_dict[f'{prefix}#FN'] = int(fn)
    if verbose:
        print('Standard classification metrics computed.')

    ####################################################################################################################
    # 4. Calibration metrics
    # Metrics marked with *** are commented out as they (a) require specify packages to run (unavailable in Cosmos), or
    # (b) require a long runtime in the Cosmos environment.
    ####################################################################################################################
    # Adjusted probabilities
    p: np.ndarray = np.clip(y_prob.astype(float), 1e-15, 1 - 1e-15)

    # Observed:Expected ratio
    oe_ratio: float = np.sum(y_true) / np.sum(p) if np.sum(p) > 0 else np.nan

    # *** Calibration intercept ***
    logit: np.ndarray = np.log(p / (1 - p))
    intercept: float = sm.GLM(y_true, np.ones((len(y_true), 1)),
                              family=sm.families.Binomial(), offset=logit).fit().params[0]
    stat_dict[f'{prefix}CalibrationIntercept'] = np.round(intercept, decimals)

    # *** Calibration slope ***
    sl_model: LogisticRegression = (LogisticRegression(penalty=None, solver='lbfgs').
                                    fit(X=np.array(logit).reshape(-1, 1), y=y_true))
    slope: float = sl_model.coef_[0][0]
    stat_dict[f'{prefix}CalibrationSlope'] = np.round(slope, decimals)

    # *** Estimated Calibration Index (ECI) ***
    # flc = localreg(y=np.array(y_true), x=np.array(p), frac=loess_frac, degree=2)      # flexible calibration curve
    # flc = lowess(endog=y_true, exog=p, frac=loess_frac, it=0, return_sorted=False)
    # eci_denominator: float = np.mean((np.full(y_true.shape, np.mean(y_true), dtype=float) - p) ** 2)
    # eci: float = np.mean((flc - p) ** 2) / eci_denominator if eci_denominator > 0 else np.nan
    # stat_dict[f'{prefix}ECI'] = np.round(eci, decimals)

    # *** Integrated Calibration Index (ICI) ***
    # ici: float = np.mean(np.abs(flc - p))
    # stat_dict[f'{prefix}ICI'] = np.round(ici, decimals)

    # Expected Calibration Error (ECE) - weighted micro ECE
    hsl: pd.DataFrame = pd.DataFrame({'x': p, 'y': y_true})
    hsl['x_quantile'] = pd.qcut(hsl['x'], q=10, duplicates='drop')
    tmp = (hsl.groupby('x_quantile', observed=True)
           .agg(mean_x=('x', 'mean'),
                mean_y=('y', 'mean'),
                n=('y', 'size')).reset_index())
    ece: float = (tmp['n'] / len(hsl) * (tmp['mean_x'] - tmp['mean_y']).abs()).sum()
    # mean_hsl: pd.DataFrame = (hsl.groupby('x_quantile', observed=True)
    #                           .agg(mean_x=('x', 'mean'), mean_y=('y', 'mean')).reset_index())
    # ece: float = (abs(mean_hsl['mean_x'] - mean_hsl['mean_y'])).mean()

    # Update the dictionary
    stat_dict[f'{prefix}O:ERatio'] = np.round(oe_ratio, decimals)
    stat_dict[f'{prefix}ECE'] = np.round(ece, decimals)
    if verbose:
        print('Calibration metrics computed.')

    ####################################################################################################################
    # 5. Overall performance metrics
    ####################################################################################################################
    lli: float = np.sum(binom.logpmf(y_true, n=1, p=p))                                         # Log-likelihood (model)
    ll0: float = np.sum(binom.logpmf(y_true, n=1, p=np.full(len(y_true), np.mean(y_true))))     # Log-likelihood (null)
    l_loss: float = log_loss(y_true, p, normalize=False)                                        # Log-loss
    brier: float = mean_squared_error(y_true, p)                                                 # Brier score
    bss: float = 1 - (brier / mean_squared_error(y_true, np.full(y_true.shape, np.mean(y_true),  # Brier skill score
                                                                 dtype=float)))
    mfr2: float = 1 - (lli / ll0)                                                    # McFadden pseudo R-squared
    csr2: float = 1 - np.exp(2 * (ll0 - lli) / len(y_true))                          # Cox-Snell pseudo R-squared
    nr2: float = csr2 / (1 - np.exp(2 * ll0 / len(y_true)))                          # Nagelkerke pseudo R-squared
    ds: float = np.mean(p[y_true == 1]) - np.mean(p[y_true == 0])                    # Discrimination slope
    mape: float = np.mean(np.abs(y_true - p))                                        # Mean absolute prediction error

    n: int = len(y_true)
    aic: float = 2 * n_params + 2 * lli
    bic: float = np.log(n) * n_params + 2 * lli

    # Update the dictionary
    stat_dict[f'{prefix}LogLikelihood'] = np.round(lli, decimals)
    stat_dict[f'{prefix}LogLoss'] = np.round(l_loss, decimals)
    stat_dict[f'{prefix}BrierScore'] = np.round(brier, decimals)
    stat_dict[f'{prefix}BrierSkillScore'] = np.round(bss, decimals)
    stat_dict[f'{prefix}McFaddenPseudoR2'] = np.round(mfr2, decimals)
    stat_dict[f'{prefix}CoxSnellPseudoR2'] = np.round(csr2, decimals)
    stat_dict[f'{prefix}NagelkerkePseudoR2'] = np.round(nr2, decimals)
    stat_dict[f'{prefix}DiscriminationSlope'] = np.round(ds, decimals)
    stat_dict[f'{prefix}MAPE'] = np.round(mape, decimals)
    stat_dict[f'{prefix}AIC'] = np.round(aic, decimals)
    stat_dict[f'{prefix}BIC'] = np.round(bic, decimals)
    if verbose:
        print('Overall metrics computed.')

    ####################################################################################################################
    # 6. Utility performance metrics
    # Metrics marked with *** are commented out as they (a) require specify packages to run (unavailable in Cosmos), or
    # (b) require a long runtime in the Cosmos environment.
    ####################################################################################################################
    # Net benefit
    flagged = (y_pred == 1)
    nb = np.mean(flagged & (y_true == 1)) - nb_weight * np.mean(flagged & (y_true == 0))

    # Standardized net benefit
    snb: float = nb / np.mean(y_true) if np.mean(y_true) > 0 else np.nan

    # Expected cost and its threshold
    ec, ecThreshold = ec_fast(p, y_true, cost_ratio)

    # Update the dictionary
    stat_dict[f'{prefix}NetBenefit'] = np.round(nb, decimals)
    stat_dict[f'{prefix}StandardizedNetBenefit'] = np.round(snb, decimals)
    stat_dict[f'{prefix}ExpectedCost'] = np.round(ec, decimals)
    stat_dict[f'{prefix}ExpectedCostThreshold'] = np.round(ecThreshold, decimals)
    if verbose:
        print('Utility performance metrics computed.')

    ####################################################################################################################
    # Return the results
    ####################################################################################################################
    stat_dict: dict = {k: stat_dict[k] for k in sorted(stat_dict.keys())} | aux_dict
    return stat_dict
