import numpy as np
import warnings
from os.path import join as oj
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import sklearn.metrics


def pecarn_rule_predict(d, o='iai_intervention', verbose=False):
    '''Predict each subgroup in the original PECARN rule
    Return the probabilistic predictions of the PECARN rule and the threshold binary predictions
    '''
    n = d.shape[0]
    npos = d[o].sum()
    if verbose:
        print(f'{"Initial":<25} {npos} / {n}')
    rules = [
        ('AbdTrauma_or_SeatBeltSign', ['yes']),
        ('GCSScore', range(14)),
        ('AbdTenderDegree', ['Mild', 'Moderate', 'Severe']),
        ('ThoracicTrauma', ['yes']),
        ('AbdomenPain', ['yes']),
        ('DecrBreathSound', ['yes']),
        ('VomitWretch', ['yes']),
    ]
    preds_high_risk = np.zeros(n).astype(bool)
    preds_probabilistic = np.zeros(n)
    d_small = d
    for rule in rules:
        k, vals = rule
        preds_high_risk_new = d[k].isin(vals)
        # this is the new cohort of patients that is high risk
        idxs_sub_category = preds_high_risk_new & (~preds_high_risk)
        preds_probabilistic[idxs_sub_category] = np.mean(
            d[idxs_sub_category][o])
        preds_high_risk = preds_high_risk | preds_high_risk_new

        if verbose:
            idxs_high_risk = d_small[k].isin(vals)
            do = d_small[idxs_high_risk]
            d_small = d_small[~idxs_high_risk]
            num2 = do[o].sum()
            denom2 = do.shape[0]
            print(
                f'{k:<25} {d_small[o].sum():>3} / {d_small.shape[0]:>5}\t{num2:>3} / {denom2:>4} ({num2/denom2*100:0.1f})')
    preds_probabilistic[~preds_high_risk] = np.mean(d[~preds_high_risk][o])
    return preds_probabilistic, preds_high_risk.astype(int)


def all_stats_curve(y_test, preds_proba, plot=False, thresholds=None):
    '''preds_proba should be 1d
    '''
    if thresholds is None:
        thresholds = sorted(np.unique(preds_proba))
    all_stats = {
        s: [] for s in ['sens', 'spec', 'ppv', 'npv', 'lr+', 'lr-']
    }
    for threshold in thresholds:
        preds = preds_proba > threshold
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            all_stats['sens'].append(sens)
            all_stats['spec'].append(spec)
            all_stats['ppv'].append(tp / (tp + fp))
            all_stats['npv'].append(tn / (tn + fn))
            all_stats['lr+'].append(sens / (1 - spec))
            all_stats['lr-'].append((1 - sens) / spec)

    if plot:
        plt.plot(all_stats['sens'], all_stats['spec'], '.-')
        plt.xlabel('sensitivity')
        plt.ylabel('specificity')
        plt.grid()
    return all_stats, thresholds
