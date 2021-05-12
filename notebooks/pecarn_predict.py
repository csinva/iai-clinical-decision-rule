import sys
from copy import deepcopy
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
sys.path.append('../src')
import data

def pecarn_rule_predict(df, o=data.outcome_def, print_rule=True):
    '''Predict each subgroup in the original PECARN rule
    '''
    d = deepcopy(df)
    d.index = np.arange(d.shape[0])
    n = d.shape[0]
    npos = d[o].sum()
    if print_rule:
        print(f'{"Initial":<25} {npos} / {n}')
    risks = np.array([np.nan] * d.shape[0])
    rules = [
        ('AbdTrauma_or_SeatBeltSign', ['yes']),
        ('GCSScore', range(14)),
        ('AbdTenderDegree', ['Mild', 'Moderate', 'Severe']),
        ('ThoracicTrauma', ['yes']),        
        ('AbdomenPain', ['yes']),
        ('DecrBreathSound', ['yes']),
        ('VomitWretch', ['yes']),
    ]
    
    preds_proba = np.zeros(df.shape[0])
    for rule in rules:
#         print(list(d.keys()))
        k, vals = rule
        
        # right-hand side
        idxs = d[k].isin(vals)
        do = d[idxs]
        
        num2 = do[o].sum()
        denom2 = do.shape[0]
        prob = num2 / denom2
#         print('prob', num2, denom2, prob, 'end')
#         print(idxs)
        preds_proba[idxs.index] = prob
        
        
        # left-hand side (print both)
        d = d[~idxs]
        if print_rule:
            print(f'{k:<25} {d[o].sum():>3} / {d.shape[0]:>5}\t{num2:>3} / {denom2:>4} ({prob*100:0.1f})')
    
    low_risk_patients = d
    patients_missed = low_risk_patients[low_risk_patients[o] == 1]
    preds_proba[d.index] = d[o].mean()
    
    # calc metrics
    fn = patients_missed.shape[0]
    tp = npos - fn
    tn = low_risk_patients.shape[0] - fn
    fp = (n - low_risk_patients.shape[0]) - tp
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    stats = {
        'fn': fn,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'sensitivity': sens * 100,
        'specificity': spec * 100,
        'ppv': tp / (tp + fp),
        'npv': tn / (tn + fn),
        'lr+': sens / (1 - spec),
        'lr-': (1 - sens) / spec,
        'acc': (tp + tn) / (tp + tn + fp + fn),
        'roc_auc': roc_auc_score(df[o], preds_proba),
        'f1': tp / (tp + 0.5 * (fp + fn)),
        'brier_score': brier_score_loss(df[o], preds_proba),
    }
    return d, patients_missed, stats