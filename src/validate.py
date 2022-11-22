from os.path import join as oj
import sys
import warnings
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import sklearn.metrics
from sklearn.metrics import confusion_matrix

def calc_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return tn, fp, fn, tp, sens, spec

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


def sensitivity_specificity_curve(y_test, preds_proba, plot=False, thresholds=None):
    '''preds_proba should be 1d
    '''
    if thresholds is None:
        thresholds = sorted(np.unique(preds_proba))
    sens = []
    spec = []
    for threshold in tqdm(thresholds):
        preds = preds_proba > threshold
        stats = sklearn.metrics.classification_report(y_test, preds,
                                                      output_dict=True, zero_division=0)
        sens.append(stats['1']['recall'])
        spec.append(stats['0']['recall'])

    if plot:
        plt.plot(sens, spec, '.-')
        plt.xlabel('sensitivity')
        plt.ylabel('specificity')
        plt.grid()
    return sens, spec, thresholds

def all_stats_curve(y_test, preds_proba, plot=False, thresholds=None):
    '''preds_proba should be 1d
    '''
    if thresholds is None:
        thresholds = sorted(np.unique(preds_proba))
    all_stats = {
        s: [] for s in ['sens', 'spec', 'ppv', 'npv', 'lr+', 'lr-', 'f1']
    }
    for threshold in tqdm(thresholds):
        preds = preds_proba > threshold
#         stats = sklearn.metrics.classification_report(y_test, preds,
#                                                       output_dict=True,
#                                                       zero_division=0)
        
#         all_stats['sensitivity'].append(stats['1']['recall'])
#         all_stats['specificity'].append(stats['0']['recall'])
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
            all_stats['f1'].append(tp / (tp + 0.5 * (fp + fn)))
        

    if plot:
        plt.plot(all_stats['sens'], all_stats['spec'], '.-')
        plt.xlabel('sensitivity')
        plt.ylabel('specificity')
        plt.grid()
    return all_stats, thresholds



scorers = {
    'balanced_accuracy': metrics.balanced_accuracy_score,
    'accuracy': metrics.accuracy_score,
    'precision': metrics.precision_score,
    'sensitivity': metrics.recall_score,
    'specificity': specificity_score,
    'f1': metrics.f1_score,
    'roc_auc': metrics.roc_auc_score,
    'precision_recall_curve': metrics.precision_recall_curve,
    'sensitivity_specificity_curve': sensitivity_specificity_curve,
    'roc_curve': metrics.roc_curve,
    'tn': lambda y_true, y_pred: metrics.confusion_matrix(y_true, y_pred).ravel()[0],
    'fp': lambda y_true, y_pred: metrics.confusion_matrix(y_true, y_pred).ravel()[1],
    'fn': lambda y_true, y_pred: metrics.confusion_matrix(y_true, y_pred).ravel()[2],
    'tp': lambda y_true, y_pred: metrics.confusion_matrix(y_true, y_pred).ravel()[3],
}


def append_score_results(scores, pred, y, suffix1='', suffix2='', thresh=None):
    '''Score given one pred, y
    '''

    # find best thresh
    '''
    if thresh is None:
        prec, rec, thresholds = metrics.precision_recall_curve(y, pred)
        sens, spec, thresholds = sensitivity_specificity_curve(y, pred, thresholds=thresholds)
    '''
    thresh = 0.5

    # compute scores
    for k in scorers.keys():
        k_new = k + suffix1 + suffix2
        if not k_new in scores.keys():
            scores[k_new] = []
        if 'roc' in k or 'curve' in k:
            scores[k_new].append(scorers[k](y, pred))
        else:
            pred_thresholded = (np.array(pred) > thresh).astype(int)
            scores[k_new].append(scorers[k](y, pred_thresholded))
    return thresh


def select_best_fold(scores):
    '''Calculate the index of the best fold
    '''
    return np.argmax(scores['roc_auc_cv_folds'])


def get_scores(predictions_list, predictions_test1_list, predictions_test2_list,
               Y_train, Y_test1, Y_test2):
    '''
    Params
    ------
    predictions_list: (num_folds, num_in_fold)
        predictions for different cv folds
    predictions_test1_list: (num_folds, num_test)
        predictions for test trained on different cv folds
    predictions_test2_list: (num_folds, num_test)
        predictions for test trained on different cv folds
    Y_train: (num_folds, num_in_fold)
        different folds have different ys
    Y_test1: (num_test1)
        test ys 
    Y_test2: (num_test2)
        test ys
    '''
    scores = {}

    # calc scores for individual folds
    # print('score cv folds...', len(preds_list))
    for i in range(len(predictions_list)):
        thresh = append_score_results(scores, predictions_list[i],
                                      Y_train[i], suffix1='_cv', suffix2='_folds')

    # select best model
    idx_best = select_best_fold(scores)
    predictions_cv = np.concatenate([p for p in predictions_list]).flatten()
    ys_cv = np.concatenate([y for y in Y_train]).flatten()
    predictions_test1 = predictions_test1_list[idx_best]
    predictions_test2 = predictions_test2_list[idx_best]

    # repeat for each fold
    for preds, ys, suffix1 in zip([predictions_cv, predictions_test1, predictions_test2],
                                  [ys_cv, Y_test1, Y_test2],
                                  ['_cv', '_test1', '_test2']):
        # score total dset
        # print('score total dset....')
        append_score_results(scores, preds, ys, suffix1=suffix1, thresh=thresh)

    # replace all length 1 lists with scalars
    s = {}
    for k in scores.keys():
        if len(scores[k]) == 1:
            s[k] = scores[k][0]
        else:
            s[k] = np.array(scores[k])
    s['idx_best'] = idx_best
    return s
