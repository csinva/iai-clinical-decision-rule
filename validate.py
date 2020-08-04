import os
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_validate, ShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
import eli5
import os.path
from imodels import RuleListClassifier, RuleFit, SLIM, GreedyRuleList
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import KFold
import pandas as pd
import data 
from collections import Counter
from typing import List
from stability_selection import StabilitySelection

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

scorers = {
   'balanced_accuracy': metrics.balanced_accuracy_score, 
   'accuracy': metrics.accuracy_score,
   'precision': metrics.precision_score, 
   'sensitivity': metrics.recall_score, 
   'specificity': specificity_score,
   'f1': metrics.f1_score, 
   'roc_auc': metrics.roc_auc_score,
   'precision_recall_curve': metrics.precision_recall_curve, 
   'roc_curve': metrics.roc_curve,
   'tn': lambda y_true, y_pred: metrics.confusion_matrix(y_true, y_pred).ravel()[0],
   'fp': lambda y_true, y_pred: metrics.confusion_matrix(y_true, y_pred).ravel()[1],
   'fn': lambda y_true, y_pred: metrics.confusion_matrix(y_true, y_pred).ravel()[2],
   'tp': lambda y_true, y_pred: metrics.confusion_matrix(y_true, y_pred).ravel()[3],
}
        
def append_score_results(scores, pred, y, suffix1='', suffix2=''):
    '''Score given one pred, y
    '''

    for k in scorers.keys():
        k_new = k + suffix1 + suffix2
        if not k_new in scores.keys():
            scores[k_new] = []
        if 'roc' in k or 'curve' in k:
            scores[k_new].append(scorers[k](y, pred))
        else:
            pred_thresh = (np.array(pred) > 0.5).astype(int)
            scores[k_new].append(scorers[k](y, pred_thresh)) 

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
            
    # repeat for each fold
    Y_test1 = np.tile(np.array(Y_test1).reshape((1, -1)), (len(predictions_list), 1))
    Y_test2 = np.tile(np.array(Y_test2).reshape((1, -1)), (len(predictions_list), 1))
    
    scores = {}
    for preds_list, Y_list, suffix1 in zip([predictions_list, predictions_test1_list, predictions_test2_list], 
                                  [Y_train, Y_test1, Y_test2],
                                  ['_cv', '_test1', '_test2']):
        
        # score cv folds (less important)
        # print('score cv folds...', len(preds_list))
        for i in range(len(preds_list)):
            append_score_results(scores, preds_list[i], Y_list[i], suffix1=suffix1, suffix2='_folds')
        
        # score total dset
        # print('score total dset....')
        all_preds = np.concatenate([p for p in preds_list]).flatten()
        all_ys = np.concatenate([y for y in Y_list]).flatten()
        append_score_results(scores, all_preds, all_ys, suffix1=suffix1)
    
    # replace all length 1 lists with scalars
    s = {}
    for k in scores.keys():
        if len(scores[k]) == 1:
            s[k] = scores[k][0]
        else:
            s[k] = np.array(scores[k])
    return s
