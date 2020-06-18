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
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import KFold
import pandas as pd
import data 
from collections import Counter
from typing import List

def get_feature_importance(model, model_type, X_val, Y_val):
    '''Get feature importance based on model
    '''
    
    if model_type in ['rf', 'dt']:
        imps = model.feature_importances_
    elif model_type == 'logistic':
        imps = model.coef_
    else:
        perm = eli5.sklearn.permutation_importance.PermutationImportance(model).fit(X_val, Y_val)
        imps = perm.feature_importances_
    return imps.squeeze()

def balance(X, y, balancing='ros', balancing_ratio: float=1):
    '''Balance classes in y using strategy specified by balancing
    
    Params
    ------
    balancing_ratio: float
        num positive / num negative desired, negative class is left the same
    '''
    if balancing == 'none':
        return X, y
    
    class0 = np.sum(y==0)
    class0_new = class0
    class1_new = int(class0 * balancing_ratio)
    desired_ratio = {0: class0_new, 1: class1_new}
    
    if balancing == 'ros':
        sampler = RandomOverSampler(desired_ratio, random_state=42)
    elif balancing == 'smote':
        sampler = SMOTE(desired_ratio, random_state=42)
        
    X_r, Y_r = sampler.fit_resample(X, y)   
    return X_r, Y_r



def train(df: pd.DataFrame, feat_names: list, model_type='rf', outcome_def='iai_intervention',
          balancing='ros', balancing_ratio=1, out_name='results/classify/test.pkl', 
          train_idxs=[1, 2, 3, 4, 5], test_idxs1=[6], test_idxs2=[7], feature_selection=None, feature_selection_num=3):
    '''Balance classes in y using strategy specified by balancing
    '''
    np.random.seed(42)
    # make logistic data
    X = df[feat_names]
    X = (X - X.mean()) / X.std()
    y = df[outcome_def].values

    # split testing data based on cell num
    idxs_test1 = df.cv_fold.isin(test_idxs1)
    X_test1, Y_test1 = X[idxs_test1], y[idxs_test1]
    idxs_test2 = df.cv_fold.isin(test_idxs2)
    X_test2, Y_test2 = X[idxs_test2], y[idxs_test2]
    
    if model_type == 'rf':
        m = RandomForestClassifier(n_estimators=100)
    elif model_type == 'dt':
        m = DecisionTreeClassifier()
    elif model_type == 'logistic':
        m = LogisticRegression(solver='lbfgs')
    elif model_type == 'svm':
        m = SVC(gamma='scale')
    elif model_type == 'mlp2':
        m = MLPClassifier()
    elif model_type == 'gb':
        m = GradientBoostingClassifier()
    
    # feature selection
    feature_selector = None
    if feature_selection is not None:
        if feature_selection == 'select_lasso':
            feature_selector_model = Lasso()
        elif feature_selection == 'select_rf':
            feature_selector_model = RandomForestClassifier()
        feature_selector = SelectFromModel(feature_selector_model, threshold=-np.inf,
                                           max_features=feature_selection_num)
        feature_selector.fit(X, y)
        X = feature_selector.transform(X)
        X_test1 = feature_selector.transform(X_test1)
        X_test2 = feature_selector.transform(X_test2)
        support = np.array(feature_selector.get_support())
    else:
        support = np.ones(len(feat_names)).astype(np.bool)    
    

    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + tp)
    
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
    scores_cv = {s: [] for s in scorers.keys()}
    scores_test1 = {s: [] for s in scorers.keys()}
    scores_test2 = {s: [] for s in scorers.keys()}
    imps = {'model': [], 'imps': []}

    kf = KFold(n_splits=5)
    cv_folds_train = train_idxs
    for cv_idx, cv_val_idx in kf.split(cv_folds_train):
        # get sample indices
        idxs_cv = df.cv_fold.isin(cv_idx + 1)
        idxs_val_cv = df.cv_fold.isin(cv_val_idx + 1)
        X_train_cv, Y_train_cv = X[idxs_cv], y[idxs_cv]
        X_val_cv, Y_val_cv = X[idxs_val_cv], y[idxs_val_cv]

        # resample training data
        X_train_r_cv, Y_train_r_cv = balance(X_train_cv, Y_train_cv, balancing, balancing_ratio)

        # fit
        m.fit(X_train_r_cv, Y_train_r_cv)

        # get preds
        preds = m.predict(X_val_cv)
        preds_test1 = m.predict(X_test1)
        preds_test2 = m.predict(X_test2)
        if model_type == 'svm':
            preds_proba = preds
            preds_test1_proba = preds_test1
            preds_test2_proba = preds_test2
        else:
            preds_proba = m.predict_proba(X_val_cv)[:, 1]
            preds_test1_proba = m.predict_proba(X_test1)[:, 1]
            preds_test2_proba = m.predict_proba(X_test2)[:, 1]


        # add scores
        for s in scorers.keys():
            scorer = scorers[s]
            if 'roc' in s or 'curve' in s:
                # print(np.unique(preds, return_counts=True), np.unique(Y_val_cv, return_counts=True))
                scores_cv[s].append(scorer(Y_val_cv, preds_proba))
                scores_test1[s].append(scorer(Y_test1, preds_test1_proba))
                scores_test2[s].append(scorer(Y_test2, preds_test2_proba))
            else:
                scores_cv[s].append(scorer(Y_val_cv, preds))
                scores_test1[s].append(scorer(Y_test1, preds_test1))
                scores_test2[s].append(scorer(Y_test2, preds_test2))
        imps['model'].append(deepcopy(m))
        imps['imps'].append(get_feature_importance(m, model_type, X_val_cv, Y_val_cv))

    # save results
    # os.makedirs(out_dir, exist_ok=True)
    results = {'metrics': list(scorers.keys()), 
               'cv': scores_cv, 
               'test1': scores_test1,
               'test2': scores_test2,
               'imps': imps,
#                'feat_names': feat_names,
               'model_type': model_type,
               'balancing': balancing,
               'feat_names_selected': np.array(feat_names)[support],               
               'balacing_ratio': balancing_ratio,
              }
    pkl.dump(results, open(out_name, 'wb'))