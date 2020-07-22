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

def get_model(model_type):
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
    return m

    
def select_features(feature_selection, feature_selection_num, X_train, X_test1, X_test2, y_train):
    '''Select features and return the selected ones
    '''
    # don't perform any features selection
    if feature_selection is None:
        return X_train, X_test1, X_test2, np.ones(len(feat_names)).astype(np.bool)


    # perform some feature selection
    if 'stab' in feature_selection:
        if feature_selection == 'select_stab_lasso':
            feature_selector_model = LogisticRegression(penalty='l1', solver='liblinear')
        feature_selector = StabilitySelection(base_estimator=feature_selector_model,
                                              lambda_name='C',
                                              lambda_grid=np.logspace(-5, -1, 20),
                                              max_features=feature_selection_num)
    else:
        if feature_selection == 'select_lasso':
            feature_selector_model = Lasso()
        elif feature_selection == 'select_rf':
            feature_selector_model = RandomForestClassifier()
        feature_selector = SelectFromModel(feature_selector_model, threshold=-np.inf,
                                           max_features=feature_selection_num)
    feature_selector.fit(X_train, y_train)
    X_train = feature_selector.transform(X_train)
    X_test1 = feature_selector.transform(X_test1)
    X_test2 = feature_selector.transform(X_test2)
    support = np.array(feature_selector.get_support())

    return X_train, X_test1, X_test2, support

def predict_over_folds(cv_folds, X, y, X_test1, X_test2,
                       m, sample_weights, balancing, train_idxs, model_type):
    '''loop over folds
    Returns
    -------
    predictions
        predictions on test should be based on the model which has best performance on individual fold
    fitted models
    importances
    '''

    models = []
    imps = []
    preds_list = []
    preds_test1_list = []    
    preds_test2_list = []    
    ys = []
    splits = KFold(n_splits=len(train_idxs)).split(train_idxs)
    for cv_idx, cv_val_idx in splits:

        # get sample indices
        idxs_cv = cv_folds.isin(cv_idx + 1)
        idxs_val_cv = cv_folds.isin(cv_val_idx + 1)
        X_train_cv, Y_train_cv = X[idxs_cv], y[idxs_cv]
        X_val_cv, Y_val_cv = X[idxs_val_cv], y[idxs_val_cv]

        # fit with appropriate weighting
        if balancing == 'sample_weights':
            m.fit(X_train_cv, Y_train_cv, sample_weight=sample_weights[idxs_cv])
        else:
            X_train_r_cv, Y_train_r_cv = balance(X_train_cv, Y_train_cv, balancing, balancing_ratio)
            m.fit(X_train_r_cv, Y_train_r_cv)
        
        # append lists
        preds_list.append(m.predict_proba(X_val_cv))
        preds_test1_list.append(m.predict_proba(X_test1))
        preds_test2_list.append(m.predict_proba(X_test2))                    
        models.append(deepcopy(m))
        imps.append(get_feature_importance(m, model_type, X_val_cv, Y_val_cv))
        ys.append(Y_val_cv)
        
    return models, imps, preds_list, preds_test1_list, preds_test2_list, ys

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

def append_score_results(scores, pred, y, prefix=''):
    '''Score given one pred, y
    '''
    
    for k in scorers.keys():
        k_new = prefix + '_' + k
        if not k_new in scores.keys():
            scores[k_new] = []
        
        print(k_new)
        if 'roc' in k or 'curve' in k:
            scores[k_new].append(scorers[k](y, pred))
        else:
            pred_thresh = (pred > 0.5).astype(int)
            print(np.unique(y), np.unique(pred_thresh))
#             scores[k_new].append(scorers[k](y, pred_thresh))
        
        

def get_scores(predictions_list, predictions_test1_list, predictions_test2_list,
               Y_train, Y_test1, Y_test2):
    
    # score cv folds
    scores = {}
    for preds_list, Y_list in zip([predictions_list, predictions_test1_list, predictions_test2_list], 
                                  [Y_train, Y_test1, Y_test2]):
        for i in range(len(preds_list)):
            append_score_results(scores, preds_list[i], Y_list[i], prefix='cv')
        
        append_score_results(scores, np.vstack(preds_list), np.vstack(Y_list))
    
    return scores


def train(df: pd.DataFrame, feat_names: list, model_type='rf', outcome_def='iai_intervention',
          sample_weights=None, balancing='ros', balancing_ratio=1,
          out_name='results/classify/test.pkl', 
          train_idxs=[1, 2, 3, 4, 5], test_idxs1=[6], test_idxs2=[7], feature_selection=None, feature_selection_num=3):
    '''Balance classes in y using strategy specified by balancing
    '''
    np.random.seed(42)
    
    # normalize the data
    X = df[feat_names]
    X = (X - X.mean()) / X.std()
    y = df[outcome_def].values

    # split data based on cv_fold
    idxs_train = df.cv_fold.isin(train_idxs)
    X_train, y_train = X[idxs_train], y[idxs_train]
    idxs_test1 = df.cv_fold.isin(test_idxs1)
    X_test1, Y_test1 = X[idxs_test1], y[idxs_test1]
    idxs_test2 = df.cv_fold.isin(test_idxs2)
    X_test2, Y_test2 = X[idxs_test2], y[idxs_test2]
    
    # get model
    m = get_model(model_type)
    
    # feature selection
    X_train, X_test1, X_test2, support = \
        select_features(feature_selection, feature_selection_num, X_train, X_test1, X_test2, y_train)
    
    
    # prediction
    models, imps, predictions_list, predictions_test1_list, predictions_test2_list, y_train = \
        predict_over_folds(df.cv_fold[idxs_train], X_train, y_train,
                           X_test1, X_test2, m, sample_weights[idxs_train],
                           balancing, train_idxs, model_type)
    
    # scoring
    scores = get_scores(predictions_list, predictions_test1_list, predictions_test2_list, y_train, Y_test1, Y_test2)
    
    
    # save results
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    results = {
        'models': models,
        'imps': imps,        
        **scores,
        
        'metrics': list(scorers.keys()), 
        
        # params
        'model_type': model_type,
        'balancing': balancing,
        'feat_names_selected': np.array(feat_names)[support],               
        'balacing_ratio': balancing_ratio,
    }
    pkl.dump(results, open(out_name, 'wb'))