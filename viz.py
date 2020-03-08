from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import confusion_matrix
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
import torch
from copy import deepcopy
from sklearn import metrics
plt.style.use('dark_background')
import mat4py
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from colorama import Fore
import pickle as pkl
from style import *

def visualize_individual_results(results, X_test, Y_test, print_results=True):
    '''Print and visualize results from a single train.
    '''
    scores_cv = results['cv']
    scores_test = results['test']
    imps = results['imps']
    m = imps['model'][0]
    
    if print_results:
        print(Fore.CYAN + f'{"metric":<25}\tvalidation') #\ttest')
        for s in results['metrics']:
            if not 'curve' in s:
                print(Fore.WHITE + f'{s:<25}\t{np.mean(scores_cv[s]):.3f} ~ {np.std(scores_cv[s]):.3f}')
        #         print(Fore.WHITE + f'{s:<25}\t{np.mean(scores_cv[s]):.3f} ~ {np.std(scores_cv[s]):.3f}\t{np.mean(scores_test[s]):.3f} ~ {np.std(scores_test[s]):.3f}')

        print(Fore.CYAN + '\nfeature importances')
        imp_mat = np.array(imps['imps'])
        imp_mu = imp_mat.mean(axis=0)
        imp_sd = imp_mat.std(axis=0)
        for i, feat_name in enumerate(results['feat_names']):
            print(Fore.WHITE + f'{feat_name:<25}\t{imp_mu[i]:.3f} ~ {imp_sd[i]:.3f}')

    # print(m.coef_)
    plt.figure(figsize=(10, 3), dpi=200)
    R, C = 1, 3
    plt.subplot(R, C, 1)
    # print(X_test.shape, results['feat_names'])
    preds = m.predict(X_test[results['feat_names']])
    preds_proba = m.predict_proba(X_test[results['feat_names']])[:, 1]    
    plot_confusion_matrix(Y_test, preds, classes=np.array(['Failure', 'Success']))

    plt.subplot(R, C, 2)
    prec, rec, thresh = scores_test['precision_recall_curve'][0]
    plt.plot(rec, prec)
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    
    
    plt.subplot(R, C, 3)
    plt.hist(preds_proba[Y_test==0], alpha=0.5, label='Failure')
    plt.hist(preds_proba[Y_test==1], alpha=0.5, label='Success')
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return preds, preds_proba

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     fig, ax = plt.subplots()
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    ax = plt.gca()
#     ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
#            title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax

def highlight_max(data, color='#0e5c99'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
    
    
# visualize biggest errs
def viz_biggest_errs(X_traces_test, Y_test, preds, preds_proba):
#     print(preds_proba.shape, X_traces_test.shape)
    residuals = np.abs(Y_test - preds_proba)
    
    R, C = 4, 4
    args = np.argsort(residuals)[::-1][:R * C]
#     print(Y_test[args])
#     print(preds[args])
#     print(residuals[args][:10])
    plt.figure(figsize=(C * 3, R * 2.5), dpi=200)
    
    i = 0
    for r in range(R):
        for c in range(C):
            plt.subplot(R, C, i + 1)
            plt.plot(X_traces_test.iloc[args[i]], color=cr)
            i += 1
    
    plt.tight_layout()
    
# visualize biggest errs
def viz_errs_spatially(df, idxs_test, preds, Y_test):
    x_pos = df['x_pos'][idxs_test]
    y_pos = df['y_pos'][idxs_test]
    
    plt.figure(dpi=200)

    plt.plot(x_pos[(preds==Y_test) & (preds==1)], y_pos[(preds==Y_test) & (preds==1)], 'o',
             color=cb, alpha=0.5, label='true pos')
    plt.plot(x_pos[(preds==Y_test) & (preds==0)], y_pos[(preds==Y_test) & (preds==0)], 'x',
             color=cb, alpha=0.5, label='true neg')
    plt.plot(x_pos[preds > Y_test], y_pos[preds > Y_test], 'o', color=cr, alpha=0.5, label='false pos')    
    plt.plot(x_pos[preds < Y_test], y_pos[preds < Y_test], 'x', color=cr, alpha=0.5, label='false neg')    
    plt.legend()
#     plt.scatter(x_pos, y_pos, c=preds==Y_test, alpha=0.5)
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.tight_layout()
    
    
def viz_errs_lifetime(X_test, preds, preds_proba, Y_test, norms):
    plt.figure(dpi=200)
    correct_idxs = preds == Y_test
    lifetime = X_test['lifetime'] * norms['lifetime']['std'] + norms['lifetime']['mu']

    plt.plot(lifetime[(preds==Y_test) & (preds==1)], preds_proba[(preds==Y_test) & (preds==1)], 'o',
             color=cb, alpha=0.5, label='true pos')
    plt.plot(lifetime[(preds==Y_test) & (preds==0)], preds_proba[(preds==Y_test) & (preds==0)], 'x',
             color=cb, alpha=0.5, label='true neg')
    plt.plot(lifetime[preds > Y_test], preds_proba[preds > Y_test], 'o', color=cr, alpha=0.5, label='false pos')    
    plt.plot(lifetime[preds < Y_test], preds_proba[preds < Y_test], 'x', color=cr, alpha=0.5, label='false neg')    
    plt.xlabel('lifetime')
    plt.ylabel('predicted probability')
    plt.legend()
    plt.show()