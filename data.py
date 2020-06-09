import os
from os.path import join as oj
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

pecarn_train_idxs = [1, 2, 3, 4, 5]
pecarn_test_idxs = [6]
psrc_train_idxs = [8, 9, 10, 11]
psrc_test_idxs = [12, 13]

def to_dummies(df: pd.DataFrame):
    """Prepare the data for classification
    """

    # convert feats to dummy
    df = pd.get_dummies(df, dummy_na=True)  # treat na as a separate category
    # remove any col that is all 0s
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def derived_feats(df):
    '''Add derived features
    '''
    binary = {
        0: 'no',
        1: 'yes',
        False: 'no',
        True: 'yes',
        'unknown': 'unknown'
    }
    df['AbdTrauma_or_SeatBeltSign'] = ((df.AbdTrauma == 'yes') | (df.SeatBeltSign == 'yes')).map(binary)
    df['Hypotension'] = (df['Age'] < 1/12) & (df['InitSysBPRange'] < 70) | \
                    (df['Age'] >= 1/12) & (df['Age'] < 5) & (df['InitSysBPRange'] < 80) | \
                    (df['Age'] >= 5) & (df['InitSysBPRange'] < 90).map(binary)
    return df

def add_cv_split(df: pd.DataFrame, dset='pecarn'):
    # set up train / test
    np.random.seed(42)
    if dset == 'pecarn':
        offset = 0
    elif dset == 'psrc':
        offset = 7
    df['cv_fold'] = np.random.randint(1, 7, size=df.shape[0]) + offset  # 6 is the test set
    return df

def get_feat_names(df):
    '''Get feature names for pecarn
    
    Original PECARN feats
    ---------------------
    Originally used features: age < 2, severe mechanism of injury (includes many things),
    vomiting, hypotension, GCS
    thoracic tenderness, evidence of thoracic wall trauma
    costal marign tenderness, decreased breath sounds, abdominal distention
    complaints of abdominal pain, abdominal tenderness (3 levels)
    evidence of abdominal wall trauma or seat belt sign
    distracting patinful injury
    femur fracture
    
    Returns
    -------
    feat_names: List[Str]
        All valid feature names
    pecarn_feats: List[Str]
        All valid feature names corresponding to original pecarn iai study
    '''
    feat_names = [k for k in df.keys()  # features to use
                  if not k in ['id', 'cv_fold']
                  and not 'iai' in k.lower()]

    PECARN_FEAT_NAMES = ['AbdDistention',
                         'AbdTenderDegree',
                         'AbdTrauma',
                         'AbdTrauma_or_SeatBeltSign',
                         'AbdomenPain',
                         'Costal',
                         'DecrBreathSound',
                         'DistractingPain',
                         'FemurFracture',
                         'GCSScore',
                         'Hypotension',
                         'LtCostalTender',
                         'MOI',
                         'RtCostalTender',
                         'SeatBeltSign',
                         'ThoracicTender',
                         'ThoracicTrauma',
                         'VomitWretch',
                         'Age'] + \
    ['Race', 'InitHeartRate', 'InitSysBPRange']
    pecarn_feats = set()
    for pecarn_feat in PECARN_FEAT_NAMES:
        for feat_name in feat_names:
            if pecarn_feat in feat_name:
                pecarn_feats.add(feat_name)
    pecarn_feats = sorted(list(pecarn_feats))
    return feat_names, set(pecarn_feats)