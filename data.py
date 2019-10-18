import os
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
import pandas as pd
import data


def get_features(ddir = 'iaip_data/Datasets', rdir = 'results', pdir = 'processed'):
    '''
    Returns
    -------
    features: pd.DataFrame
    '''
    fnames = sorted([fname for fname in os.listdir(ddir) 
                     if 'csv' in fname
                     and not 'formats' in fname
                     and not 'form6' in fname]) # remove outcome
    feature_names = [fname[:-4].replace('form', '').replace('-', '_') for fname in fnames]
    # demographics = pd.read_csv('iaip_data/Datasets/demographics.csv')
    # print(fnames)

    r = {}
    for fname in tqdm(fnames):
        df = pd.read_csv(oj(ddir, fname), engine='python')
        df.rename(columns={'SubjectID': 'id'}, inplace=True)
        df.rename(columns={'subjectid': 'id'}, inplace=True)
        assert('id' in df.keys())
        r[fname] = df

    df = r[fnames[0]]
    how = 'left'

    fnames_small = [fname for fname in fnames
                    if 'form1' in fname
                    or 'form5' in fname
                    or 'form4' in fname]
    for i, fname in tqdm(enumerate(fnames_small)): # this should include more than 3!
        df2 = r[fname].copy()
        df2 = df2.drop_duplicates(subset=['id'], keep='last') # if subj has multiple entries, only keep first
        rename_dict = {
            key: key + '_' + fname[:-4].replace('form', '')
            for key in df2.keys()
            if not key == 'id'
        }

        df2.rename(columns=rename_dict, inplace=True)
        df = pd.merge(df, df2, on='id', how=how)
    print('final shape', df.shape)
    # df.to_pickle(oj(pdir, 'features.pkl'))
    return df

def get_outcomes(NUM_PATIENTS=12044):
    '''
    Returns
    -------
    outcomes: pd.DataFrame
        iai (has 761 positives)
        iai_intervention (has 203 positives)
    '''
    form4abdangio = pd.read_csv('iaip_data/Datasets/form4bother_abdangio.csv').rename(columns={'subjectid': 'id'})
    form6a = pd.read_csv('iaip_data/Datasets/form6a.csv').rename(columns={'subjectid': 'id'})
    form6b = pd.read_csv('iaip_data/Datasets/form6b.csv').rename(columns={'SubjectID': 'id'}) 
    form6c = pd.read_csv('iaip_data/Datasets/form6c.csv').rename(columns={'subjectid': 'id'})

    # (6b) Intra-abdominal injury diagnosed in the ED/during hospitalization by any diagnostic method
    # 1 is yes, 761 have intra-abdominal injury
    # 2 is no -> remap to 0, 841 without intra-abdominal injury
    idxs_iai = form6b.id[form6b['IAIinED1'] == 1]
    iai = np.zeros(NUM_PATIENTS).astype(np.int)
    iai[idxs_iai] = 1

    
    def get_ids(form, keys):
        ids_all = set()
        for key in keys:
            ids = form.id.values[form[key] == 1]
            for i in ids:
                ids_all.add(i)     
#             print(key, np.sum(form[key] == 1), np.unique(ids).size)            
        return ids_all

    

    # print(form4abdangio.keys())
    ids_allangio = get_ids(form4abdangio, ['AbdAngioVessel'])
    # print('num in 4angio', len(ids_allangio))
    # print(form6a.keys())
    # ids_alla = get_ids(form6a, ['DeathCause'])
    # print('num in a', len(ids_alla))
    # print(form6b.keys())
    ids_allb = get_ids(form6b, ['IVFluids', 'BldTransfusion'])
    # print('num in b', len(ids_allb))
    # print(form6c.keys())
    ids_allc = get_ids(form6c, ['IntervenDurLap'])
    # print('num in c', len(ids_allc))

    ids = ids_allb.union(ids_allangio).union(ids_allc)
    ids_np = np.array(list(ids)) - 1
    iai_intervention = np.zeros(NUM_PATIENTS).astype(np.int)
    iai_intervention[ids_np] = 1
#     print('num total', len(ids))
    # print(len(ids_allb.union(ids_allc)))
    
    
    df_iai = pd.DataFrame.from_dict({
        'id': np.arange(1, NUM_PATIENTS + 1),
        'iai': iai,
        'iai_intervention': iai_intervention
    })
    return df_iai