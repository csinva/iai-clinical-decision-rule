# directories
import numpy as np
import pandas as pd

import data_pecarn
import data_psrc

pecarn_train_idxs = [1, 2, 3, 4]
pecarn_test_idxs = [5, 6]
psrc_train_idxs = [8, 9, 10, 11]
psrc_test_idxs = [12, 13]

# common feats
feats_numerical = ['InitSysBPRange', 'InitHeartRate', 'GCSScore', 'Age']
feats_categorical = ['AbdTenderDegree', 'Race', 'MOI']
meta = ['iai_intervention', 'cv_fold', 'dset']
outcome_def = 'iai_intervention'  # output



def load_it_all(dummy=True, impute=True, frac_missing_allowed=0.1, use_processed=True):
    df_pecarn = data_pecarn.get_data(use_processed=use_processed,
                                     frac_missing_allowed=frac_missing_allowed,
                                     dummy=dummy,
                                     impute_feats=impute)
    all_feats_pecarn, filtered_feats_pecarn = get_feat_names(df_pecarn)
    try:
        df_psrc = data_psrc.get_data(use_processed=use_processed, dummy=dummy, impute_feats=impute)
        all_feats_psrc, filtered_feats_psrc = get_feat_names(df_psrc)
        common_feats = meta + list(filtered_feats_pecarn.intersection(filtered_feats_psrc))
    except:
        print('PSRC data not loaded (not public)')
        df_psrc = df_pecarn[df_pecarn.cv_fold > 100] # select 0 rows
        filtered_feats_psrc = None
        common_feats =  ['AbdDistention_or_AbdomenPain_yes',
                         'AbdTenderDegree_None',
                         'AbdTrauma_or_SeatBeltSign_yes',
                         'Age<2_yes',
                         'CostalTender_yes',
                         'DecrBreathSound_yes',
                         'GCSScore_Full_yes',
                         'Hypotension_yes',
                         'MOI_Bike collision/fall',
                         'MOI_Fall from an elevation',
                         'MOI_Motor vehicle collision',
                         'MOI_Motorcycle/ATV/Scooter collision',
                         'MOI_Object struck abdomen',
                         'MOI_Pedestrian/bicyclist struck by moving vehicle',
                         'ThoracicTrauma_yes',
                         'VomitWretch_yes'] + meta
        
    

    feats_binary = [feat for feat in common_feats
                    if not feat in feats_numerical + feats_categorical + meta]
    return df_pecarn, df_psrc, common_feats, filtered_feats_pecarn, filtered_feats_psrc


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
    df['AbdDistention_or_AbdomenPain'] = ((df.AbdDistention == 'AbdomenPain') | (df.SeatBeltSign == 'yes')).map(binary)
    df['Hypotension'] = (df['Age'] < 1 / 12) & (df['InitSysBPRange'] < 70) | \
                        (df['Age'] >= 1 / 12) & (df['Age'] < 5) & (df['InitSysBPRange'] < 80) | \
                        (df['Age'] >= 5) & (df['InitSysBPRange'] < 90)
    df['Hypotension'] = df['Hypotension'].map(binary)
    df['GCSScore_Full'] = (df['GCSScore'] == 15).map(binary)
    df['Age<2'] = (df['Age'] < 2).map(binary)
    df['CostalTender'] = ((df.LtCostalTender == 1) | (df.RtCostalTender == 1)).map(binary)  # | (df.DecrBreathSound)

    # Combine hispanic as part of race
    df['Race'] = df['Race_orig']
    df.loc[df.Hispanic == 'yes', 'Race'] = 'Hispanic'
    df.loc[df.Race == 'White', 'Race'] = 'White (Non-Hispanic)'

    return df


def remove_from_list(l, removes):
    '''deletes all elements in removes from the list l and returns
    '''
    return [x for x in l
            if not x in removes]

def select_final_feats(feat_names,
                       collapse_abd_tender=True,
                       collapse_abd_distention=True,
                       collapse_age=True):
    '''Return an interpretable set of the best features
    '''
    feat_names = [f for f in feat_names
                  if not f in meta
                  and not f.endswith('_no')
                  and not 'Race' in f
                  and not 'other' in f.lower()
                  and not 'unknown' in f.lower()
                  ]
    feat_names = remove_from_list(feat_names, ['LtCostalTender', 'RtCostalTender'])
    feat_names = remove_from_list(feat_names, ['AbdTrauma_yes', 'SeatBeltSign_yes'])
    feat_names = remove_from_list(feat_names, ['GCSScore'])
    feat_names = remove_from_list(feat_names, ['InitHeartRate', 'InitSysBPRange']) # remove these so we can only have binary vars
    
    
    # make abd tender into a None or not-None variable
    if collapse_abd_tender:
        feat_names = remove_from_list(feat_names, ['AbdTenderDegree_Mild', 'AbdTenderDegree_Moderate', 'AbdTenderDegree_Severe'])
        
    # whether to combine AbdomenPain and AbdDistention
    if collapse_abd_distention:
        feat_names = remove_from_list(feat_names, ['AbdomenPain_yes', 'AbdDistention_yes'])
    else:
        feat_names = remove_from_list(feat_names, ['AbdDistention_or_AbdomenPain_yes'])
    
    if collapse_age:
        feat_names = remove_from_list(feat_names, ['Age'])
    else:
        feat_names = remove_from_list(feat_names, ['Age<2_yes'])
        
    
        
    return sorted(feat_names)


fewest_feats = [
    #     'AbdDistention_yes',
    'AbdTenderDegree_None',
    'AbdTrauma_or_SeatBeltSign_yes',
#     'AbdomenPain_yes',
#     'Age',
    'CostalTender_yes',
    'DecrBreathSound_yes',
    'GCSScore_Full_yes',
    'MOI_Fall from an elevation',
    'MOI_Motor vehicle collision',
    'MOI_Motorcycle/ATV/Scooter collision',
    #  'MOI_Pedestrian/bicyclist struck by moving vehicle',
    'ThoracicTrauma_yes',
    'VomitWretch_yes']


def add_cv_split(df: pd.DataFrame, dset='pecarn'):
    # set up train / test
    np.random.seed(1)
    if dset == 'pecarn':
        offset = 0
    elif dset == 'psrc':
        offset = 7
    df['cv_fold'] = np.random.randint(1, 7, size=df.shape[0]) + offset
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
                         'Age',
                         'Sex'] + \
                        ['Race', 'InitHeartRate', 'InitSysBPRange']  # new ones to consider
    pecarn_feats = set()
    for pecarn_feat in PECARN_FEAT_NAMES:
        for feat_name in feat_names:
            if pecarn_feat in feat_name:
                pecarn_feats.add(feat_name)
    pecarn_feats = sorted(list(pecarn_feats))
    return feat_names, set(pecarn_feats)


def get_sample_weights(df, df_pecarn, df_psrc, balancing_ratio):
    '''Get sample weights which also account for age / gender
    '''
    # class weights
    class_weights = {0: 1, 1: balancing_ratio}
    sample_weights_class = pd.Series(df[outcome_def]).map(class_weights).values

    # weights for different risk populations
    age_discrete = pd.cut(df['Age'], bins=(-1, 4, 9, 1000), labels=['<5', '5-9', '>9']).values
    # we don't have sex for psrc, so just fill in 0 (only matters for training anyway)
    sex = pd.Series(np.hstack((df_pecarn['Sex_M'].values, np.zeros(df_psrc.shape[0])))).map({0: 'F', 1: 'M'}).values
    risk_identity = [(sex[i], age_discrete[i]) for i in range(age_discrete.shape[0])]

    risk_weights = {
        ('F', '<5'): 33.9, ('F', '5-9'): 25.8, ('F', '>9'): 27.2,
        ('M', '<5'): 14.8, ('M', '5-9'): 13.7, ('M', '>9'): 13.1
    }
    sample_weights_identity = pd.Series(risk_identity).map(risk_weights).values
    sample_weights = sample_weights_class * sample_weights_identity  # elementwise multiply
    return sample_weights
