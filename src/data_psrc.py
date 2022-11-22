import data
import pandas as pd
import numpy as np
import os
import sys
from os.path import join as oj
from config import PROCESSED_DIR, PSRC_DIR
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path


def get_data(use_processed=False,
             use_processed_feats=False,
             processed_file=oj(PROCESSED_DIR, 'df_psrc.pkl'), dummy=False, impute_feats=True):
    '''Run all the preprocessing

    Params
    ------
    use_processed: bool, optional
        determines whether to load df from cached pkl (only for reading from the csv
    save_processed: bool, optional
        if not using processed, determines whether to save the df
    '''
    if use_processed and os.path.exists(processed_file):
        return pd.read_pickle(processed_file)
    else:
        data_file = oj(PSRC_DIR, 'psrc_data_processed.csv')
        df = pd.read_csv(data_file)

        # fix col names
        df['id'] = -1 * np.arange(1, df.shape[0] + 1)
        df = df.rename(columns=lambda x: x.replace(
            'choice=0ne', 'choice=None').strip())
        df = df.replace('0ne', 'None')

        # rename values
        df = rename_values(df)
        df = data.derived_feats(df)
        if impute_feats:
            df = impute(df)

        # drop unnecessary
        ks_drop = [k for k in df.keys()
                   if k in ['Time patient trauma alert concluded/ left trauma bay (military time)',
                            'Time CT performed',
                            ]]
        df = df.drop(
            columns=ks_drop + ['Record ID', 'Arrival time in ED', 'Age in months', 'Age in years'])

        # outcomes
        iai_keys = [k for k in df.keys() if 'Interventions for IAI' in k]
        iai_with_intervention_keys = [
            k for k in iai_keys if not 'choice=None' in k]
        outcomes = ['Admission',
                    'ICU admission',
                    'Length of inpatient stay (days)',
                    'Delayed inpatient diagnosis of IAI (more than 24 hours after admission)',
                    'Mortality (within 30 days of injury)',
                    'Mortality related to trauma',
                    'Mortality secondary to intra-abdominal injury',
                    'Missed diagnosis of IAI (following discharge)'
                    ] + iai_keys

        # iai
        df['iai'] = df[iai_keys].sum(axis=1) > 0
        df['iai_intervention'] = df[iai_with_intervention_keys].sum(axis=1) > 0

        df = df.infer_objects()
        df = data.add_cv_split(df, dset='psrc')
        if dummy:
            df = data.to_dummies(df)
        df['dset'] = 'psrc'

        # save
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        df.to_pickle(processed_file)
        return df


def rename_values(df):
    '''Map values to meanings
    Rename some features
    Compute a couple new features
    set types of 
    '''
    df = df.rename(columns={
        'Seatbelt sign': 'SeatBeltSign',
        'Initial GCS': 'GCSScore',
        'Lower chest wall/costal margin tenderness to palpation  (choice=1 on left)': 'LtCostalTender',
        'Lower chest wall/costal margin tenderness to palpation  (choice=1 on right)': 'RtCostalTender'
    })

    # fill with median

    df['Age'] = df['Age in years'].fillna(
        0) + df['Age in months'].fillna(0) / 12
    df['InitSysBPRange'] = df['Initial ED systolic BP']
    df['InitHeartRate'] = df['Initial ED HR']
    df['FemurFracture'] = df['Femur fracture'].sum(axis=1)
    binary = {
        0: 'no',
        1: 'yes',
        False: 'no',
        True: 'yes',
        'unknown': 'unknown'
    }
    df['Race_orig'] = df['Race'].fillna('unknown')
    df['Hispanic'] = df['Hispanic ethnicity'].fillna('unknown').map(binary)
    df['SeatBeltSign'] = df['SeatBeltSign'].map(binary)
    df['AbdDistention'] = df['Abdominal distension'].fillna(
        'unknown').map(binary)
    df['VomitWretch'] = df['Emesis post injury'].fillna('unknown').map(binary)
    df['AbdTrauma'] = (
        1 - df['Evidence of abdominal wall trauma (choice=None)']).map(binary)
    df['AbdomenPain'] = (df['Complainabd. pain'] !=
                         '0').astype(int).map(binary)
    df['ThoracicTrauma'] = (
        1 - df['Evidence of thoracic trauma  (choice=None)']).map(binary)
    df['DecrBreathSound'] = df['Evidence of thoracic trauma  (choice=Decreased breath sounds)'].map(
        binary)

    df['DistractingPain'] = np.array([False] * df.shape[0])
    for k in ['Chest X-ray (choice=Rib fracture)',
              'Indicate thoracic injury (choice=Clavicle fracture)',
              'Chest X-ray (choice=Scapula fracture)',
              'FemurFracture', 'Pelvic fracture']:
        df['DistractingPain'] = df['DistractingPain'] | df[k]
    # df['FemurFracture'] = df['Femur fracture'] #.map(binar)

    abdTenderDegree = {
        'None': 'None',
        'Mild': 'Mild',
        'Moderate': 'Moderate',
        'Severe': 'Severe',
        'Limited exam secondary to intubation/sedation': 'Severe',  # probably severe
        'unknown': 'None'
    }
    df['AbdTenderDegree'] = df['Abdominal tenderness to palpation'].fillna(
        'None').map(abdTenderDegree)

    moi = {
        'Mechanism of injury (choice=Assault/struck)': 'Object struck abdomen',
        'Mechanism of injury (choice=ATV injury)': 'Motorcycle/ATV/Scooter collision',
        'Mechanism of injury (choice=Bike crash)': 'Bike collision/fall',
        'Mechanism of injury (choice=Bike struck by auto)': 'Pedestrian/bicyclist struck by moving vehicle',
        'Mechanism of injury (choice=Fall > 10 ft. height)': 'Fall from an elevation',
        'Mechanism of injury (choice=Golf cart injury)': 'Motorcycle/ATV/Scooter collision',
        'Mechanism of injury (choice=Motorcycle/dirt bike crash)': 'Motorcycle/ATV/Scooter collision',
        'Mechanism of injury (choice=MVC)': 'Motor vehicle collision',
        'Mechanism of injury (choice=Pedestrian struck by auto)': 'Pedestrian/bicyclist struck by moving vehicle',
        'Mechanism of injury (choice=Other blunt mechanism)': 'Object struck abdomen',
    }
    df['MOI'] = ['unknown'] * df.shape[0]

    for k in moi:
        df.loc[df[k] == 1, 'MOI'] = moi[k]

    df['CTScan'] = df['Abdominal CT scan performed']

    return df


def impute(df: pd.DataFrame):
    """Returns df with imputed features
    """
    # filling some continuous vars with median
    df['GCSScore'] = (df['GCSScore'].fillna(
        df['GCSScore'].median())).astype(int)
    df['InitSysBPRange'] = df['InitSysBPRange'].fillna(
        df['InitSysBPRange'].median()).astype(int)
    df['InitHeartRate'] = df['InitHeartRate'].fillna(
        df['InitHeartRate'].median())

    # other vars get specific imputations
    # df['AbdTenderDegree'] = df['AbdTenderDegree'].fillna('None')
    df['AbdomenPain'] = df['AbdomenPain'].fillna('other')
    return df

def get_FAST(d):
    fast_recieved = d['FAST (choice=not performed)'] == 0
    # fast_interpretation_known = (d['UltrasoundRes'] != 3) # some patients have "no interpretetation"
    fast_study_cohort = fast_recieved # & fast_interpretation_known
    abnormal = ~(d['FAST (choice=Negative)'] == 1)
    fast_abnormal = fast_study_cohort & abnormal
    # ((d['UltrasoundType'] == 1) & (d['UltrasoundRes'] == 3) & (d['iai'] == 1)).sum() # look at patients with no interpretation
    return fast_study_cohort, fast_abnormal