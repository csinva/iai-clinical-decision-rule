import os
from os.path import join as oj
import sys
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import numpy as np
from tqdm import tqdm
import pandas as pd
NUM_PATIENTS = 12044
import data


def get_data(use_processed=False, processed_file='processed/df_psrc.pkl'):
    '''Run all the preprocessing
    
    Params
    ------
    use_processed: bool, optional
        determines whether to load df from cached pkl
    save_processed: bool, optional
        if not using processed, determines whether to save the df
    '''
    if use_processed and os.path.exists(processed_file):
        return pd.read_pickle(processed_file)
    else:
        print('computing psrc preprocessing...')
        
        data_file='data_psrc/psrc_data_processed.csv'
        df = pd.read_csv(data_file)


        NUM_PATIENTS = df.shape[0]

        # fix col names
        df['id'] = -1 * np.arange(1, NUM_PATIENTS + 1)
        df = df.rename(columns=lambda x: x.replace('choice=0ne', 'choice=None').strip())
        df = df.replace('0ne', 'None')

        # rename values
        df = rename_values(df)
        
        # drop unnecessary
        ks_drop = [k for k in df.keys() 
                   if k in ['Time patient trauma alert concluded/ left trauma bay (military time)',
                            'Time CT performed',
                           ]]
        df = df.drop(columns=ks_drop + ['Record ID',  'Arrival time in ED', 'Age in months', 'Age in years'])
        
        # outcomes
        iai_keys = [k for k in df.keys() if 'Interventions for IAI' in k]
        iai_with_intervention_keys = [k for k in iai_keys if not 'choice=None' in k]
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
        df = data.add_dummies_and_cv_split(df, dset='psrc')
        # df = df.fillna('unknown')
        
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
    df['GCSScore'] = (df['GCSScore'].fillna(df['GCSScore'].median())).astype(int)
    df['Age'] = df['Age in years'].fillna(0) + df['Age in months'].fillna(0) / 12
    df['InitSysBPRange'] = df['Initial ED systolic BP'].fillna(df['Initial ED systolic BP'].median()).astype(int)
    # these need matching
    df['InitHeartRate'] = df['Initial ED HR'].fillna(df['Initial ED HR'].median())
    binary = {
        0: 'no',
        1: 'yes',
        False: 'no',
        True: 'yes',
        'unknown': 'unknown'
    }
    df['SeatBeltSign'] = df['SeatBeltSign'].map(binary)
    df['AbdDistention'] = df['Abdominal distension'].fillna('unknown').map(binary)
    df['VomitWretch'] = df['Emesis post injury'].fillna('unknown').map(binary)
    df['AbdTrauma'] = df['Evidence of abdominal wall trauma (choice=None)'].map(binary)
    df['AbdomenPain'] = (df['Complainabd. pain']!='0').astype(int).map(binary).fillna('other')
    df['ThoracicTrauma'] = (1 - df['Evidence of thoracic trauma  (choice=None)']).map(binary)
    df['DecrBreathSound'] = df['Evidence of thoracic trauma  (choice=Decreased breath sounds)'].map(binary)
    # df['FemurFracture'] = df['Femur fracture'] #.map(binar)
    df['Hypotension'] = (df['Age'] < 1/12) & (df['InitSysBPRange'] < 70) | \
                    (df['Age'] >= 1/12) & (df['Age'] < 5) & (df['InitSysBPRange'] < 80) | \
                    (df['Age'] >= 5) & (df['InitSysBPRange'] < 90).map(binary)

    abdTenderDegree = {
        'None': 'Mild',
        'Mild': 'Mild',
        'Moderate': 'Moderate',
        'Severe': 'Severe',
        'Limited exam secondary to intubation/sedation': 'unknown',
        'unknown': 'unknown'
    }
    df['AbdTenderDegree'] = df['Abdominal tenderness to palpation'].fillna('unknown').map(abdTenderDegree)

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
    df['RecodedMOI'] = ['unknown'] * df.shape[0]
    
    for k in moi:
        df.loc[df[k] == 1, 'RecodedMOI'] = moi[k]
    return df