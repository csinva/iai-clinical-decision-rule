import os
from os.path import join as oj
import sys
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import numpy as np
from tqdm import tqdm
import pandas as pd
NUM_PATIENTS = 12044
import data


def get_data(use_processed=False, save_processed=True, processed_file='processed/df_psrc.pkl'):
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
        df['Age'] = df['Age in years'].fillna(0) + df['Age in months'].fillna(0) / 12

        # drop unnecessary
        ks_drop = [k for k in df.keys() 
                   if k in ['Time patient trauma alert concluded/ left trauma bay (military time)',
                            'Time CT performed',
                           ]]
        df = df.drop(columns=ks_drop + ['Record ID',  'Arrival time in ED', 'Age in months', 'Age in years'])

        # rename values
        df = rename_values(df)
        # other vars present
        # Race
        # Thoracic injury - matches ThoracicTrauma
        
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
        
        vals = {
            0: 'no',
            1: 'yes'
        }
        
        df['SeatBeltSign'] = [vals[x] for x in df['SeatBeltSign'].values]
        
        df = df.infer_objects()
        df = data.classification_setup(df, dset='psrc')
        # df = df.fillna('unknown')
        
        if save_processed:
            df.to_pickle(processed_file)
        return df
    


def rename_values(df):
    '''Map values to meanings
    Rename some features
    Compute a couple new features
    set types of 
    '''
    df = df.rename(columns={'Seatbelt sign': 'SeatBeltSign', 
                                'Initial GCS': 'GCSScore',
                                'Lower chest wall/costal margin tenderness to palpation  (choice=1 on left)': 'LtCostalTender',
                                'Lower chest wall/costal margin tenderness to palpation  (choice=1 on right)': 'RtCostalTender'
                               })
    # fill with unknown
    df['GCSScore'] = (df['GCSScore'].fillna(df['GCSScore'].median())).astype(int)


    # these need matching
    df = df.rename(columns={'Abdominal distension': 'AbdDistention',
                            'Abdominal tenderness to palpation': 'AbdTenderDegree',
                           })
    binary = {
        0: 'no',
        1: 'yes',
        'unknown': 'unknown'
    }
    df['AbdDistention'] = [binary[v] for v in df.AbdDistention.fillna('unknown').values]

    abdTenderDegree = {
        'None': 'Mild',
        'Mild': 'Mild',
        'Moderate': 'Moderate',
        'Severe': 'Severe',
        'Limited exam secondary to intubation/sedation': 'unknown',
        'unknown': 'unknown'
    }
    df['AbdTenderDegree'] = [abdTenderDegree[v] for v in df.AbdTenderDegree.fillna('unknown').values]

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