import sys

from tqdm import tqdm

import data
import train
from data import outcome_def

if __name__ == '__main__':
    job_num = None
    if len(sys.argv) > 1:
        job_num = int(sys.argv[1])

    # load data
    df_pecarn, df_psrc, common_feats, filtered_feats_pecarn, filtered_feats_psrc = data.load_it_all(dummy=True)
    df = df_pecarn[common_feats].append(df_psrc[common_feats])
    processed_feats = data.select_final_feats(common_feats)
    print(len(processed_feats), sorted(processed_feats))


    # params
    class p:
        out_dir = f'results/aug5_10'
        balancing = ['sample_weights']  # 'ros', 'smote', sample_weights
        balancing_ratio = [1000, 750, 500, 100]
        # options: brl, slim, grl, rulefit, logistic, dt, 'rf', 'mlp2', 'svm', 'gb'
        model_type = ['logistic', 'dt', 'slim', 'grl',
                      'brl']  # ['logistic', 'dt'] # ['grl', 'slim', 'brl', 'rulefit'] #, 'rf', 'mlp2', 'svm']): #
        feature_selection = ['select_stab_lasso', 'select_lasso',
                             'select_rf']  # , 'select_lasso', 'select_rf']: # select_lasso, select_rf, None
        feature_selection_nums = [5, 6, 7, 10, 100]
        train_idxs = data.pecarn_train_idxs
        test_idxs1 = data.pecarn_test_idxs
        test_idxs2 = data.psrc_train_idxs + data.psrc_test_idxs


    print('possible combos', len(p.balancing) * len(p.balancing_ratio) * len(p.model_type) * len(
        p.feature_selection * len(p.feature_selection_nums)))

    # predict
    job_counter = 0
    for balancing in p.balancing:
        for balancing_ratio in p.balancing_ratio:
            sample_weights = data.get_sample_weights(df, df_pecarn, df_psrc, balancing_ratio)
            for model_type in p.model_type:
                for feature_selection in p.feature_selection:
                    for feature_selection_num in tqdm(p.feature_selection_nums):

                        # if job_num is passed, only run one job
                        if job_num is None or job_num == job_counter:
                            out_name = f'{model_type}_{feature_selection}={feature_selection_num}_{balancing}={balancing_ratio}'
                            train.train(df,
                                        feat_names=processed_feats,
                                        model_type=model_type,
                                        balancing=balancing,
                                        outcome_def=outcome_def,
                                        sample_weights=sample_weights,
                                        balancing_ratio=balancing_ratio,
                                        out_name=f'{p.out_dir}/{out_name}.pkl',
                                        feature_selection=feature_selection,
                                        feature_selection_num=feature_selection_num,
                                        train_idxs=p.train_idxs,
                                        test_idxs1=p.test_idxs1,
                                        test_idxs2=p.test_idxs2)
