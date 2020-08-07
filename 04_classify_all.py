import sys

from tqdm import tqdm

import data
import train
from data import outcome_def

# params
class p:
    out_dir = 'results/aug7_logistics'
    balancing = ['sample_weights']  # 'ros', 'smote', sample_weights
    balancing_ratio = [1, 10, 100, 500, 750, 1000]
    # options: brl, slim, grl, rulefit, logistic, dt, 'rf', 'mlp2', 'svm', 'gb'
    model_type = ['logistic'] #, 'dt', 'grl', 'slim', 'brl']
    # , 'select_lasso', 'select_rf']: # select_lasso, select_rf, None
    feature_selection = ['select_stab_lasso', 'select_lasso', 'select_rf']  
    feature_selection_nums = [5, 6, 7, 10, 20]
    train_idxs = data.pecarn_train_idxs
    test_idxs1 = data.pecarn_test_idxs
    test_idxs2 = data.psrc_train_idxs + data.psrc_test_idxs
    collapse_abd_tender = [True, False]
    collapse_abd_distention = [True, False]
    hyperparam = [0, 1, 2, 3] # different hyperparams

if __name__ == '__main__':
    job_num = None
    if len(sys.argv) > 1:
        job_num = int(sys.argv[1])
    print('possible combos', len(p.balancing) * len(p.balancing_ratio) * len(p.model_type) * len(p.feature_selection * len(p.feature_selection_nums)))

    # load data
    df_pecarn, df_psrc, common_feats, filtered_feats_pecarn, filtered_feats_psrc = data.load_it_all(dummy=True)
    df = df_pecarn[common_feats].append(df_psrc[common_feats])

    # predict
    job_counter = 0
    for collapse_abd_tender in p.collapse_abd_tender:
        for collapse_abd_distention in p.collapse_abd_distention:
            processed_feats = data.select_final_feats(common_feats, p.collapse_abd_tender, p.collapse_abd_distention)
#             print(len(processed_feats), sorted(processed_feats))
            for balancing in p.balancing:
                for balancing_ratio in p.balancing_ratio:
                    sample_weights = data.get_sample_weights(df, df_pecarn, df_psrc, balancing_ratio)
                    for model_type in p.model_type:
                        for feature_selection in p.feature_selection:
                            for feature_selection_num in p.feature_selection_nums:
                                for hyperparam in p.hyperparam:
                                    if hyperparam == 0 or model_type == 'logistic': # only vary hyperparams for logistic
                                        # if job_num is passed, only run one job
                                        if job_num is None or job_num == job_counter:
                                            out_name = f'{model_type}_{feature_selection}={feature_selection_num}_{balancing}={balancing_ratio}_h={hyperparam}_c1={collapse_abd_tender}_c2={collapse_abd_distention}'
                                            print('training', out_name)

                                            train.train(df,
                                                        feat_names=processed_feats,
                                                        model_type=model_type,
                                                        hyperparam=hyperparam,
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
                                            print('success!', out_name)
                                        job_counter += 1