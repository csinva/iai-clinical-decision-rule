import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from tqdm import tqdm

def venn_overlap(df, col1: str, col2: str, val1=1, val2=1):
    '''Plots venn diagram of overlap between 2 cols with values specified
    '''
    cind = df[df[col1]==val1].index.values
    rind = df[df[col2]==val2].index.values
    venn2((set(cind), set(rind)), (f'{col1} ({str(val1)})', f'{col2} ({str(val2)})'))
    
    
def sensitivity_specificity_curve(y_test, preds_proba, plot=True):
    '''
    preds_proba should be 1d
    '''
    thresholds = sorted(np.unique(preds_proba))
    sens = []
    spec = []
    for threshold in tqdm(thresholds):
        preds = preds_proba > threshold
        stats = sklearn.metrics.classification_report(y_test, preds, output_dict=True)
        sens.append(stats['1']['recall'])
        spec.append(stats['0']['recall'])
    
    if plot:
        plt.plot(sens, spec, '.-')
        plt.xlabel('sensitivity')
        plt.ylabel('specificity')
        plt.grid()
    return sens, spec