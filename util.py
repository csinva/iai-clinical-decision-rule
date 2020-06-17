from matplotlib_venn import venn2
def venn_overlap(df, col1: str, col2: str, val1=1, val2=1):
    '''Plots venn diagram of overlap between 2 cols with values specified
    '''
    cind = df[df[col1]==val1].index.values
    rind = df[df[col2]==val2].index.values
    venn2((set(cind), set(rind)), (f'{col1} ({str(val1)})', f'{col2} ({str(val2)})'))