from matplotlib_venn import venn2
def venn_overlap(col1: str, val1, col2: str, val2, df):
    cind = df[df['LtCostalTender']==val1].index.values
    rind = df[df['CostalTender']==val2].index.values
    venn2((set(cind), set(rind)), (col1, col2))