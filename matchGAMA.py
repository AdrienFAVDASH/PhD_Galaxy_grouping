from astropy.io import fits
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from astropy.table import Table

def start(df,rec_name):
    reclengths=[]
    recIDs = df.groupby(rec_name).count().mean(axis=1)
    for i in (recIDs.index):
        reclengths.append(int(recIDs[i]))
    dfrec = pd.DataFrame()
    dfrec['N'] = recIDs
    dfrec['N'] = dfrec['N'].astype('int')
    dfrec.index.names=[rec_name]
    dfrec = dfrec[dfrec['N'] != 1]
    dfrec = dfrec[dfrec.index != 0]
        
    recdict={}
    for i in tqdm(dfrec.index):
        recdict[i] = (df[df[rec_name] == i].index.values).tolist()
    
    return dfrec, recdict