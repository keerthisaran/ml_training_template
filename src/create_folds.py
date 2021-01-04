import pandas as pd
from sklearn import model_selection
import numpy as np
import config
import os

def create_folds(k=5,label='y'):
    train_file=config.TRAINING_FILEPATH
    train_df=pd.read_csv(train_file)
    train_df['kfold']=-1
    y=train_df[label]
    train_df=train_df.rename(columns={label:'target'})
    train_df=train_df.sample(frac=1).reset_index(drop=True)

    kf=model_selection.StratifiedKFold(n_splits=k)
    for fold,(train_ids,valid_ids) in enumerate(kf.split(train_df,y)):
        train_df.loc[valid_ids,'kfold']=fold
    
    train_df.to_csv(config.TRAINING_FOLDFILEPATH,index=False)
    


if __name__=='__main__':
    create_folds(5,'target')


