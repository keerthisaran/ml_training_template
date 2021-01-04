import pandas as pd
import numpy as np
import config
from sklearn import preprocessing
import model_dispatcher
from sklearn import metrics
import joblib
import os

def train(fold,model_key):

    df=pd.read_csv(config.TRAINING_FOLDFILEPATH)

    cat_cols=[col for col in df.columns if col not in ['id','label','kfold','target']]

    for col in cat_cols:
        df[col]=df[col].astype(str).fillna('None')
    
    for col in cat_cols:
        lbe=preprocessing.LabelEncoder()
        lbe.fit(df[col])
        df[col]=lbe.transform(df[col])
    df_train=df.loc[df.kfold!=fold].reset_index(drop=True)
    df_valid=df.loc[df.kfold==fold].reset_index(drop=True)

    X_train=df_train[cat_cols]
    y_train=df_train['target']

    X_valid=df_valid[cat_cols]
    y_valid=df_valid['target']

    model=model_dispatcher.models[model_key]

    model.fit(X=X_train,y=y_train)

    y_pred=model.predict(X_valid)

    print(metrics.roc_auc_score(y_pred,y_valid))

    model_save_filename=os.path.join(config.MODEL_SAVE_DIR,f'{model_key}_{fold}.joblib')

    joblib.dump(model,model_save_filename)



    



if __name__=='__main__':

    for i in range(5):
        train(i,'xgb')



    
    