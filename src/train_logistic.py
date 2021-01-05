import pandas as pd
import numpy as np
import config
from sklearn import preprocessing
import model_dispatcher
from sklearn import metrics
import joblib
from sklearn import linear_model
import os

def train(fold,model_key):

    df=pd.read_csv(config.TRAINING_FOLDFILEPATH)

    cat_cols=[col for col in df.columns if col not in ['id','label','kfold','target']]
    
    ohe_df=pd.DataFrame()

    # for col in cat_cols:
    df[cat_cols]=df[cat_cols].astype(str).fillna('None')
    ohe=preprocessing.OneHotEncoder()
    ohe.fit(df[cat_cols])
    ohe_df=ohe.transform(df[cat_cols])

    X_train=ohe_df[df.kfold!=fold]
    X_valid=ohe_df[df.kfold==fold]

    y_train=df.loc[df.kfold!=fold,'target']
    y_valid=df.loc[df.kfold==fold,'target']

    model=linear_model.LogisticRegression()

    model.fit(X=X_train,y=y_train)
    y_pred=model.predict(X_valid)

    print(metrics.roc_auc_score(y_pred,y_valid))

    model_save_filename=os.path.join(config.MODEL_SAVE_DIR,f'{model_key}_{fold}.joblib')

    joblib.dump(model,model_save_filename)


if __name__=='__main__':

    for i in range(5):
        train(i,'xgb')



    
    