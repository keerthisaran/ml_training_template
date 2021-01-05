import pandas as pd
import numpy as np
import config
from sklearn import preprocessing
import model_dispatcher
from sklearn import metrics
import joblib
from sklearn import linear_model
import os

import tensorflow as tf


def create_emb_model(data,cat_cols):
    inputs=[]
    outputs=[]

    for col in cat_cols:
        nunique=data[col].nunique()
        emb_dims=min((nunique+1)//2,50)

        input=tf.keras.layers.Input(shape=(1,))

        emb_layer=tf.keras.layers.Embedding(input_dim=nunique+1,output_dim=emb_dims)
        emb_out=emb_layer(input)
        out=tf.keras.layers.SpatialDropout1D(0.3)(emb_out)
        out=tf.keras.layers.Reshape(target_shape=(emb_dims,))(out)

        inputs.append(input)
        outputs.append(out)
    
    outs=tf.keras.layers.Concatenate(axis=1)(outputs)
    outs=tf.keras.layers.BatchNormalization()(outs)
    
    outs=tf.keras.layers.Dense(300,activation='relu')(outs)
    outs=tf.keras.layers.Dropout(0.3)(outs)
    outs=tf.keras.layers.BatchNormalization()(outs)

    outs=tf.keras.layers.Dense(300,activation='relu')(outs)
    outs=tf.keras.layers.Dropout(0.3)(outs)
    outs=tf.keras.layers.BatchNormalization()(outs)


    
    y=tf.keras.layers.Dense(2,activation='softmax')(outs)

    model=tf.keras.models.Model(inputs,y)
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model



def train(fold):

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

    model=create_emb_model(df,cat_cols)

    X_train=[df_train[col].values for col in cat_cols]
    X_valid=[df_valid[col].values for col in cat_cols]

    y_train=df_train['target'].values
    y_valid=df_valid['target'].values

    y_train_cat=tf.keras.utils.to_categorical(y_train)
    y_valid_cat=tf.keras.utils.to_categorical(y_valid)

    model.fit(X_train,y_train_cat,
              validation_data=(X_valid,y_valid_cat),
              verbose=1,
              batch_size=1024,
              epochs=5)
    y_prob_cat=model.predict(X_valid)
    y_pred=y_prob_cat.argmax(axis=-1)
    
    print(metrics.roc_auc_score(y_pred,y_valid))
    weights_path=os.path.join(config.MODEL_SAVE_DIR,f'emb_weights_fold_{fold}/')
    model.save_weights(weights_path)

    


if __name__=='__main__':
    for fold in range(5):
        train(fold)




    
    