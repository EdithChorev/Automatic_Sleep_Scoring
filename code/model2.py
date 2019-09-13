import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,Model
from keras.layers import  Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import Concatenate, concatenate,BatchNormalization
from keras.layers import  GRU, Input, LSTM, TimeDistributed, Bidirectional
from keras.callbacks import History
import keras.backend as K
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from itertools import product
import json
import pickle


def get_data(fold): ## cnn only part
    name='/home/edith/Documents/EEG/2018/folds/fold'+str(fold)+'_X.npy'
    x = np.load(name)
    name='/home/edith/Documents/EEG/2018/folds/fold'+str(fold)+'_Y.npy'
    y = np.load(name)
    
    xx=np.array(np.split(x,x.shape[0]//3,axis = 0))
    shuffle = np.random.permutation(len(xx))  
    xx = xx[shuffle]
    y = y[shuffle]
    x= []
    for i in range (len(xx)):
        tmp = xx[i]
        for j in np.arange(0,30000,3000):
            x.append(tmp[:,j:j+3000])  
    
    yy = np.split(y,y.shape[0],axis = 0)
    y=[]
    for i in range (len(yy)):
        tmp = yy[i]
        for j in np.arange(0,30000,3000):
            y.append(tmp[0][j])  

    y=np.array(y)
    y=OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))
    x=np.split(np.array(x), len(x)//5, axis=0)
    y=np.split(np.array(y), len(y)//5, axis=0)
    # per_cat=np.sum(y,axis=0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                random_state=42)
   
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test) 
    
def cnn_model1(name,rate,x_shaped):
    input = Input(shape=x_shaped)
    my_cnn = Conv2D(filters=64, kernel_size=(50,1), strides=(8,1), padding="same", input_shape=x_shaped, activation="relu")(input)
    my_cnn = MaxPool2D(pool_size=(8,1), strides=(8,1), padding="same")(my_cnn)
    my_cnn = Dropout(rate=rate)(my_cnn)
    for _ in range(3):
        my_cnn = Conv2D(filters=40,kernel_size=(8,1),strides=(1,1), padding="same", activation="relu")(my_cnn)
    my_cnn = MaxPool2D (pool_size=(4,1), strides=(4,1), padding="same")(my_cnn)
    my_cnn = Flatten()(my_cnn)
    my_cnn = Model(inputs=input,outputs = my_cnn)

    return my_cnn
def cnn_model2(name,rate,x_shaped):
    input = Input(shape=x_shaped)
    my_cnn= TimeDistributed(input)
    my_cnn = Conv2D(filters=64, kernel_size=(400,1), strides=(50,1), padding="same", input_shape=x_shaped, activation="relu")(input)
    my_cnn = MaxPool2D(pool_size=(4,1), strides=(4,1), padding="same")(my_cnn)
    my_cnn = Dropout(rate=rate)(my_cnn)
    for _ in range(3):
        my_cnn = Conv2D(filters=40,kernel_size=(6,1),strides=(1,1), padding="same", activation="relu")(my_cnn)
    my_cnn = MaxPool2D (pool_size=(2,1), strides=(2,1), padding="same")(my_cnn)
    my_cnn = Flatten()(my_cnn)
    my_cnn = Model(inputs=input,outputs = my_cnn)
    return my_cnn


def build_merged_model(keep_proba,x_shaped,seq_len):
    input1 = Input(shape=(None,3,3000,1))
    input2 = Input(shape=(None,3,3000,1))
    model1 = cnn_model1('model1', keep_proba, x_shaped)
    model2 = cnn_model2('modle2', keep_proba, x_shaped)   
    model1_t = TimeDistributed(model1)(input1)
    model2_t= TimeDistributed(model2)(input2)
    merged_cnns =  concatenate([model1_t,model2_t])
    merged_cnns = BatchNormalization()(merged_cnns)
    merged_cnns = Dense(units=30,activation="relu")(merged_cnns)
    merged_cnns = Bidirectional(LSTM(units=32, return_sequences=True, activation="tanh", 
                recurrent_activation="sigmoid", dropout=keep_proba))(merged_cnns)
    merged_cnns = BatchNormalization()(merged_cnns)
    predictions = TimeDistributed(Dense(units=5, activation="softmax"))(merged_cnns)
    model=Model(inputs=[input1,input2], outputs=[predictions])

    return model



def w_categorical_crossentropy():
    def loss(y_true, y_pred):
        weights=np.array([0.18608124, 0.13560335, 0.43690263, 0.08632019, 0.15509259])
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1, keepdims=True)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (weights[[c_t,c_p]] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_pred, y_true) * final_mask
    return loss

def run_all():
    history = History()
    acc_tr = []
    loss_tr = []
    cm=[]
    ck_score=[]
    batch_size =2
    x_shaped=(3,3000,1)
    model=build_merged_model(keep_proba=0.5, x_shaped=x_shaped,seq_len=10)
    optimizer = keras.optimizers.Adam(lr=0.0001,clipnorm=1.0)
    train_loss = keras.losses.categorical_crossentropy
   
    model.compile(optimizer=optimizer, loss=train_loss, metrics=['accuracy'])
    print(model.summary())
    for fold in range(1):
        X_train, X_test, y_train, y_test  = get_data(fold)
        history=model.fit([X_train.reshape(-1,5,3,3000,1),X_train.reshape(-1,5,3,3000,1)],
                y_train, batch_size=batch_size, epochs=5, verbose=1, 
                callbacks=[history])#,validation_data=([X_test.reshape(-1,3,3000,1),X_test.reshape(-1,3,3000,1)],y_test),
          
        history_dict=history.history 
        json.dump(history_dict, open('/home/edith/Documents/EEG/history.json', 'w'))   
        acc_tr.append(history_dict['acc'])
        loss_tr.append(history_dict['loss'])
        y_hat=model.predict([X_test.reshape(-1,3,3000,1),X_test.reshape(-1,3,3000,1)], batch_size=batch_size, verbose=1)
        cm.append(confusion_matrix(y_test, y_hat, labels=range(5)))
        ck_score.append(cohen_kappa_score(y_test, y_hat))
                # callbacks=keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True) )
    
    ## add train data to eval
    # balance data
    # log training history
    # 
    with open('confusion.pkl', 'wb') as f:
        pickle.dump(cm, f)
    with open('ck_score.pkl', 'wb') as f:
        pickle.dump(ck_score, f)
    
    np.save('train_acc_res',np.array(acc_tr))
    np.save('train_loss_res',np.array(loss_tr))
    pass
run_all()