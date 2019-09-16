import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,Model
from keras.layers import  Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import Concatenate, concatenate
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
    # per_cat=np.sum(y,axis=0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                random_state=42)
   
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test) 
    
def cnn_model1(name,rate,x_shaped):
    
    my_cnn = Sequential(name=name)
    my_cnn.add(Conv2D(filters=64, kernel_size=(50,1), strides=(8,1), padding="same", input_shape=x_shaped, activation="relu"))
    my_cnn.add(MaxPool2D(pool_size=(8,1), strides=(8,1), padding="same"))
    my_cnn.add(Dropout(rate=rate))
    for _ in range(3):
        my_cnn.add(Conv2D(filters=128,kernel_size=(8,1),strides=(1,1), padding="same", activation="relu"))
    my_cnn.add(MaxPool2D (pool_size=(4,1), strides=(4,1), padding="same"))
    my_cnn.add(Flatten())
    

    return my_cnn
def cnn_model2(name,rate,x_shaped):
    
    my_cnn = Sequential(name=name)
    my_cnn.add(Conv2D(filters=64, kernel_size=(400,1), strides=(50,1), padding="same", input_shape=x_shaped, activation="relu"))
    my_cnn.add(MaxPool2D(pool_size=(4,1), strides=(4,1), padding="same"))
    my_cnn.add(Dropout(rate=rate))
    for _ in range(3):
        my_cnn.add(Conv2D(filters=128,kernel_size=(6,1),strides=(1,1), padding="same", activation="relu"))
    my_cnn.add(MaxPool2D (pool_size=(2,1), strides=(2,1), padding="same"))
    my_cnn.add(Flatten())
    
    return my_cnn

def build_merged_model(keep_proba,x_shaped):
    
    model1 = cnn_model1('model1', keep_proba, x_shaped)
    model2 = cnn_model2('modle2', keep_proba, x_shaped)    
    
    merged_input = Concatenate()([model1.output,model2.output])
    merged = Dense(units=128, activation = "relu")(merged_input)
    merged = Dense(units=5, activation = "softmax")(merged)
    model  = Model(inputs=[model1.input,model2.input], outputs=merged)

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

def validate_model(y_true,y_pred,num_cls):
    # loss = w_categorical_crossentropy(y_true, y_pred)
    y_true = y_true.argmax(axis=1).flatten()
    y_pred = y_pred.argmax(axis=1).flatten()

    cm = confusion_matrix(y_true, y_pred, labels=range(num_cls))
    ck = cohen_kappa_score(y_true, y_pred,labels=range(num_cls))
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
     # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
     # Overall accuracy
    acc = (TP + TN) / (TP + FP + FN + TN)
    acc_macro = np.mean(acc)
    F1 = (2 * PPV * TPR) / (PPV + TPR)
    F1_macro = np.mean(F1)
    acc = accuracy_score(y_true, y_true)
    return  F1, F1_macro, ck, cm, acc, acc_macro, TPR, TNR, PPV

def run_all():
    history = History()
    acc_tr = []
    loss_tr = []
    f1_s = []
    cm=[]
    acc_val = []
    ck_score=[]
    batch_size =16
    x_shaped=(3,3000,1)
    model=build_merged_model(keep_proba=0.5, x_shaped=x_shaped)
    optimizer = keras.optimizers.Adam(lr=0.0001,clipnorm=1.0)
    train_loss = keras.losses.categorical_crossentropy
   
    model.compile(optimizer=optimizer, loss=train_loss, metrics=['accuracy'])
    print(model.summary())
    v_fold=9
    folds=np.random.permutation(10)
    for _ in range 100:
        for fold in range(10):
            if fold != v_fold:
                X_train, X_test, y_train, y_test  = get_data(fold)
                history=model.fit([X_train.reshape(-1,3,3000,1),X_train.reshape(-1,3,3000,1)],
                        y_train, batch_size=batch_size, epochs=1, verbose=1, 
                        callbacks=[history])#,validation_data=([X_test.reshape(-1,3,3000,1),X_test.reshape(-1,3,3000,1)],y_test),
                
                history_dict=history.history 
                json.dump(history_dict, open('/home/edith/Documents/EEG/history.json', 'w'))   
                acc_tr.append(history_dict['acc'])
                loss_tr.append(history_dict['loss'])
        folds=np.random.permutation(10)
    model.save('model.h5')
    y_hat=model.predict([X_test.reshape(-1,3,3000,1),X_test.reshape(-1,3,3000,1)], batch_size=batch_size, verbose=1)
    F1, F1_macro, ck_s, cmat, acc, acc_macro, TPR, TNR, PPV = validate_model(y_test,y_hat,5)         
            #cm.append(confusion_matrix(y_test, y_hat, labels=range(5)))
        #ck_score.append(cohen_kappa_score(y_test, y_hat))
                # callbacks=keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True) )
    cm.append(cmat)
    f1_s.append(F1)
    acc_val.append(acc)
    ck_score.append(ck_s)
    ## add train data to eval
    # balance data
    # log training history
    # 
    # with open('confusion.pkl', 'wb') as f:
    #     pickle.dump(cm, f)
    # with open('ck_score.pkl', 'wb') as f:
    #     pickle.dump(ck_score, f)
    
    np.save('train_acc_res',np.array(acc_tr))
    np.save('train_loss_res',np.array(loss_tr))
    np.save('val_acc_res',np.array(acc_val))
    # np.save('val_loss_res',np.array(loss_val))
    np.save('val_f1',np.array(f1_s))
    np.save('val_cm',np.array(cm))
    np.save('val_ck',np.array(ck_score))

    pass
run_all()