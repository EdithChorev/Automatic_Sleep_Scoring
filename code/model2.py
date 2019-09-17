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
from keras.models import load_model
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from itertools import product
import json
import pickle
import random


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, seq_len=10, num_channels=3,epi_samples=3000, n_classes=5, shuffle=True):
        'Initialization'
        self.seq_len = seq_len
        self.num_channels=num_channels
        self.epi_samples=epi_samples
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.seq_len, self.num_channels,self.epi_samples),dtype='float32')
        
        y = np.empty((self.batch_size,self.seq_len,self.n_classes), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(ID)

            # Store classnum_channenum_channelsnum_channelsnum_channelsls
            
            y[i] = np.load(ID[:-5]+'y.npy)
            
        return [X.reshape(-1,self.seq_len,self.num_channels,self.epi_samples,1),X.reshape(-1,self.seq_len,self.num_channels,self.epi_samples,1)], y
    
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
    input1 = Input(shape=(None,x_shaped[0],x_shaped[1],x_shaped[2]))
    input2 = Input(shape=(None,x_shaped[0],x_shaped[1],x_shaped[2]))
    model1 = cnn_model1('model1', keep_proba, x_shaped)
    model2 = cnn_model2('modle2', keep_proba, x_shaped)   
    model1_t = TimeDistributed(model1)(input1)
    model2_t= TimeDistributed(model2)(input2)
    merged_cnns =  concatenate([model1_t,model2_t])
    merged_cnns = BatchNormalization()(merged_cnns)
    merged_cnns = Dense(units=30,activation="relu")(merged_cnns)
    merged_cnns = Bidirectional(LSTM(units=32, return_sequences=True, activation="tanh", 
                recurrent_activation="sigmoid", dropout=keep_proba))(merged_cnns)
    merged_cnns = Bidirectional(LSTM(units=32, return_sequences=True, activation="tanh", 
                recurrent_activation="sigmoid", dropout=keep_proba))(merged_cnns)
    merged_cnns = BatchNormalization()(merged_cnns)
    predictions = TimeDistributed(Dense(units=5, activation="softmax"))(merged_cnns)
    model=Model(inputs=[input1,input2], outputs=[predictions])
    

    return model



def w_categorical_crossentropy(y_true, y_pred):
  
    weights=np.array([0.18608124, 0.13560335, 0.43690263, 0.08632019, 0.15509259])
    weights = K.variable(weights)
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

def validate_model(y_true,y_pred,num_cls):
    # loss = w_categorical_crossentropy(y_true, y_pred)
    y_true = y_true.argmax(axis=2).flatten()
    y_pred = y_pred.argmax(axis=2).flatten()

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
    loss_val = []
    acc_val = []
    f1_s =[]
    cm=[]
    ck_score=[]
    batch_size = 1
    num_epochs = 100
    num_channels = 3
    epi_samples =3000
    seq_len = 10
    num_cls = 5

    p_name='~/folds/'
    list_files = []
    for i in range(9):
        all_files = os.listdir(p_name+'fold'+str(i))
        for f in range(len(all_files)//2):
            list_files.append(p_name+'fold'+str(i)+'/'+str(f)+'x.npy')
    train_files = np.array(list_files)

    p_name='~/folds/'
    list_files = []

    all_files = os.listdir(p_name+'fold'+str(9))
    for f in range(len(all_files)//2):
        list_files.append(p_name+'fold'+str(i)+'/'+str(f)+'x.npy')
    val_files = np.array(list_files)

    val_files = val_files[np.random.permutation(len(val_files))]
    train_files = train_files[np.random.permutation(len(train_files))]


    training_generator = DataGenerator(list_IDs=train_files,  batch_size=batch_size,seq_len=seq_len,num_channels=num_channels,epi_samples=epi_samples, 
                 n_classes=num_cls, shuffle=True)
    val_generator = DataGenerator(list_IDs=val_files, batch_size=batch_size,seq_len=seq_len,num_channels=num_channels,epi_samples=epi_samples, 
                 n_classes=num_cls, shuffle=True)
    x_shaped=(num_channels,epi_samples,1)
    model=build_merged_model(keep_proba=0.5, x_shaped=x_shaped,seq_len=10)
    optimizer = keras.optimizers.Adam(lr=0.0001,clipnorm=1.0)
    train_loss = keras.losses.categorical_crossentropy
   
    model.compile(optimizer=optimizer, loss=train_loss, metrics=['accuracy'])
    print(model.summary())
    
    # val_fold = np.random.permutation(9)
    #for v_fold in val_fold: 
    history=model.fit_generator(generator=training_generator,
                    validation_data=val_generator, 
                    use_multiprocessing=True,epochs=num_epochs,verbose=1, 
                    callbacks=[history])
    # history=model.fit([X_train.reshape(-1,seq_len,num_channels,epi_samples,1),X_train.reshape(-1,seq_len,num_channels,epi_samples,1)],
    #         y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, 
    #         callbacks=[history])#, validation_data=([X_test.reshape(-1,seq_len,num_channels,epi_samples,1),X_test.reshape(-1,seq_len,num_channels,epi_samples,1)],y_test),validation_freq=10,
    
    history_dict=history.history 
    json.dump(history_dict, open('~/model2/history.json', 'w'))   
    acc_tr.append(history_dict['acc'])
    loss_tr.append(history_dict['loss'])
    acc_val.append(history_dict['val_acc'])
    loss_val.append(history_dict['val_loss'])
 
    model.save('model2.h5')
    # X_test,  y_test  = get_data(v_fold, seq_len)    
    # y_hat = model.predict([X_test.reshape(-1,seq_len,num_channels,epi_samples,1),X_test.reshape(-1,seq_len,num_channels,epi_samples,1)], batch_size=batch_size, verbose=1)
    #F1, F1_macro, ck_s, cmat, acc, acc_macro, TPR, TNR, PPV = validate_model(y_test,y_hat,num_cls) 

        # f1, ck, confm, acc = validate_model(y_test,y_hat)
        
    # cm.append(cmat)
    # f1_s.append(F1)
    # acc_val.append(acc)
    # ck_score.append(ck_s)
        
                    # callbacks=keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True) )
    
    np.save('~/model2/train_acc_res',np.array(acc_tr))
    np.save('~/model2/train_loss_res',np.array(loss_tr))
    np.save('~/model2/val_acc_res',np.array(acc_val))
    # np.save('val_loss_res',np.array(loss_val))
    np.save('~/model2/val_f1',np.array(f1_s))
    np.save('~/model2/val_cm',np.array(cm))
    np.save('~/model2/val_ck',np.array(ck_score))
    
    pass
run_all()