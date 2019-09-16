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
            
            y[i] = np.load(ID[:43]+'y'+ID[44:])
            
        return [X.reshape(-1,self.seq_len,self.num_channels,self.epi_samples,1),X.reshape(-1,self.seq_len,self.num_channels,self.epi_samples,1)], y
        
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
    loss_val = []
    acc_val = []
    f1_s =[]
    cm=[]
    ck_score=[]
    batch_size = 4
    num_epochs = 100
    num_channels = 3
    epi_samples =3000
    seq_len = 1
    num_cls = 5

    p_name='/home/edith/Documents/EEG/2018/folds/'
    list_files = []
    for i in range(9):
        all_files = os.listdir(p_name+'fold'+str(i))
        for f in range(len(all_files)//2):
            list_files.append(p_name+'fold'+str(i)+'/x'+str(f)+'.npy')
    train_files = np.array(list_files)

    p_name='/home/edith/Documents/EEG/2018/folds/'
    list_files = []

    all_files = os.listdir(p_name+'fold'+str(9))
    for f in range(len(all_files)//2):
        list_files.append(p_name+'fold'+str(i)+'/x'+str(f)+'.npy')
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
    json.dump(history_dict, open('/home/edith/Documents/EEG/history.json', 'w'))   
    acc_tr.append(history_dict['acc'])
    loss_tr.append(history_dict['loss'])
    acc_val.append(history_dict['val_acc'])
    loss_val.append(history_dict['val_loss'])
    model.save('model1.h5')

    pass
run_all()