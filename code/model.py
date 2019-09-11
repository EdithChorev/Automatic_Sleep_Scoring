import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,Model
from keras.layers import  Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import Concatenate, concatenate
from keras.callbacks import History
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



def get_data(batchsize =16): ## cnn only part
    x = np.load('/home/edith/Documents/EEG/2018/folds/fold0_X.npy')
    y = np.load('/home/edith/Documents/EEG/2018/folds/fold0_Y.npy')
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   
    
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
    x_shaped=(3,3000,1)
    model1 = cnn_model1('model1', keep_proba, x_shaped)
    model2 = cnn_model2('modle2', keep_proba, x_shaped)    
    x_shaped=(3,3000,1)
    features = keras.layers.Input(x_shaped)
    features_shape = tf.reshape(features,(-1,3,3000,1))
    merged_input = Concatenate()([model1.output,model2.output])
    merged = Dense(units=128, activation = "relu")(merged_input)
    merged = Dense(units=5, activation = "softmax")(merged)
    model  = Model(inputs=[model1.input,model2.input], outputs=merged)

    return model




def run_all():
    history = History()
    acc_tr = []
    loss_tr = []
    

    X_train, X_test, y_train, y_test  = get_data()
   
    
    batch_size = 16
   
    model=build_merged_model(keep_proba=0.5)
    optimizer = keras.optimizers.Adam(lr=0.0001,clipnorm=1.0)
    train_loss = keras.losses.categorical_crossentropy
   
    model.compile(optimizer=optimizer, loss=train_loss, metrics=['accuracy'])
    print(model.summary())
    model.fit([X_train.reshape(-1,3,3000,1),X_train.reshape(-1,3,3000,1)],
                y_train, batch_size=batch_size, epochs=10, verbose=1,
                callbacks=[history])#,
               
    acc_tr.append(history.history['acc'])
	loss_tr.append(history.history['loss'])
                # callbacks=keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True) )
    
    ## add train data to eval
    # balance data
    # log training history
    # 
    pass

run_all()