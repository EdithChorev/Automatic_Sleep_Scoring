import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import  Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import Concatenate
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
    
def cnn_model(name,rate):
    x_shaped=(3,3000,1)
    my_cnn = Sequential(name=name)
    my_cnn.add(Conv2D(filters=64, kernel_size=(50,1), strides=(8,1), padding="same", input_shape=x_shaped, activation="relu"))
    my_cnn.add(MaxPool2D(pool_size=(8,1), strides=(8,1), padding="same"))
    my_cnn.add(Dropout(rate=rate))
    for _ in range(3):
        my_cnn.add(Conv2D(filters=128,kernel_size=(8,1),strides=(1,1), padding="same", activation="relu"))
    my_cnn.add(MaxPool2D (pool_size=(4,1), strides=(4,1), padding="same"))
    my_cnn.add(Flatten())
    my_cnn.add(Dense(units=16, activation="relu"))
    my_cnn.add(Dense(units=5, activation="softmax"))

    return my_cnn

# def cnn_part(x_shaped, keep_proba=0.5):
    
#     ### short filter cnn
#     x_shaped=(3,3000)
#     features = keras.layers.Input(x_shaped)
#     features_shape = tf.reshape(features,(-1,3,3000,1))
#     my_cnn = Conv2D ( filters=64, kernel_size=(50,1), 
#                     strides=(6,1), padding='same', activation ="relu")(features_shape)
#     my_cnn = MaxPool2D ( pool_size = (8,1), strides = (8,1), padding='same')(my_cnn)
#     my_cnn = Dropout( rate=1-keep_proba)(my_cnn)
#     for _ in range (3): 
#         my_cnn = Conv2D (filters=128, kernel_size=(8,1), 
#                     strides=(1,1), padding='same', activation ="relu")(my_cnn)
    
#     my_cnn = MaxPool2D ( pool_size = (4,1), strides = (4,1), padding='same')(my_cnn)
#     flat1= Flatten()(my_cnn)

#     ### long filter cnn
#     my_cnn = Conv2D ( filters=64, kernel_size=(400,1), 
#                     strides=(50,1), padding='same', activation ="relu")(features_shape)
#     my_cnn = MaxPool2D ( pool_size = (4,1), strides = (4,1), padding='same')(my_cnn)
#     my_cnn = Dropout(rate=keep_proba)(my_cnn) 
#     for _ in range (3): 
#         my_cnn = Conv2D (filters=128, kernel_size=(6,1), 
#                     strides=(1,1), padding='same', activation ="relu")(my_cnn)
    
#     my_cnn = MaxPool2D ( pool_size = (2,1), strides = (2,1), padding='same')(my_cnn)
#     flat2 = Flatten()(my_cnn)

#     ### concat output of cnns
#     my_cnn = Concatenate([flat1,flat2],axis=1)
#     my_cnn = Dense (units=256, activation = "relu")(my_cnn)
#     my_cnn = Dropout (rate=keep_proba)(my_cnn)
#     ### add fully connected layer
#     my_cnn = Dense (units=256, activation = "relu")(my_cnn)
    
#     classes = Dense ( units=5, activation = "softmax")(my_cnn)
#     model=keras.models.Model(inputs= features,outputs= classes)
#     return model

def run_all():
    # tf.enable_eager_execution()
    X_train, X_test, y_train, y_test  = get_data()
    # x_shaped = tf.placeholder(tf.float32,[3,3000])
    # output = tf.placeholder(tf.float32,[None,5])
    
    model = cnn_model ('my_cnn',0.5)
    batch_size = 16

    optimizer = keras.optimizers.Adam(lr=0.0001,clipnorm=1.0)
    train_loss = keras.losses.categorical_crossentropy
    model.compile(optimizer=optimizer, loss=train_loss)
    print(model.summary())
    model.fit(X_train.reshape(-1,3,3000,1),y_train,batch_size=batch_size, epochs=10,verbose=2)#,
                # callbacks=keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True) )
    
    pass

run_all()