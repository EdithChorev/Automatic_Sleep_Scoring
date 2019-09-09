import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from keras.layers import  Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import Concatenate
from sklearn.model_selection import train_test_split

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
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    
    return X_train, X_test, y_train, y_test 

def cnn_part(x_shaped, keep_proba=0.5):
    
    ### short filter cnn
    x_shaped=(-1,3,3000,1)
    features = keras.Input(x_shaped)
    my_cnn = Conv2D (input_shape=features, filters=64, kernel_size=(50,1), 
                    strides=(6,1), padding='same', activation ="relu")
    my_cnn = MaxPool2D (inputs=my_cnn, pool_size = (8,1), strides = (8,1), padding='same')
    my_cnn = Dropout(my_cnn, keep_prob=keep_proba)
    for _ in range (3): 
        my_cnn = Conv2D (inputs=my_cnn, filters=128, kernel_size=(8,1), 
                    strides=(1,1), padding='same', activation ="relu")
    
    my_cnn = MaxPool2D (inputs=my_cnn, pool_size = (4,1), strides = (4,1), padding='same')
    flat1= Flatten(inputs=my_cnn)

    ### long filter cnn
    my_cnn = Conv2D (input_shape=features, filters=64, kernel_size=(400,1), 
                    strides=(50,1), padding='same', activation ="relu")
    my_cnn = MaxPool2D (inputs=my_cnn, pool_size = (4,1), strides = (4,1), padding='same')
    my_cnn = Dropout(my_cnn, keep_prob=keep_proba)
    for _ in range (3): 
        my_cnn = Conv2D (inputs=my_cnn, filters=128, kernel_size=(6,1), 
                    strides=(1,1), padding='same', activation ="relu")
    
    my_cnn = MaxPool2D (inputs=my_cnn, pool_size = (2,1), strides = (2,1), padding='same')
    flat2 = Flatten(inputs=my_cnn)

    ### concat output of cnns
    my_cnn = Concatenate(axis=0)([flat1,flat2])
    my_cnn = Dropout (my_cnn, keep_prob=keep_proba)
    ### add fully connected layer
    my_cnn = Dense (inputs=my_cnn, units=256, activation = "relu")
    
    classes = Dense (inputs=my_cnn, units=5, activation = "softmax")
    model=keras.Model(inputs=features, ouputs= classes)
    return model

def run_all():
    # tf.enable_eager_execution()
    X_train, X_test, y_train, y_test  = get_data()
    # x_shaped = tf.placeholder(tf.float32,[3,3000])
    # output = tf.placeholder(tf.float32,[None,5])
    x_shaped = (-1,1,3,300)
    model = cnn_part (x_shaped, keep_proba=0.5)
    batch_size = 16

    optimizer = keras.optimizers.Adam(lr=0.0001,clipnorm=1.0)
    train_loss = keras.losses.categorical_crossentropy
    model.compile(optimizer=optimizer, loss=train_loss)
    model.fit(X_train.reshape(-1,3,3000,1),y_train,batch_size=batch_size, epochs=100,verbose=2,
                callbacks=keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True) )
    # loss_history = []
    
    # for (batch,(features, targets)) in enumerate(trainset): #.take(1000)
    #     with tf.GradientTape() as tape:
    #         logits = model(features, training = True)
    #         loss_vlue = tf.losses.sparse_categorical_crossentropy(targets, logits)
    #     loss_history.append(loss_value.numpy())
    #     grads = tape.gradient(loss_value,model.trainable_variables)
    #     optimizer.apply_gradients(zip(grads,model.trainable_variables),
    #     global_step = tf.train.get_or_create_global_step())

    # plt.plot(loss_history)
    # plt.xlabel('Batch #')
    # plt.ylabel('Loss [entropy]')
    pass

run_all()