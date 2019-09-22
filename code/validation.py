import keras
import numpy as np
from keras.models import load_model
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import tensorflow as tf


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
            
            y[i] = np.load(ID[:-5]+'y.npy')
            
        return [X.reshape(-1,self.seq_len,self.num_channels,self.epi_samples,1),X.reshape(-1,self.seq_len,self.num_channels,self.epi_samples,1)], y

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




list_files = []
batch_size = 16
num_epochs = 50
num_channels = 3
epi_samples =3000
seq_len = 10
num_cls = 5

all_files = os.listdir('/home/ubuntu/folds/'+'fold'+str(9))
for f in range(len(all_files)//2):
    list_files.append('/home/ubuntu/folds/'+'fold'+str(9)+'/'+str(f)+'x.npy')
val_files = np.array(list_files)

val_files = val_files[np.random.permutation(len(val_files))]

val_generator = DataGenerator(list_IDs=val_files, batch_size=batch_size,seq_len=seq_len,num_channels=num_channels,epi_samples=epi_samples, 
                n_classes=num_cls, shuffle=True)
optimizer = keras.optimizers.Adam(lr=0.0001,clipnorm=1.0)
train_loss = keras.losses.categorical_crossentropy

model = load_model("/home/ubuntu/model2/model2.h5")
y_hat = model.predict_generator(val_generator, verbose=1 )
labels = (val_generator.class_indices)
F1, F1_macro, ck_s, cmat, acc, acc_macro, TPR, TNR, PPV = validate_model(labels,y_hat,num_cls) 
print (F1, F1_macro, ck_s, cmat, acc, acc_macro, TPR, TNR, PPV)

np.save('/home/ubuntu/model2/F1',np.array(F1))
np.save('/home/ubuntu/model2/F1_macro',np.array(F1_macro))
np.save('/home/ubuntu/model2/ck',np.array(ck_s))
np.save('/home/ubuntu/model2/cmat',np.array(cmat))
np.save('/home/ubuntu/model2/acc',np.array(acc))
np.save('/home/ubuntu/model2/acc_macro',np.array(acc_macro))
np.save('/home/ubuntu/model2/TPR',np.array(TPR))
np.save('/home/ubuntu/model2/TNR',np.array(TNR))
np.save('/home/ubuntu/model2/PPV',np.array(PPV))

 #for v_fold in val_fold: 

