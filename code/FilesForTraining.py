import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np
seq_length=10

for fold in range(10):
    p_name='/home/ubuntu/folds/'
    name='fold' + str(fold)
    os.mkdir(p_name+name)
    x = np.load(p_name+name+'_X.npy')
    y = np.load(p_name+name+'_Y.npy')
    xx=np.array(np.split(x,x.shape[0]//3,axis = 0))
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
    x=np.split(np.array(x), len(x)//seq_length, axis=0)
    y=np.split(np.array(y), len(y)//seq_length, axis=0)
    p_name=p_name + name +'/'
    print(p_name)
    for ind in range (len(y)):
        np.save(p_name+str(ind)+'x'+'.npy', x[ind])
        np.save(p_name+str(ind)+'y'+'.npy', y[ind])
                       
