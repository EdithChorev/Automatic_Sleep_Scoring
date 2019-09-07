# adapted from akaraspt 
# https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py#L86


import numpy as np
import matplotlib as plt
import pandas as pd
import mne as mn
import mne.io
import math
import os
from datetime import datetime

EPOCH_SEC_SIZE = 30
#label values
W  = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict={
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REN": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
      0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}
mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

data_dir = "/home/edith//Documents/EEG/2018/"  
output_dir_fpz = "/home/edith//Documents/EEG/2018/FPZ/"
output_dir_pz = "/home/edith//Documents/EEG/2018/PZ/"
output_dir_eog = "/home/edith//Documents/EEG/2018/EOG/"


keyword = 'Hypnogram'
info= pd.DataFrame()
# generating file_list
file_names = []
for fn in os.listdir(data_dir):
    if ((fn.endswith('edf')) and ('PSG' in fn) and ('SC4' in fn)):
        file_names.append(fn)
# loading EDF/EDF+ files and extracting the raw signals and lables 
# updating info data frame with info about channels and files locations

for fn in file_names:
    data = mne.io.read_raw_edf(data_dir + fn, 
                                  montage = None, misc = None, stim_channel = None, \
                                  exclude = (), preload = False, verbose=None)
    # read raw data from file each channel seperatly for flexibility later on
    sampling_rate=data.info['sfreq']
    times = data[0][1] * 100
   
    raw_ch_fpz = pd.DataFrame(data[0][0].T,columns=['raw'])
    raw_ch_pz = pd.DataFrame(data[1][0].T,columns=['raw'])
    raw_ch_eog = pd.DataFrame(data[2][0].T,columns=['raw'])
    
    data.set_channel_types(mapping)
    data_orig_time = data.annotations.orig_time
    
    # find and read annotation file 
    for fname in os.listdir(data_dir):
        if (keyword in fname) and (fn[:6] in fname):
            annot = mn.read_annotations(data_dir + fname)
    # generate lables and remove indices
    remove_idx = []
    labels = []
    label_idx = []
    for i in range (len(annot)):
       onset = annot[i]["onset"]
       duration =  annot[i]["duration"]
       label = annot[i]["description"]
       label = ann2label[label]
       # detecting data which is wrongly or unlabed
       if label != UNKNOWN:
           if duration % EPOCH_SEC_SIZE != 0:
               raise Exception ("something wrong with epoch length")
           duration_epoch = int(duration / EPOCH_SEC_SIZE)
           label_epoch = np.ones(int(duration_epoch*sampling_rate*EPOCH_SEC_SIZE), dtype=np.int) * label
           labels.append(label_epoch)
           idx = int(onset * sampling_rate)+np.arange(duration * sampling_rate,dtype=np.int) 
           label_idx.append(idx)
       else:
           idx = int(onset * sampling_rate) + np.arange(duration * sampling_rate, dtype=np.int)
           remove_idx.append(idx)
    labels = np.hstack(labels)
    # taking care of unlabled or wrongly labled data
    if len(remove_idx) > 0:
        remove_idx = np.hstack(remove_idx)
        select_idx = np.setdiff1d(np.arange(len(raw_ch_fpz)), remove_idx)
    else:
        select_idx = np.arange(len(raw_ch_fpz))

    label_idx = np.hstack(label_idx)
    select_idx = np.intersect1d(select_idx, label_idx)    
    if len(label_idx) > len(select_idx):
        print ("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
        extra_idx = np.setdiff1d(label_idx, select_idx)
        # Trim the tail
        if np.all(extra_idx > select_idx[-1]):
            labels = labels[select_idx]
    print ("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
    
    # remove bad times from raw data
    raw_fpz = raw_ch_fpz.values[select_idx]
    raw_pz = raw_ch_pz.values[select_idx]
    raw_eog = raw_ch_eog.values[select_idx]

    # verify that we can still split to 30 s epochs
    if len(raw_fpz) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
        raise Exception ("something went wrong in cleaning")
    n_epochs = len(raw_fpz) / (EPOCH_SEC_SIZE * sampling_rate)

    y = labels.astype(np.int32)
    x = np.asarray(raw_fpz,dtype=np.float32)
    assert len(x) == len(y)

    #select sleep periods +- 1 h from first and last sleep stage
    w_edge_mins = 30
    nw_idx = np.where(y != stage_dict["W"])[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx <0: start_idx = 0
    if end_idx >= len(y): end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx+1)
    print("Data before selection: {}, {}".format(x.shape, y.shape))
    x = x[select_idx]
    y = y[select_idx]
    print("Data after selection: {}, {}".format(x.shape, y.shape))

    # Save
    filename=fn.replace('-PSG.edf','.npz')
    
    save_dict = {
        "x": x, 
        "y": y, 
        "fs": sampling_rate,
        "ch_label": 'fpz',
        
    }
    np.savez(output_dir_fpz+filename, **save_dict)

     
    x = np.asarray(raw_pz,dtype=np.float32)
    x = x[select_idx]
    save_dict = {
        "x": x, 
        "y": y, 
        "fs": sampling_rate,
        "ch_label": 'pz',
        
    }
    np.savez(output_dir_pz+filename, **save_dict)

    x = np.asarray(raw_eog,dtype=np.float32)
    x = x[select_idx]
    save_dict = {
        "x": x, 
        "y": y, 
        "fs": sampling_rate,
        "ch_label": 'eog',
        
    }
    np.savez(output_dir_eog+filename, **save_dict)
    print ("\n=======================================\n")


