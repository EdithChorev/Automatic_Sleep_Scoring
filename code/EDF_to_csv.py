import numpy as np
import matplotlib as plt
import pandas as pd
import mne as mn
import mne.io
import os

# reading patient and experiment info
info=pd.read_excel('/home/edith/Documents/EEG/sleep-edf-database-expanded-1.0.0/sleep-casset_info.xls')
# adding some columns that will hold info about recording
info["EEG Fpz-Cz_min"] = None
info["EEG Fpz-Cz_max"] = None
info["EEG Pz-Oz_min"] = None
info["EEG Pz-Oz_max"] = None
info["EOG horizontal_min"] = None
info["EOG horizontal_max"] = None
info["Event marker_min"] = None
info["Event marker_max"]= None
info["PSG_data"] = None
info["label_data"] = None
# channel mapping
mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

keyword = 'Hypnogram'
data_dir = "/home/edith/Documents/EEG/sleep-edf-database-expanded-1.0.0/"   

# loading EDF/EDF+ files and extracting the raw signals and lables updaing info data frame with locations of data & labels
for f in range (info.shape[0]):
    data = mne.io.read_raw_edf(data_dir + info.loc[f,'PSG_file'], 
                                  montage = None, misc = None, stim_channel = 'auto', \
                                  exclude = (), preload = False, verbose=None)
    keystr = 'SC4' + info.loc[f,'PSG_file'][18:21]
    
    for fname in os.listdir(data_dir + 'sleep-cassette/'):
        if (keyword in fname) and (keystr in fname):
            annot = mn.read_annotations(data_dir + 'sleep-cassette/' + fname)
    
    data.set_annotations(annot, emit_warning = False)
    data.set_channel_types(mapping)
    
    orig_time = data.annotations.orig_time
    onset = data.annotations.onset
    label = data.annotations.description
    rec_label = pd.DataFrame({'label':label,'onset':onset})
    rec_label.insert(2, "orig_time", orig_time, True) 
    
    
    rec = data.to_data_frame()
#     print(rec.columns)
    rec = rec.drop(['Resp oro-nasal','EMG submental', 'Temp rectal'],axis=1)
    cols = rec.columns
    for i in range(4):
        info.loc[f,cols[i] + '_min'] = rec.loc[:,cols[i]].min()
        info.loc[f,cols[i] + '_max'] = rec.loc[:,cols[i]].max()
   
    
    
    keystr = info.loc[f,'PSG_file'][18:21]
    name='rec' + keystr + '.csv'
    info.loc[f,"PSG_data"]=name
    rec.to_csv(data_dir + 'data/' + name)
    name='rec_label' + keystr + '.csv'
    rec_label.to_csv(data_dir + 'data/' + name)
    info.loc[f,"label_data"]=name
    if info.loc[f,cols[2] + '_max']==None:
        print(f,info.loc[f,:])
        
info.to_csv(data_dir + 'data/rec_info.csv')    