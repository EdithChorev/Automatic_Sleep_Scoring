import os
import numpy as np

class SeqDataLoader(object):
    
    def __init__(self, datadir, n_folds, fold_idx, classes):
        self.data_dir = datadir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.classes = classes
        
        
    
    def load_npz_file (self,npz_file):
        labels=[]
        data=[]
        # load data and labels of a single file
        for dir_ind in  (["FPZ","PZ","EOG"]):
            # change path to each data source path
            base,tf = os.path.split(npz_file)
            base,_ = os.path.split(base)
            tf = os.path.join(base, dir_ind, tf)
            
            with np.load(tf) as f:
                data.append(f["x"])
                labels=(f["y"])
                sampling_rate = f["fs"]
            assert sampling_rate == 100
            
        data=np.array(data).reshape(3,-1)
        
        print(data.shape)

        return data,labels,sampling_rate


    # def save_to_npz_file (self, data, labels , sampling_rate,filename):

    def _load_npzlist_files(self, npz_files):
        # load data and labels of all files in list
        data= []
        labels= []
        fs= None
        for npz_f in npz_files:
            print("loading {} ..." .format(npz_f))
            tmp_data, tmp_label, fs = self.load_npz_file(npz_f)
            if fs is None: 
                fs = 100
            
            
            # reshape data to fit conv 2d model + casting
            tmp_data = np.squeeze(tmp_data).astype(np.float32)
           

            # normalize samples such that they have zero mean and unit variance
            tmp_data = (tmp_data - np.expand_dims(tmp_data.mean(axis = 1),axis = 1)) /np.expand_dims(tmp_data.std(axis = 1),axis = 1)

            data.append(tmp_data)
            labels.append( tmp_label)
            
        return data, labels, fs
    def create_seq_data(self,data, labels, seq_len, sf ):
        sl= int(seq_len*sf*30)
        first=True
        seq_data=[]
        seq_labels=[]
        for d in data:
            for i in np.arange(0, d.shape[1]-sl, sl):
                if first:
                    seq_data = d[:, i:i+sl]
                    first = False
                else:
                    seq_data = np.append(seq_data,d[:, i:i+sl],axis=0)
                    print(i//sl, i )
        first=True
        for l in labels: 
            for i in np.arange(0, len(l)-sl, sl):
                if first:
                    seq_labels = l[i:i+sl]
                    first=False
                else:
                    seq_labels = np.append(seq_labels, l[i:i+sl])
                    print(i//sl, i )
                seq_labels = seq_labels.reshape(-1,sl)
        return seq_data, seq_labels


    def load_data(self,seq_len = 10, shuffle =True,n_files=None):
        # create list of all relevant file names this means all three channel sources 
        # stored under 2018 as 2018/..FPZ/..PZ/..EOG files from same rec have the same name
        # create train and validation sets and from data create 10 folds
        sampling_rate=100
        data_dir=os.path.join(self.data_dir,'FPZ')
        all_files=os.listdir(data_dir)
        npzfiles =[] 
        # out of 83 patients save 3 for final validation 
        # val_subjects=np.random.randint(0, 83, 3)
        # np.savez('val_subjects.npz', val_subjects  )
        # create file list for training (no validation subjects) 
        for f in (all_files):
            if (".npz" in f): #and ("SC4"+str(val_subjects[0]) not in f) and ("SC4"+str(val_subjects[1]) not in f) and ("SC4"+str(val_subjects[2]) not in f):
                npzfiles.append(os.path.join(self.data_dir,'FPZ',f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles [:n_files]
        
        # randomize files order once and save so fold remain constant through training
        # split to train and test 
        r_permute = np.random.permutation(len(npzfiles))
        filename ="r_permute.npz"
        if (os.path.isfile(filename)):
            with np.load(filename) as f:
                r_permute = f["inds"]
        else:
            save_dict={"inds":r_permute,
            }
            np.savez(filename, **save_dict)

        npzfiles = np.asarray(npzfiles)[r_permute]
        train_files = np.array_split(npzfiles, self.n_folds)
        test_files = train_files[self.fold_idx]
        train_files = list(set(npzfiles) - set(test_files))
        # Load training and test sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        #data_train, label_train,sampling_rate = self._load_npzlist_files(train_files)
        print (" ")
        print ("Load Test set:")
        data_test, label_test,sampling_rate = self._load_npzlist_files(test_files)
        print (" ")

        #print ("Training set: n_subjects={}".format(len(data_train)))
        n_train_examples = 0
        
        # for d in data_train:
        #     print (d.shape)
        #     n_train_examples += d.shape[0]
        # print ("Number of examples = {}".format(n_train_examples))
        # self.print_n_samples_each_class(np.hstack(label_train),self.classes)
        print (" ")
        print ("Test set: n_subjects = {}".format(len(data_test)))
        n_test_examples = 0
        for d in data_test:
            print (d.shape)
            n_test_examples += d.shape[0]
        print ("Number of examples = {}".format(n_test_examples))
        self.print_n_samples_each_class(np.hstack(label_test),self.classes)
        print (" ")
        seq_data_train, seq_label_train =(0,0)
        # take care of sequence generation here currently not working
        #seq_data_train, seq_label_train = self.create_seq_data(data_train, label_train, seq_len, sampling_rate )
        seq_data_test, seq_label_test = self.create_seq_data(data_test, label_test, seq_len, sampling_rate )          
        # shuffle
        # if shuffle is True:
        #     sdt=[]
        #     slt=[]
        #     permute = np.random.permutation(seq_data_train.shape[0]//3)
        #     first=True
        #     for i in permute:
        #         if first:
        #             sdt = seq_data_train[i*3:i*3+3,:]
        #             slt = np.array(seq_label_train[i*3:i*3+3])
        #             first = False
        #         else:
        #             sdt = np.append (sdt, seq_data_train[i*3:i*3+3,:])
        #             slt = np.append (slt, seq_label_train[i*3:i*3+3])
        #     seq_data_train = sdt
        #     seq_label_train = slt

        return seq_data_train, seq_label_train, seq_data_test, seq_label_test

            
            
        
             
    @staticmethod
    def print_n_samples_each_class(labels,classes):
        class_dict = dict(zip(range(len(classes)),classes))
        unique_labels = np.unique(labels)
        for c in unique_labels:
            n_samples = len(np.where(labels == c)[0])
            print ("{}: {}".format(class_dict[c], n_samples))
        
classes = ['W', 'N1', 'N2', 'N3', 'REM']
data_loader = SeqDataLoader('/home/edith/Documents/EEG/2018/', 10, 1, classes=classes)
X_train, y_train, X_test, y_test = data_loader.load_data(seq_len=10, n_files=None)

np.save("fold1_X",X_train)
# data_loader = SeqDataLoader('_load_npzlist_files(self, npz_files)/home/edith/Documents/EEG/2018', 10, 1, classes=classes)
# data_loader = SeqDataLoader('/home/edith/Documents/EEG/2018', 10, 1, classes=classes)
# data_loader = SeqDataLoader('/home/edith/Documents/EEG/2018', 10, 1, classes=classes)
#data,labels,sampling_rate=data_loader.load_npz_file ('/home/edith/Documents/EEG/2018/FPZ/SC4001E0.npz')   
#data,labels,sampling_rate=data_loader._load_npzlist_files (['/home/edith/Documents/EEG/2018/FPZ/SC4001E0.npz','/home/edith/Documents/EEG/2018/FPZ/SC4102E0.npz'])   
print(X_test.shape)