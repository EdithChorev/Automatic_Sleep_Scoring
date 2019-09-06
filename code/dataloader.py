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
                labels.append(f["y"])
                sampling_rate = f["fs"]
        data=np.array(data).reshape(3,-1)
        labels=np.array(labels).reshape(3,-1)
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
            tmp_data, tmp_label, self.sampling_rate = self.load_npz_file(npz_f)
            if fs is None: 
                fs = self.sampling_rate
            elif fs != self.sampling_rate:
                raise Exception("Sampling rate mismatch.")
            
            # reshape data to fit conv 2d model + casting
            tmp_data = np.squeeze(tmp_data).astype(np.float32)
            tmp_label = np.array(tmp_label,dtype=np.int32)

            # normalize samples such that they have zero mean and unit variance
            tmp_data = (tmp_data - np.expand_dims(tmp_data.mean(axis = 1),axis = 1)) /np.expand_dims(tmp_data.std(axis = 1),axis = 1)

            data.append(tmp_data)
            labels.append( tmp_label)
            
        return data, labels


    # def load_test_data(self):
    
    def load_data(self,seq_len = 10, shuffle =True,n_files=None):
        # create list of all relevant file names this means all three channel sources 
        # stored under 2018 as 2018/..FPZ/..PZ/..EOG files from same rec have the same name
        self.data_dir=os.path.join(self.data_dir,'FPZ')
        all_files=os.listdir(self.data_dir)
        npzfiles =[] 
        for f in (all_files):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir,f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles [:n_files]
        
        # randomize files order once and save so fold remain constant through training 
        r_permute = np.random.permutation(len(npzfiles))
        filename ="r_permute,npz"
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
        # Load training and validation sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        data_train, label_train = self._load_npzlist_files(train_files)
        print (" ")
        print ("Load Test set:")
        data_test, label_test = self._load_npzlist_files(test_files)
        print (" ")

        print ("Training set: n_subjects={}".format(len(data_train)))
        n_train_examples = 0

        for d in data_train:
            print (d.shape)
            n_train_examples += d.shape[0]
        print ("Number of examples = {}".format(n_train_examples))
        self.print_n_samples_each_class(np.hstack(label_train),self.classes)
        print (" ")
        print ("Test set: n_subjects = {}".format(len(data_test)))
        n_test_examples = 0
        for d in data_test:
            print (d.shape)
            n_test_examples += d.shape[0]
        print ("Number of examples = {}".format(n_test_examples))
        self.print_n_samples_each_class(np.hstack(label_test),self.classes)
        print (" ")

        # take care of sequence generation here currently not working
        data_train = np.hstack(data_train)
        label_train = np.hstack(label_train)
        data_train = [data_train[i:i + seq_len] for i in range(0, len(data_train), seq_len)]
        label_train = [label_train[i:i + seq_len] for i in range(0, len(label_train), seq_len)]
        if data_train[-1].shape[0]!=seq_len:
            data_train.pop()
            label_train.pop()

        data_train = np.asarray(data_train)
        label_train = np.asarray(label_train)

        data_test = np.vstack(data_test)
        label_test = np.hstack(label_test)
        data_test = [data_test[i:i + seq_len] for i in range(0, len(data_test), seq_len)]
        label_test = [label_test[i:i + seq_len] for i in range(0, len(label_test), seq_len)]

        if data_test[-1].shape[0]!=seq_len:
            data_test.pop()
            label_test.pop()

        data_test = np.asarray(data_test)
        label_test = np.asarray(label_test)

        # shuffle
        if shuffle is True:
            
            permute = np.random.permutation(len(label_train))
            data_train = np.asarray(data_train)
            data_train = data_train[permute]
            label_train = label_train[permute]

            
            permute = np.random.permutation(len(label_test))
            data_test = np.asarray(data_test)
            data_test = data_test[permute]
            label_test = label_test[permute]

        return data_train, label_train, data_test, label_test
             
    @staticmethod
    def print_n_samples_each_class(labels,classes):
        class_dict = dict(zip(range(len(classes)),classes))
        unique_labels = np.unique(labels)
        for c in unique_labels:
            n_samples = len(np.where(labels == c)[0])
            print ("{}: {}".format(class_dict[c], n_samples))
        
classes = ['W', 'N1', 'N2', 'N3', 'REM']
data_loader = SeqDataLoader('/home/edith/Documents/EEG/2018/', 10, 1, classes=classes)
X_train, y_train, X_test, y_test = data_loader.load_data(seq_len=10, n_files=4)
# data_loader = SeqDataLoader('_load_npzlist_files(self, npz_files)/home/edith/Documents/EEG/2018', 10, 1, classes=classes)
# data_loader = SeqDataLoader('/home/edith/Documents/EEG/2018', 10, 1, classes=classes)
# data_loader = SeqDataLoader('/home/edith/Documents/EEG/2018', 10, 1, classes=classes)
#data,labels,sampling_rate=data_loader.load_npz_file ('/home/edith/Documents/EEG/2018/FPZ/SC4001E0.npz')   
#data,labels,sampling_rate=data_loader._load_npzlist_files (['/home/edith/Documents/EEG/2018/FPZ/SC4001E0.npz','/home/edith/Documents/EEG/2018/FPZ/SC4102E0.npz'])   
print(X_train.shape)