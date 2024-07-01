import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch. nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import BatchSampler
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
import math
import copy

def get_index(index_lists, output):
    if len(index_lists)>0:
        output.append(index_lists.pop(0))
        return False

    else:
        return True
class SuperSampler(torch.utils.data.sampler.Sampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset,  batch_size, status = None):
        self.status = status
        self.batch_size = batch_size

        loader = DataLoader(dataset)
        labels_list = []
        datasets_list = []
        ids_ = []
        for images, labels, actual_labels, datasets,ids in loader:
            labels_list.append(actual_labels)

            datasets_list.append(datasets[0])
            ids_.append(ids)

        self.num_batches = math.ceil( len(labels_list)/batch_size)

        labels_list=np.array(labels_list)
        datasets_list=np.array(datasets_list)
        ids_=np.array(ids_)

        #ucsf
        temp = copy.deepcopy(labels_list)
        temp[datasets_list!='ucsf'] = -99
        ucsf_labels_list=temp

        self.ucsf_label_to_indices = {label: np.where(ucsf_labels_list == label)[0]
                                 for label in [0,1,2,3]}


        #lab
        temp = copy.deepcopy(labels_list)
        temp[datasets_list!='lab'] = -99
        lab_labels_list=temp
        self.lab_label_to_indices = {label: np.where(lab_labels_list == label)[0]
                                 for label in [0,2]}
        #adni
        temp = copy.deepcopy(labels_list)
        temp[datasets_list!='adni'] = -99
        adni_labels_list=temp
        self.adni_label_to_indices = {label: np.where(adni_labels_list == label)[0]
                                 for label in [0,1]}

        # ucsf

        self.all_ucsf_index = []
        for l in [0,1,2,3]:
            self.all_ucsf_index.extend(self.ucsf_label_to_indices[l])
        # lab
        self.all_lab_index = []
        for l in [0,2]:
            self.all_lab_index.extend(self.lab_label_to_indices[l])

        # adni
        self.all_adni_index = []
        for l in [0,1]:
            self.all_adni_index.extend(self.adni_label_to_indices[l])

    def __iter__(self):
        ucsf_0 = list(np.copy(self.ucsf_label_to_indices[0]))
        ucsf_1 = list(np.copy(self.ucsf_label_to_indices[1]))
        ucsf_2 = list(np.copy(self.ucsf_label_to_indices[2]))
        ucsf_3 = list(np.copy(self.ucsf_label_to_indices[3]))
        lab_0 = list(np.copy(self.lab_label_to_indices[0]))
        lab_2 = list(np.copy(self.lab_label_to_indices[2]))
        adni_0 = list(np.copy(self.adni_label_to_indices[0]))
        adni_1 = list(np.copy(self.adni_label_to_indices[1]))

        all_index = {}
        all_index[0] = ucsf_0
        all_index[1] = ucsf_1
        all_index[2] = ucsf_2
        all_index[3] = ucsf_3
        all_index[4] = lab_0
        all_index[5] = lab_2
        all_index[6] = adni_0
        all_index[7] = adni_1
        for i in range(0,8):
            np.random.shuffle(all_index[i])

        indices = []

        output = {}
        for i in range(0,self.num_batches):
            output[i] = []
        reverse = False
        current_p = 0

        old_i = 0
        old_count = 0
        for i in range(0,8):
            selected_list = all_index[i]
            while len(selected_list)>0:
                if reverse == False:
                    for j in range(current_p,self.num_batches):
                        none_left = get_index(selected_list,output[current_p])

                        if none_left:
                            break
                        else:
                            if current_p == self.num_batches-1:
                                current_p = 0
                            else:
                                current_p+=1


        for i in range(0,self.num_batches):
            indices.extend(output[i])
            print(i,len(output[i]))

        return iter(indices)
    def __len__(self):
        return 1

class UCSFSampler(torch.utils.data.sampler.Sampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples (only UCSF dataset)
    """

    def __init__(self, dataset,  batch_size, num_batches = 27,status = None):
        self.status = status
        self.batch_size = batch_size
        loader = DataLoader(dataset)
        labels_list = []
        datasets_list = []
        for images, labels, actual_labels, ids,ages,genders,npzs in loader:
            labels_list.append(actual_labels)

        self.num_batches = num_batches #math.ceil( len(labels_list)/batch_size)
        labels_list=np.array(labels_list)
        datasets_list=np.array(datasets_list)
        #ucsf
        temp = np.copy(labels_list)
        ucsf_labels_list=temp
        #self.ucsf_label_to_indices = {label: np.where(ucsf_labels_list == label)[0]
        #                         for label in [0,1,2]}
        self.ucsf_label_to_indices = {label: np.where(ucsf_labels_list == label)[0]
                                 for label in [0,1,2,3]}
        self.all_ucsf_index = []
        for l in [0,1,2,3]:
        #for l in [0,1,2]:
            self.all_ucsf_index.extend(self.ucsf_label_to_indices[l])


    def __iter__(self):
        ucsf_0 = list(np.copy(self.ucsf_label_to_indices[0]))
        ucsf_1 = list(np.copy(self.ucsf_label_to_indices[1]))
        ucsf_2 = list(np.copy(self.ucsf_label_to_indices[2]))
        ucsf_3 = list(np.copy(self.ucsf_label_to_indices[3]))

        all_index = {}
        all_index[0] = ucsf_0
        all_index[1] = ucsf_1
        all_index[2] = ucsf_2
        all_index[3] = ucsf_3

        for i in range(4):
        #for i in range(3):
            np.random.shuffle(all_index[i])
        all_index_copy = copy.deepcopy(all_index)
        indices = []
        output = {}
        for i in range(0,self.num_batches):
            output[i]=[]
            self.get_index(all_index, output[i],all_index_copy)

        for i in range(0,self.num_batches):
            indices.extend(output[i])

        return iter(indices)

    def __len__(self):
        return 1

    def get_index(self, all_index, output, all_index_copy):

        class_size = 10
        for i in range(class_size):
            for j in range(4):
            #for j in range(3):
                if len(all_index[j])>0:
                    output.append(all_index[j].pop(0))
                else:
                    all_index[j] = list(np.copy(all_index_copy[j]))
                    np.random.shuffle(all_index[j])

                    output.append(all_index[j].pop(0))

class MixedSampler(torch.utils.data.sampler.Sampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset,  batch_size, num_batches = 27,status = None):
        self.status = status
        self.batch_size = batch_size
        self.dataset = ['ucsf', 'adni', 'lab']#,'adni']
        loader = DataLoader(dataset)
        labels_list = []
        datasets_list = []
        for images, labels, actual_labels, datasets,ids,ages,genders in loader:
            labels_list.append(actual_labels)

            datasets_list.append(datasets[0])

        self.num_batches = num_batches #math.ceil( len(labels_list)/batch_size)

        labels_list=np.array(labels_list)
        datasets_list=np.array(datasets_list)
        #ucsf
        temp = np.copy(labels_list)
#         print(temp)
#         print(datasets_list)
        temp[datasets_list!='ucsf'] = -99
        ucsf_labels_list=temp
        self.ucsf_label_to_indices = {label: np.where(ucsf_labels_list == label)[0]
                                 for label in [0,1,2,3]}
        #lab
        temp = np.copy(labels_list)
        temp[datasets_list!='lab'] = -99
        lab_labels_list=temp
        self.lab_label_to_indices = {label: np.where(lab_labels_list == label)[0]
                                 for label in [0,2]}

        #adni
        temp = copy.deepcopy(labels_list)
        temp[datasets_list!='adni'] = -99
        adni_labels_list=temp
        self.adni_label_to_indices = {label: np.where(adni_labels_list == label)[0]
                                 for label in [0,1]}

#         ucsf
        self.all_ucsf_index = []
        for l in [0,1,2,3]:
            self.all_ucsf_index.extend(self.ucsf_label_to_indices[l])
        # lab
        self.all_lab_index = []
        for l in [0,2]:
            self.all_lab_index.extend(self.lab_label_to_indices[l])

        # adni
        self.all_adni_index = []
        for l in [0,1]:
            self.all_adni_index.extend(self.adni_label_to_indices[l])


    def __iter__(self):
        ucsf_0 = list(np.copy(self.ucsf_label_to_indices[0]))
        ucsf_1 = list(np.copy(self.ucsf_label_to_indices[1]))
        ucsf_2 = list(np.copy(self.ucsf_label_to_indices[2]))
        ucsf_3 = list(np.copy(self.ucsf_label_to_indices[3]))
        lab_0 = list(np.copy(self.lab_label_to_indices[0]))
        lab_2 = list(np.copy(self.lab_label_to_indices[2]))
        adni_0 = list(np.copy(self.adni_label_to_indices[0]))
        adni_1 = list(np.copy(self.adni_label_to_indices[1]))

        all_index = {}
        all_index[0] = ucsf_0
        all_index[1] = ucsf_1
        all_index[2] = ucsf_2
        all_index[3] = ucsf_3
        all_index[4] = lab_0
        all_index[5] = lab_2
        all_index[6] = adni_0
        all_index[7] = adni_1
        for i in range(0,8):#(0,6):
            np.random.shuffle(all_index[i])
        all_index_copy = copy.deepcopy(all_index)
        indices = []
        output = {}
        for i in range(0,self.num_batches):
            output[i]=[]
            self.get_index(all_index, output[i],all_index_copy)

        for i in range(0,self.num_batches):
            indices.extend(output[i])


        return iter(indices)

    def __len__(self):
        return 1


    def get_index(self, all_index, output, all_index_copy):

        class_size = 10
        for i in range(0,class_size):

            for j in range(0,8):
#                 if (i>=10) and(j ==0 or j == 2):
#                     if j == 0:
#                         index = 4
#                     else:
#                         index = 5

#                     if len(all_index[index])>0:
#                         output.append(all_index[index].pop(0))
#                     else:
#                         all_index[index] = list(np.copy(all_index_copy[index]))
#                         np.random.shuffle(all_index[index])

#                         output.append(all_index[index].pop(0))

#                 else:
                if len(all_index[j])>0:
                    output.append(all_index[j].pop(0))
                else:
                    all_index[j] = list(np.copy(all_index_copy[j]))
                    np.random.shuffle(all_index[j])

                    output.append(all_index[j].pop(0))

class PairedSampler(torch.utils.data.sampler.Sampler):
    """
    From Sampler docs
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.

    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples

    Only return UCSF data
    """

    def __init__(self, dataset,  batch_size, num_batches = 27,status = None):
        self.status = status
        self.batch_size = batch_size
        loader = DataLoader(dataset)
        labels_list = []
        #only retrieve sample labels from UCSF dataset
        for images, labels, actual_labels,ids,ages,genders,npzs in loader:
            labels_list.append(actual_labels)
        self.num_batches = num_batches #math.ceil( len(labels_list)/batch_size)

        ucsf_labels_list=np.array(labels_list)
        #dict=label:indices where labels occurs
        self.ucsf_label_to_indices = {label: np.where(ucsf_labels_list == label)[0]
                                 for label in [0,1,2,3]}

        #just a list of all indices
        self.all_ucsf_index = []
        for l in [0,1,2,3]:
            self.all_ucsf_index.extend(self.ucsf_label_to_indices[l])


    def __iter__(self):

        #retrieve indices with each label
        ucsf_0 = list(np.copy(self.ucsf_label_to_indices[0]))
        ucsf_1 = list(np.copy(self.ucsf_label_to_indices[1]))
        ucsf_2 = list(np.copy(self.ucsf_label_to_indices[2]))
        ucsf_3 = list(np.copy(self.ucsf_label_to_indices[3]))

        #dict=label:indices where labels occurs/same as ucsf_label_to_indices
        all_index = {}
        all_index[0] = ucsf_0
        all_index[1] = ucsf_1
        all_index[2] = ucsf_2
        all_index[3] = ucsf_3
        #shuffle indices for each label
        for i in range(0,4):
            np.random.shuffle(all_index[i])
        all_index_copy = copy.deepcopy(all_index)
        indices = []
        output = {}
        #make each batch (a list of indices drawn with replacement per label)
        for i in range(0,self.num_batches):
            output[i]=[]
            self.get_index(all_index, output[i],all_index_copy)
        #'flatten' batch dictionary into list
        for i in range(0,self.num_batches):
            indices.extend(output[i])
    
        #return iterator yielding pairs (0,0) --> (1,1) --> (2,2) --> (3,3) --> (0,0) --> ...
        return iter(indices)

    def __len__(self):
        return 2 #because pair


    def get_index(self, all_index, output, all_index_copy):

        class_size = 10 #this is a magic number and I'm not sure how it was determined :(
        for i in range(0,class_size):

            #for each label
            for j in range(0,4):
                #while more than one index is yet unassigned,
                #remove the first two elements remaining and add to to the batch
                if len(all_index[j])>1:
                    pair = (all_index[j].pop(0), all_index[j].pop(0))
                    output.append(pair)
                #otherwise, 'refresh' list and shuffle again
                else:
                    all_index[j] = list(np.copy(all_index_copy[j]))
                    np.random.shuffle(all_index[j])

                    pair = (all_index[j].pop(0), all_index[j].pop(0))
                    output.append(pair)

class PairedSamplerNPZ(torch.utils.data.sampler.Sampler):
    """
    From Sampler docs
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.

    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples

    PairedSamplerNPZ -- __iter__ returns paired samples such that (1) both samples have the same label
    (2) both samples either have or do not have a known NPZ score (3) sampling is weighted to provide 
    an equal number of each label 
    """

    def __init__(self, dataset,  batch_size, num_batches = 27, status = None):
        self.status = status
        self.batch_size = batch_size
        loader = DataLoader(dataset)
        npz_labels_list = []
        missing_labels_list = []

        #only retrieve sample labels containing NPZ scores from UCSF dataset
        for (i, (images, labels, actual_labels,ids,ages,genders,npzs)) in enumerate(loader):
            if npzs != 99.0 and not math.isnan(npzs):
                npz_labels_list.append((i,int(actual_labels)))
            else:
                missing_labels_list.append((i,int(actual_labels)))

        self.num_batches = num_batches #math.ceil( len(labels_list)/batch_size)

        npz_labels_list=np.array(npz_labels_list)
        missing_labels_list=np.array(missing_labels_list)
        
        #dict=label:indices where labels occurs
    
        self.npz_label_to_indices = {0:[], 1: [], 2: [], 3:[]}
        #self.npz_label_to_indices = {0:[], 1: [], 2: []}
        for (i,actual_labels) in npz_labels_list:
            self.npz_label_to_indices[actual_labels].append(i)

        self.missing_label_to_indices = {0:[], 1: [], 2: [], 3:[]}
        #self.missing_label_to_indices = {0:[], 1: [], 2: []}
        for (i,actual_labels) in missing_labels_list:
            self.missing_label_to_indices[actual_labels].append(i)
       

        #just a list of all indices
        self.all_npz_index = []
        for l in [0,1,2,3]:
        #for l in [0,1,2]:
            self.all_npz_index.extend(self.npz_label_to_indices[l])
        
        self.all_missing_index = []
        for l in [0,1,2,3]:
        #for l in [0,1,2]:
            self.all_missing_index.extend(self.missing_label_to_indices[l])



    def __iter__(self):

        #retrieve indices with each label
        npz_0 = list(np.copy(self.npz_label_to_indices[0]))
        npz_1 = list(np.copy(self.npz_label_to_indices[1]))
        npz_2 = list(np.copy(self.npz_label_to_indices[2]))
        npz_3 = list(np.copy(self.npz_label_to_indices[3]))

        #dict=label:indices where labels occurs/same as ucsf_label_to_indices
        npz_index = {}
        npz_index[0] = npz_0
        npz_index[1] = npz_1
        npz_index[2] = npz_2
        npz_index[3] = npz_3

        #shuffle indices for each label
        
        for i in range(0,4):
        #for i in range(0,3):
            np.random.shuffle(npz_index[i])
        npz_index_copy = copy.deepcopy(npz_index)
        
        #retrieve indices with each label
        missing_0 = list(np.copy(self.missing_label_to_indices[0]))
        missing_1 = list(np.copy(self.missing_label_to_indices[1]))
        missing_2 = list(np.copy(self.missing_label_to_indices[2]))
        missing_3 = list(np.copy(self.missing_label_to_indices[3]))

        #dict=label:indices where labels occurs/same as ucsf_label_to_indices
        missing_index = {}
        missing_index[0] = missing_0
        missing_index[1] = missing_1
        missing_index[2] = missing_2
        missing_index[3] = missing_3

        #shuffle indices for each label
        for i in range(0,4):
        #for i in range(0,3):
            np.random.shuffle(missing_index[i])
            #all_index[i] = np.concatenate((npz_index[i], missing_index[i]), axis=None)

        
        missing_index_copy = copy.deepcopy(missing_index)
        # print({idx: len(vals) for idx,vals in npz_index.items()}, {idx: len(vals) for idx,vals in missing_index.items()})

        indices = []
        output = {}

        #make each batch (a list of indices drawn with replacement per label)
        for i in range(0,self.num_batches):
            output[i]=[]
            self.get_index(npz_index, missing_index, output[i], npz_index_copy, missing_index_copy)
  
        #'flatten' batch dictionary into list
        for i in range(0,self.num_batches):
            indices.extend(output[i])


        #return iterator yielding pairs (0,0) --> (1,1) --> (2,2) --> (0,0) --> ...
        return iter(indices)

    def __len__(self):
        return 2 #because pair


    def get_index(self, npz_index, missing_index, output, npz_index_copy, missing_index_copy):

        #class_size=5
        class_size = 10 #this is a magic number and I'm not sure how it was determined :(
        for i in range(0,class_size):

            #for each label
            #for j in range(0,3):
            for j in range(0,4):
                
                pair = None
                #while more than one index is yet unassigned,
                #remove the first two elements remaining and add to to the batch
                if len(npz_index[j])>1:
                    pair = (npz_index[j].pop(0), npz_index[j].pop(0))
                elif len(missing_index[j])>1:
                    pair = (missing_index[j].pop(0), missing_index[j].pop(0))
                #otherwise, 'refresh' list and shuffle again
                else:
                    npz_index[j] = list(np.copy(npz_index_copy[j]))
                    np.random.shuffle(npz_index[j])

                    missing_index[j] = list(np.copy(missing_index_copy[j]))
                    np.random.shuffle(missing_index[j])

                    if len(npz_index[j])>1:
                        pair = (npz_index[j].pop(0), npz_index[j].pop(0))
                    elif len(missing_index[j])>1:
                        pair = (missing_index[j].pop(0), missing_index[j].pop(0))
                output.append(pair)

class PairedSamplerOnlyNPZ(torch.utils.data.sampler.Sampler):
    """
    From Sampler docs
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.

    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples

    PairedSamplerNPZ -- __iter__ returns paired samples such that both samples have a known NPZ score
    """

    def __init__(self, dataset):
        loader = DataLoader(dataset)
        self.npz_index_list = []
        #only retrieve sample labels containing NPZ scores from UCSF dataset
        for (i, (images, labels, actual_labels,ids,ages,genders,npzs)) in enumerate(loader):
            if npzs != 99 and not math.isnan(npzs):
                self.npz_index_list.append(i)
        self.batch_size = len(self.npz_index_list)//2
    def __iter__(self):

        indices = []
        self.get_index(indices)
        print(indices) 
        #return iterator yielding pairs 
        return iter(indices)

    def __len__(self):
        return 2 #because pair

    def get_index(self,indices):
        #while there are samples left
        while len(self.npz_index_list) >= 2:
            pair = (self.npz_index_list.pop(0), self.npz_index_list.pop(0))
            indices.append(pair)

class NoOverSampler(torch.utils.data.sampler.Sampler):
    """
    From Sampler docs
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.

    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples

    Only return UCSF data
    """

    def __init__(self, dataset,  batch_size, num_batches = 27, status = None):
        self.status = status
        self.batch_size = batch_size
        loader = DataLoader(dataset)
        npz_labels_list = []
        missing_labels_list = []

        #only retrieve sample labels containing NPZ scores from UCSF dataset
        for (i, (images, labels, actual_labels,ids,ages,genders,npzs)) in enumerate(loader):
            if npzs != 99.0 and not math.isnan(npzs):
                npz_labels_list.append((i,int(actual_labels)))
            else:
                missing_labels_list.append((i,int(actual_labels)))

        self.num_batches = math.ceil(len(npz_labels_list+missing_labels_list)/batch_size)

        npz_labels_list = np.array(npz_labels_list)
        missing_labels_list = np.array(missing_labels_list)
        
        #dict=label:indices where labels occurs
        self.npz_label_to_indices = {0:[], 1: [], 2: [], 3:[]}
        #self.npz_label_to_indices = {0:[], 1: [], 2: []}
        for (i,actual_labels) in npz_labels_list:
            self.npz_label_to_indices[actual_labels].append(i)
        self.missing_label_to_indices = {0:[], 1: [], 2: [], 3:[]}
        #self.missing_label_to_indices = {0:[], 1: [], 2: []}
        for (i,actual_labels) in missing_labels_list:
            self.missing_label_to_indices[actual_labels].append(i)
       

        #just a list of all indices
        self.all_npz_index = []
        for l in [0,1,2,3]:
        #for l in [0,1,2]:
            self.all_npz_index.extend(self.npz_label_to_indices[l])
        
        self.all_missing_index = []
        for l in [0,1,2,3]:
        #for l in [0,1,2]:
            self.all_missing_index.extend(self.missing_label_to_indices[l])



    def __iter__(self):

        #retrieve indices with each label
        npz_0 = list(np.copy(self.npz_label_to_indices[0]))
        npz_1 = list(np.copy(self.npz_label_to_indices[1]))
        npz_2 = list(np.copy(self.npz_label_to_indices[2]))
        npz_3 = list(np.copy(self.npz_label_to_indices[3]))

        #dict=label:indices where labels occurs/same as ucsf_label_to_indices
        npz_index = {}
        npz_index[0] = npz_0
        npz_index[1] = npz_1
        npz_index[2] = npz_2
        npz_index[3] = npz_3

        #shuffle indices for each label
        #for i in range(0,3):
        for i in range(0,4):
            np.random.shuffle(npz_index[i])
        npz_index_copy = copy.deepcopy(npz_index)
        #retrieve indices with each label
        missing_0 = list(np.copy(self.missing_label_to_indices[0]))
        missing_1 = list(np.copy(self.missing_label_to_indices[1]))
        missing_2 = list(np.copy(self.missing_label_to_indices[2]))
        missing_3 = list(np.copy(self.missing_label_to_indices[3]))

        #dict=label:indices where labels occurs/same as ucsf_label_to_indices
        missing_index = {}
        missing_index[0] = missing_0
        missing_index[1] = missing_1
        missing_index[2] = missing_2
        missing_index[3] = missing_3

        #shuffle indices for each label
        #for i in range(0,3):
        for i in range(0,4):
            np.random.shuffle(missing_index[i])
       
        missing_index_copy = copy.deepcopy(missing_index)

        indices = []
        output = {}
        i = 0
        #make each batch (a list of indices drawn without replacement per label)
        while True in ([len(j)>=2 for j in missing_index.values()]) or True in ([len(j)>=2 for j in npz_index.values()]):
            output[i]=[]
            try:
              npz_index, missing_index = self.get_index(npz_index, missing_index, output[i])
            except:
              break
            i += 1
  
        #'flatten' batch dictionary into list
        for i in range(0,self.num_batches):
            indices.extend(output[i])

        #return iterator yielding pairs (0,0) --> (1,1) --> (2,2) --> (0,0) --> ...
        return iter(indices)

    def __len__(self):
        return 2 #because pair


    def get_index(self, npz_index, missing_index, output):
        #while there are still items in the index lists
        #iterate over labels i:[0,1,2]
        #pop item with label i off list
        #when label i exhausted, pop from label list
        num_added = 0
        labels = [0,1,2,3]
        #labels = [0,1,2]
        while len(labels) > 0:
            for j in labels:
                pair = None
                #when num_added to output list = batch size, 
                #break loop and return current dictionaries
                if num_added == self.batch_size:
                    return npz_index, missing_index
                #while more than one index is yet unassigned,
                #remove the first two elements remaining and add to to the batch
                if j in npz_index.keys() and len(npz_index[j])>1:
                    pair = (npz_index[j].pop(0), npz_index[j].pop(0))
                    num_added += 2
                elif j in missing_index.keys() and len(missing_index[j])>1:
                    if j in npz_index.keys():
                        npz_index.pop(j)
                    pair = (missing_index[j].pop(0), missing_index[j].pop(0))
                    num_added += 2
                else:
                    if j in labels:
                        labels.pop(labels.index(j))
                if pair != None:
                    output.append(pair)
