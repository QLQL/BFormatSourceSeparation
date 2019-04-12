# # ############################################################################
# # The following resamples the recorded BRIRs from 48k to 16k
#
# import librosa
# from hparams import hparams
# import scipy.io as sio
# import numpy as np
#
# RIRs = sio.loadmat('./RIRs/B_format_RIRs_12BB01_Alfredo_S3A.mat')['rirs_final']
# n_channels = 4
# n_doas = 36
# RIRs_len = len(librosa.core.resample(RIRs[0, 0, :], 48000, hparams.sample_rate))
# # resample to 16kHz
# RIRs_resample = np.zeros([n_doas, n_channels, RIRs_len])
# for doa in range(n_doas):
#     for ch in range(n_channels):
#         RIRs_resample[doa, ch, :] = librosa.core.resample(RIRs[doa, ch, :], 48000, hparams.sample_rate)
#
#
# file_name_save = "./B_format_RIRs_12BB01_Alfredo_S3A_16k.mat"
# # sio.savemat(file_name_save, mdict={'rirs': RIRs_resample})



#############################################################################
# The following generates lists for training, validation and testing data

import os
import random
from random import shuffle
seedNum = 123456789
random.seed(seedNum)


# # Make a nested list based on the name
# def nestedListFromName(alist):
#     IDList = []
#     NestedList = []
#     for file in alist:
#         # temp = os.path.basename(file)[:2]
#         fileSplit = file.split(os.sep)
#         temp = fileSplit[-2]
#         if temp not in IDList:
#             IDList.append(temp)
#             NestedList.append([file])
#         else:
#             ind = IDList.index(temp)
#             NestedList[ind].append(file)
#
#     return IDList, NestedList


target_data_root1 = '/DirectoryToTIMIT/TIMIT/Clean/timit/train/'
target_data_root2 = '/DirectoryToTIMIT/TIMIT/Clean/timit/test/'


for target_data_root in [target_data_root1,target_data_root2]:
    targetList = []
    for path, subdirs, files in os.walk(target_data_root):
        pathSplit = path.split(os.sep)
        print((len(pathSplit) - 1) * '-', os.path.basename(path))
        for filename in files:
            if filename.endswith(".wav"):
                if filename not in ['sa1.wav','sa2.wav']:
                    targetList.append(os.path.join(path, filename))


    if target_data_root==target_data_root1:
        # split to training and validation 80-20
        trainNum = int(round(len(targetList) * 0.8))
        validNum = len(targetList) - trainNum
        shuffle(targetList)
        targetListTrain = targetList[:trainNum]
        targetListValid = targetList[trainNum:]

    else:
        shuffle(targetList)
        targetListTest = targetList

# import pickle
#
# # Saving the objects:
# with open('trainingSignalList.pkl', 'wb') as f:  # Python 2 open(..., 'w') Python 3: open(..., 'wb')
#     pickle.dump([targetListTrain, targetListValid, targetListTest], f)
#
# # Getting back the objects:
# with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
#     obj0, obj1, obj2 = pickle.load(f)

a = 0

