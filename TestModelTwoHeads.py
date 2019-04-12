# -*- coding: utf-8 -*-
"""
QL
"""
import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:",matplotlib.get_backend())

from keras import backend as K
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import librosa
import random
import numpy as np
import sklearn.cluster as skcluster
import scipy.signal as sg
from hparams import hparams

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")



K.set_learning_phase(0) #set learning phase
K.set_image_data_format('channels_last')
# to check if there are gpu available
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUFlag = True
ExistGPUs = get_available_gpus()
if len(ExistGPUs)==0:
    GPUFlag = False


# setting the hyper parameters
import os
import argparse

# setting the hyper parameters
parser = argparse.ArgumentParser(description="TestEncoder")
parser.add_argument('--srcNum', default=3, type=int, help="Number of sources")
parser.add_argument('--featureFlag', default=0, type=int)  # featureFlag (0 spectrum shift1 shift2, 1 spectrum angeldistance)
parser.add_argument('--outputFlag', default=1, type=int)  # outputFlag (0 mse, 1 perceptual weight)

args = parser.parse_args()
print(args)

############################################################
dst_dir = "./TempResults/Mixture{}".format(args.srcNum)
os.makedirs(dst_dir, exist_ok=True)


# # ###########################################################
# # ###########################################################
# # ############################################################
# # from GenerateData import dataGen, ExtractFeatureOneMixture, NormLP2Spec
# # dg = dataGen(seedNum = 123456789, verbose = False, verboseDebugTime=False)
# # # [aa,bb] = dg.myDataGenerator(0)# change yield to return to debug the generator
# # # generate N = 200 samples for testing
# # random.seed(666666)
# # N = 200
# # chosenSeqsList = []
# # chosenAngsList = []
# # chosenSeqsList3 = []
# # chosenAngsList3 = []
# # for i in range(N):
# #     print(i)
# #     chosenSeqs = dg.randomNseq(dg.targetListTest,2)
# #     chosenAngs = dg.randomNang(2)
# #     chosenSeqs3 = dg.randomNseq(dg.targetListTest, 3)
# #     chosenAngs3 = dg.randomNang(3)
# #     # test if these audio files can be read or not
# #     successFlag = False
# #     while not successFlag:
# #         try:
# #             for seq in chosenSeqs:
# #                 source_i, sr = librosa.load(seq, sr=None)
# #             successFlag = True
# #         except:
# #             print("Loading data {} fail".format(chosenSeqs))
# #             chosenSeqs = dg.randomNseq(dg.targetListTest, 2)
# #
# #     successFlag = False
# #     while not successFlag:
# #         try:
# #             for seq in chosenSeqs3:
# #                 source_i, sr = librosa.load(seq, sr=None)
# #             successFlag = True
# #         except:
# #             print("Loading data {} fail".format(chosenSeqs3))
# #             chosenSeqs3 = dg.randomNseq(dg.targetListTest, 3)
# #
# #     chosenSeqsList.append(chosenSeqs)
# #     chosenAngsList.append(chosenAngs)
# #     chosenSeqsList3.append(chosenSeqs3)
# #     chosenAngsList3.append(chosenAngs3)
# #     # print(chosenSeqs,chosenAngs)
# #
# # import pickle
# # # Saving the objects:
# # with open('TestingSignalList.pkl', 'wb') as f:  # Python 2 open(..., 'w') Python 3: open(..., 'wb')
# #     pickle.dump([chosenSeqsList, chosenAngsList, chosenSeqsList3, chosenAngsList3, dg.rirs, dg.thetaNorm, dg.batchLen], f)
#
# import pickle
# with open('TestingSignalList.pkl', 'rb') as f:
#     chosenSeqsList, chosenAngsList, chosenSeqsList3, chosenAngsList3, rirs, thetaNorm, batchLen = pickle.load(f)
# for sample_i in range(len(chosenSeqsList)): # N
#     chosenSeqs = chosenSeqsList[sample_i]
#     chosenSeqs3 = chosenSeqsList3[sample_i]
#     chosenAngs = chosenAngsList[sample_i]
#     chosenAngs3 = chosenAngsList3[sample_i]
#     print(chosenAngs,chosenSeqs,'\n',chosenAngs3,chosenSeqs3,'\n')
#     # try:
#     #     for seq in chosenSeqs:
#     #         source_i, sr = librosa.load(seq, sr=None)
#     #     print("Success loading data {}".format(chosenSeqs))
#     # except:
#     #     print("Loading data {} fail".format(chosenSeqs))
#     #
#     # try:
#     #     for seq in chosenSeqs3:
#     #         source_i, sr = librosa.load(seq, sr=None)
#     #     print("Success loading data {}".format(chosenSeqs3))
#     # except:
#     #     print("Loading data {} fail".format(chosenSeqs3))



import pickle
from GenerateData import ExtractFeatureOneMixture, NormLP2Spec
# Getting back the objects:
with open('TIMITtestingSignalList.pkl', 'rb') as f:
    chosenSeqsList, chosenAngsList, chosenSeqsList3, chosenAngsList3, rirs, thetaNorm, batchLen = pickle.load(f)
if args.srcNum==3:
    chosenSeqsList = chosenSeqsList3
    chosenAngsList = chosenAngsList3
N = len(chosenSeqsList)


# file_name_save = "./TestingSignalList.mat"
# with open('TestingSignalList.txt', 'w') as f:
#     for item in chosenSeqsList:
#         f.write("%s\n" % item)
# import scipy.io as sio
# sio.savemat(file_name_save, mdict={'chosenAngsList':chosenAngsList, 'rirs':rirs, 'thetaNorm':thetaNorm, 'batchLen':batchLen})



from keras.utils.generic_utils import get_custom_objects
if args.outputFlag==0:
    from LossFunctions import my_loss_DC_Nsrc as LossA, my_loss_MSE_Nsrc as LossB

    # loss = SSD_Loss(neg_pos_ratio=neg_pos_ratio, alpha=alpha)
    # get_custom_objects().update({"SSD_Loss": loss.computeloss})
    get_custom_objects().update({'my_loss_DC_Nsrc_Ref': LossA(Nsrc=args.srcNum), 'my_loss_MSE_Nsrc_Ref': LossB(Nsrc=args.srcNum)})
elif args.outputFlag==1:
    from LossFunctions import weight_loss_DC_Nsrc as LossA, weight_loss_MSE_Nsrc as LossB
    get_custom_objects().update({'weight_loss_DC_Nsrc_Ref': LossA(Nsrc=args.srcNum), 'weight_loss_MSE_Nsrc_Ref': LossB(Nsrc=args.srcNum)})


from GenerateModels import ChimeraNet as GenerateModel
tag = 'Feature{}Nsrc{}{}'.format(['Proposed','SpecSpat','SpecOnly'][args.featureFlag],args.srcNum,['','Wei'][args.outputFlag])
print(tag)
modelDirectory = './TrainedModels'

# from GenerateModelsOld import ChimeraNet as GenerateModel
# tag = 'ChimeraNsrc{}{}'.format(args.srcNum,['','Wei'][args.outputFlag])
# print(tag)
# modelDirectory = '/Directory2TrainedModels/ChimeraModels'


# Load the model
EncoderModel = GenerateModel(Nsrc = args.srcNum, inputFeature = args.featureFlag)
# The separation model
from keras.models import load_model
EncoderModel = load_model("{}/{}/modelsave.h5".format(modelDirectory,tag))
# EncoderModel = load_model("{}/{}/modelsave.h5".format(modelDirectory,tag), custom_objects={'loss_DC': LossA(Nsrc=args.srcNum), 'lossMSE': LossB(Nsrc=args.srcNum)})

print(EncoderModel.summary())
# plot_model(train_model, to_file='Model.png')

import itertools

# tag = 'IBM' # comment this out later, this is for IBM baseline
# tag = 'GMM' # comment this out later, this is for GMM baseline

mseResults = np.zeros((N,),dtype='float32')
mseResults2 = np.zeros((N,),dtype='float32')
# now apply source separation
for sample_i in range(N): # N
    print('Now separate signals ------ {}'.format(sample_i))
    chosenSeqs = chosenSeqsList[sample_i]
    chosenAzis = chosenAngsList[sample_i]

    chosenAzis.sort()
    print(chosenAzis)
    # chosenAzis = self.randomNang(3)
    # chosenAzis.sort(reverse=True)
    chosenAziIndex = (np.asarray(chosenAzis) / 10).astype(int)
    RIRsUse = rirs[chosenAziIndex]
    musUse = thetaNorm[chosenAziIndex]


    chooseIndexNormalised = None
    (DNN_in, groundtruthIBM, groundtruthIRM, chooseIndex, two_angles) = \
        ExtractFeatureOneMixture(args.featureFlag, args.outputFlag, chosenSeqs, RIRsUse, musUse, chooseIndexNormalised, batchLen)
    input_angle, MIX_angle = two_angles[0], two_angles[1]
    # MIX_angle is the DOA features from the mixture, which is the same as DNN_in[:,:,:,1] when args.featureFlag==1

    Nframe = 64
    Nfrequency = 512
    EMBEDDINGS_DIMENSION = 20
    Noutput = Nfrequency

    # Mix_angle_shift = DNN_in[:, :, :, 1::]
    # masks = DNN_in/np.sum(Mix_angle_shift,axis=-1,keepdims=True)

    # apply the model to get the embeddings
    [estimate_embed, estimate_irm] = EncoderModel.predict(DNN_in)  # The embedding
    V = np.reshape(estimate_embed, [-1, Nframe * Noutput, EMBEDDINGS_DIMENSION])
    # V = 1 / np.sqrt(np.sum(np.square(V), axis=-1, keepdims=True)) * V

    # apply kmeans clustering to the embeddings
    centroid = np.zeros((len(chooseIndex), args.srcNum, EMBEDDINGS_DIMENSION), dtype='f')
    labels = np.zeros((len(chooseIndex), Nframe * Noutput), dtype='int32')
    for i in range(len(chooseIndex)):
        centroid[i], labels[i], _ = skcluster.k_means(V[i], args.srcNum)

    #################### solve the permutation problem of the DC head
    # to generate one-hot IBM masks for srcNum sources
    masks = np.zeros(labels.shape + (args.srcNum,)).astype(np.float32)  # (nums,WxT,nsrc)
    for i in range(0, len(chooseIndex)):
        label = labels[i]
        Y = masks[i]
        for source_i in range(args.srcNum):
            t = np.zeros(args.srcNum).astype(int)
            t[source_i] = 1
            Y[label == source_i] = t
        masks[i] = Y

    masks = np.reshape(masks, (len(chooseIndex), Nframe, Noutput, args.srcNum))

    if args.featureFlag == 0:
        Mix_angle_shift = DNN_in[:, :, :, 1::]
    else:
        Mix_angle_shift = np.tile(MIX_angle, [args.srcNum, 1, 1, 1])

        for source_i in range(args.srcNum):
            temp = Mix_angle_shift[source_i]
            temp -= musUse[source_i]
            temp = (temp + np.pi) % (2 * np.pi) - np.pi
            # nonlinearly convert the angle distance
            temp = np.power(temp, 2)
            temp = np.exp(-temp / hparams.dspatial_sigma2)
            Mix_angle_shift[source_i] = temp
        Mix_angle_shift = np.transpose(Mix_angle_shift,[1,2,3,0])

    for i in range(len(chooseIndex)):
        best_path = np.arange(args.srcNum)
        best_corr = -100000000  # maximising this one
        currentmask = masks[i]
        currentref = Mix_angle_shift[i]

        for path in itertools.permutations(np.arange(args.srcNum)):
            path = np.asarray(path)
            # print(path)
            tempmasks = currentmask[:, :, path]
            corr = np.sum(tempmasks * currentref)
            if corr > best_corr:
                best_path = path
                best_corr = corr

        if not np.array_equal(best_path,np.arange(args.srcNum)):
            masks[i] = currentmask[:, :, best_path]


    # index = 2
    # plt.figure(88)
    # plt.title('Groundtruth, mixtures and estimations')
    # ax1 = plt.subplot(411)
    # im1 = ax1.pcolor(groundtruthIBM[index, :, :, 0].T)
    # plt.colorbar(im1)
    # ax2 = plt.subplot(412, sharex=ax1)
    # im2 = ax2.pcolor(DNN_in[index, :, :, 1].T)
    # plt.colorbar(im2)
    # ax3 = plt.subplot(413, sharex=ax1)
    # im3 = ax3.pcolor(estimate_irm[index, :, :, 0].T)
    # plt.colorbar(im3)
    # ax4 = plt.subplot(414, sharex=ax1)
    # im4 = ax4.pcolor(estimate_irm[index, :, :, 1].T)
    # plt.colorbar(im4)
    # plt.show()



    # output1 = output_mask.astype('float32')
    # output2 = (1-output_mask).astype('float32')
    #
    # mixLP = DNN_in[:, :, :, 0]
    # if args.featureFlag!=1:
    #     mask1 = DNN_in[:, :, :, 1]
    #     mask2 = DNN_in[:, :, :, 2]
    # else:
    #     MIX_angle_shift1 = DNN_in[:, :, :, 1].copy()
    #     MIX_angle_shift2 = DNN_in[:, :, :, 1].copy()
    #
    #     MIX_angle_shift1 -= mu1mu2[0]
    #     MIX_angle_shift1 = (MIX_angle_shift1 + np.pi) % (2 * np.pi) - np.pi
    #     MIX_angle_shift1 = np.power(MIX_angle_shift1, 2)
    #     mask1 = np.exp(-MIX_angle_shift1 / hparams.dspatial_sigma2)
    #
    #     MIX_angle_shift2 -= mu1mu2[1]
    #     MIX_angle_shift2 = (MIX_angle_shift2 + np.pi) % (2 * np.pi) - np.pi
    #     MIX_angle_shift2 = np.power(MIX_angle_shift2, 2)
    #     mask2 = np.exp(-MIX_angle_shift2 / hparams.dspatial_sigma2)
    #
    #path
    # temp11 = np.sum(output1 * mask1)
    # temp22 = np.sum(output2 * mask2)
    # temp12 = np.sum(output1 * mask2)
    # temp21 = np.sum(output2 * mask1)


    mixLP = DNN_in[:, :, :, 0]

    estimate_specs = np.zeros((args.srcNum,chooseIndex[-1] + batchLen, Noutput), dtype='float32')
    groundtruth_specs = np.zeros((args.srcNum,chooseIndex[-1] + batchLen, Noutput), dtype='float32')
    est_IBMmasks = np.zeros((args.srcNum,chooseIndex[-1] + batchLen, Noutput), dtype='float32')
    est_IRMmasks = np.zeros((args.srcNum, chooseIndex[-1] + batchLen, Noutput), dtype='float32')
    mix_spec = np.zeros((chooseIndex[-1] + batchLen, Noutput), dtype='float32')

    for n, i in enumerate(chooseIndex):
        mix_spec[i:i + batchLen] = mixLP[n]

        for source_i in range(args.srcNum):
            # tempMask = masks[n, :, :, source_i]
            # temp = mixLP[n] + 20 * np.log10(np.maximum(tempMask, 0.001)) / hparams.norm_scale_db

            tempMask = groundtruthIBM[n, :, :, source_i]
            temp = mixLP[n] + 20 * np.log10(np.maximum(tempMask, 0.001)) / hparams.norm_scale_db
            groundtruth_specs[source_i,i:i + batchLen] = temp

            tempMask = masks[n, :, :, source_i]
            est_IBMmasks[source_i,i:i + batchLen] = tempMask


            irm_mask = estimate_irm[n, :, :, source_i]
            est_IRMmasks[source_i, i:i + batchLen] = irm_mask

            # Use either the DC head or soft mask head

            # # DC head
            # temp = mixLP[n] + 20 * np.log10(np.maximum(tempMask, 0.001)) / hparams.norm_scale_db

            # Soft head
            temp = mixLP[n] + 20 * np.log10(np.maximum(irm_mask, 0.001)) / hparams.norm_scale_db

            estimate_specs[source_i,i:i + batchLen] = temp

    # index = 2
    # plt.figure(99,figsize=(6, 10))
    # plt.title('Groundtruth, mixtures and estimations')
    # ax1 = plt.subplot(711)
    # im1 = ax1.pcolor(mix_spec.T)
    # plt.colorbar(im1)
    # ax2 = plt.subplot(712, sharex=ax1)
    # im2 = ax2.pcolor(groundtruth_specs[0].T)
    # plt.colorbar(im2)
    # ax3 = plt.subplot(713, sharex=ax1)
    # im3 = ax3.pcolor(groundtruth_specs[1].T)
    # plt.colorbar(im3)
    # ax4 = plt.subplot(714, sharex=ax1)
    # im4 = ax4.pcolor(estimate_specs[0].T)
    # plt.colorbar(im4)
    # ax5 = plt.subplot(715, sharex=ax1)
    # im5 = ax5.pcolor(estimate_specs[1].T)
    # plt.colorbar(im5)
    # ax6 = plt.subplot(716, sharex=ax1)
    # im6 = ax6.pcolor(est_IRMmasks[0].T)
    # plt.colorbar(im6)
    # ax7 = plt.subplot(717, sharex=ax1)
    # im7 = ax7.pcolor(est_IRMmasks[1].T)
    # plt.colorbar(im7)
    # plt.show()
    #






    #
    # index = 2
    # plt.figure(100,figsize=(6, 8))
    # plt.title('Groundtruth, mixtures and estimations')
    # ax1 = plt.subplot(411)
    # im1 = ax1.pcolor(groundtruthIBM[index, :, :, 0].T)
    # plt.colorbar(im1)
    # ax2 = plt.subplot(412, sharex=ax1)
    # im2 = ax2.pcolor(DNN_in[index, :, :, 1].T)
    # plt.colorbar(im2)
    # ax3 = plt.subplot(413, sharex=ax1)
    # im3 = ax3.pcolor(masks[index,:,:,0].T)
    # plt.colorbar(im3)
    # ax4 = plt.subplot(414, sharex=ax1)
    # im4 = ax4.pcolor(estimate_irm[index,:,:,0].T)
    # plt.colorbar(im2)
    # plt.show()


    #reconstruct signals from estimate_specs

    zeros_end = np.zeros(shape=(1,chooseIndex[-1] + batchLen), dtype='float32')
    input_angle = np.concatenate((input_angle,zeros_end),axis=0)
    input_angle_exp = np.exp(1j * input_angle)

    for source_i in range(args.srcNum):
        estimate_spec_complex = NormLP2Spec(estimate_specs[source_i], input_angle_exp)
        dst_wav_path = "{}/Ind_{}_{}Soft_est{}.wav".format(dst_dir, sample_i + 1, tag, source_i)

        # estimate_spec_complex = NormLP2Spec(groundtruth_specs[source_i], input_angle_exp)
        # dst_wav_path = "{}/Ind_{}_{}_src{}.wav".format(dst_dir, sample_i + 1, tag, source_i)

        [_, s_est] = sg.istft(estimate_spec_complex, fs=hparams.sample_rate, noverlap=hparams.overlap)
        librosa.output.write_wav(dst_wav_path, s_est, sr=hparams.sample_rate)

    a = 0








    # # solve the global permutation problem
    # # to generate one-hot IBM masks for srcNum sources
    # masks = np.zeros(labels.shape+(args.srcNum,)).astype(np.float32)  # (nums,WxT,nsrc)
    # for i in range(0, len(chooseIndex)):
    #     label = labels[i]
    #     Y = masks[i]
    #     for source_i in range(args.srcNum):
    #         t = np.zeros(args.srcNum).astype(int)
    #         t[source_i] = 1
    #         Y[label == source_i] = t
    #     masks[i] = Y
    #
    # masks = np.reshape(masks,(len(chooseIndex), Nframe, Noutput,args.srcNum))
    #
    # best_path = np.arange(args.srcNum)
    # best_corr = -100000000  # maximising this one
    # for path in itertools.permutations(np.arange(args.srcNum)):
    #     path = np.asarray(path)
    #     # print(path)
    #     tempmasks = masks[:,:,:,path]
    #     corr = np.sum(tempmasks*DNN_in[:,:,:,1::])
    #     if corr > best_corr:
    #         best_path = path
    #         best_corr = corr
    # masks = masks[:, :, :, best_path]


    # if args.featureFlag==0:
    #     if temp11+temp22>temp12+temp21: # the second output is the target temp11 > temp21:
    #         estimate1 = np.maximum(output1, estimate_irm[:, :, :, 0])
    #         estimate2 = np.maximum(output2, estimate_irm[:, :, :, 1])
    #     else:
    #         estimate1 = np.maximum(output2, estimate_irm[:, :, :, 0])
    #         estimate2 = np.maximum(output1, estimate_irm[:, :, :, 1])
    #
    # elif args.featureFlag==1:
    #     if temp11+temp22>temp12+temp21: # the second output is the target temp11 > temp21:
    #         estimate1 = output1
    #         estimate2 = output2
    #     else:
    #         estimate1 = output2
    #         estimate2 = output1
    #
    # elif args.featureFlag == 2:
    #     estimate1 = output1
    #     estimate2 = output2

    # # solve the permutation problem between neighboring frames for the DC head
    # for i in range(1, len(chooseIndex)):
    #     pc = centroid[i - 1]  # previous centroid
    #     cc = centroid[i]  # current centroid
    #     best_path = np.arange(args.srcNum)
    #     best_dist=10000000000 # minimise this one
    #     for path in itertools.permutations(np.arange(args.srcNum)):
    #         path = np.asarray(path)
    #         # print(path)
    #         tempcc = cc[path]
    #         dist = np.linalg.norm(tempcc-pc)
    #         if dist<best_dist:
    #             best_path = path
    #             best_dist = dist
    #
    #     centroid[i] = cc[best_path]
    #     label = labels[i]
    #     def simplyMap(x, best_path):
    #         return best_path[x]
    #     vfunc = np.vectorize(simplyMap)
    #     # https://stackoverflow.com/questions/34065412/np-vectorize-giving-me-indexerror-invalid-index-to-scalar-variable
    #     vfunc.excluded.add(1)
    #     label = vfunc(label, best_path)
    #     labels[i] = label


    # for i in range(1, len(chooseIndex)):
    #     pc = centroid[i - 1]  # previous centroid
    #     cc = centroid[i]  # current centroid
    #     d00 = np.sum(np.square(pc[0] - cc[0]))
    #     d01 = np.sum(np.square(pc[0] - cc[1]))
    #     d10 = np.sum(np.square(pc[1] - cc[0]))
    #     d11 = np.sum(np.square(pc[1] - cc[1]))
    #
    #     if (d00 + d11) > (d01 + d10):  # do the permutation
    #         cc = cc[::-1]
    #         centroid[i] = cc
    #         labels[i] = 1 - labels[i]
    #
    # # to generate one mask for the target
    # output_mask = np.reshape(labels, (len(chooseIndex), Nframe, Noutput))

    # estimate_spec1 = np.zeros((chooseIndex[-1] + batchLen, Noutput),dtype='float32')
    # estimate_spec2 = np.zeros((chooseIndex[-1] + batchLen, Noutput),dtype='float32')
    # groundtruth_spec1 = np.zeros((chooseIndex[-1] + batchLen, Noutput),dtype='float32')
    # groundtruth_spec2 = np.zeros((chooseIndex[-1] + batchLen, Noutput), dtype='float32')
    # est1_mask = np.zeros((chooseIndex[-1] + batchLen, Noutput), dtype='float32')
    # est2_mask = np.zeros((chooseIndex[-1] + batchLen, Noutput), dtype='float32')
    # mix_spec = np.zeros((chooseIndex[-1] + batchLen, Noutput),dtype='float32')
    # for n, i in enumerate(chooseIndex):
    #     mix_spec[i:i + batchLen] = mixLP[n]
    #     tempMask = groundtruthIBM[n, :, :, 0]
    #     temp = mixLP[n] + 20*np.log10(np.maximum(tempMask,0.001))/hparams.norm_scale_db
    #     groundtruth_spec1[i:i + batchLen] = temp
    #     tempMask = groundtruthIBM[n, :, :, 1]
    #     temp = mixLP[n] + 20 * np.log10(np.maximum(tempMask, 0.001)) / hparams.norm_scale_db
    #     groundtruth_spec2[i:i + batchLen] = temp
    #
    #     est1_mask[i: i + batchLen] = estimate1[n]
    #     est2_mask[i: i + batchLen] = estimate2[n]
    #
    #     temp = mixLP[n] + 20 * np.log10(np.maximum(estimate1[n],0.001)) / hparams.norm_scale_db
    #     estimate_spec1[i:i + batchLen] = temp
    #     temp = mixLP[n] + 20 * np.log10(np.maximum(estimate2[n], 0.001)) / hparams.norm_scale_db
    #     estimate_spec2[i:i + batchLen] = temp

    # index = 2
    # plt.figure(99)
    # plt.title('Groundtruth, mixtures and estimations')
    # ax1 = plt.subplot(511)
    # im1 = ax1.pcolor(mix_spec.T)
    # plt.colorbar(im1)
    # ax2 = plt.subplot(512, sharex=ax1)
    # im2 = ax2.pcolor(groundtruth_spec1.T)
    # plt.colorbar(im2)
    # ax3 = plt.subplot(513, sharex=ax1)
    # im3 = ax3.pcolor(groundtruth_spec2.T)
    # plt.colorbar(im3)
    # ax4 = plt.subplot(514, sharex=ax1)
    # im4 = ax4.pcolor(estimate_spec1.T)
    # plt.colorbar(im4)
    # ax5 = plt.subplot(515, sharex=ax1)
    # im5 = ax5.pcolor(estimate_spec2.T)
    # plt.colorbar(im5)
    # plt.show()


    # index = 2
    # plt.figure(100)
    # plt.title('Groundtruth, mixtures and estimations')
    # ax1 = plt.subplot(411)
    # im1 = ax1.pcolor(groundtruthIBM[index, :, :, 0].T)
    # plt.colorbar(im1)
    # ax2 = plt.subplot(412, sharex=ax1)
    # im2 = ax2.pcolor(DNN_in[index, :, :, 1].T)
    # plt.colorbar(im2)
    # ax3 = plt.subplot(413, sharex=ax1)
    # im3 = ax3.pcolor(estimate1[index].T)
    # plt.colorbar(im3)
    # ax4 = plt.subplot(414, sharex=ax1)
    # im4 = ax4.pcolor(estimate_irm[index,:,:,0].T)
    # plt.colorbar(im2)
    # plt.show()




    # #reconstruct signals from estimate_spec1 and estimate_spec2
    # zeros_end1 = np.zeros(shape=(1,chooseIndex[-1] + batchLen), dtype='float32')
    # input_angle = np.concatenate((input_angle,zeros_end1),axis=0)
    # input_angle_exp = np.exp(1j * input_angle)
    #
    # # estimate_spec1_complex = NormLP2Spec(groundtruth_spec1,input_angle_exp)
    # # estimate_spec2_complex = NormLP2Spec(groundtruth_spec2, input_angle_exp)
    #
    # estimate_spec1_complex = NormLP2Spec(estimate_spec1,input_angle_exp)
    # estimate_spec2_complex = NormLP2Spec(estimate_spec2, input_angle_exp)
    #
    # [_, s1_est] = sg.istft(estimate_spec1_complex, fs=hparams.sample_rate, noverlap=hparams.overlap)
    # [_, s2_est] = sg.istft(estimate_spec2_complex, fs=hparams.sample_rate, noverlap=hparams.overlap)
    #
    #
    #
    # dst_wav_path = "{}/Ind_{}_{}_est1.wav".format(dst_dir, sample_i+1, tag2)
    # librosa.output.write_wav(dst_wav_path, s1_est, sr=hparams.sample_rate)
    # dst_wav_path = "{}/Ind_{}_{}_est2.wav".format(dst_dir, sample_i+1, tag2)
    # librosa.output.write_wav(dst_wav_path, s2_est, sr=hparams.sample_rate)
    #
    # a = 0




