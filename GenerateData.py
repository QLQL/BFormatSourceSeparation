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



import numpy as np
import scipy.io as sio
import pickle
import scipy.signal as sg
import os
import sys
from random import shuffle
import random
import time
import re
from hparams import hparams
import datetime
import librosa

from multiprocessing import cpu_count
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial


multiProcFlag = True


def STFT2dB(x): # get the dB scale feature
    X = 20 * np.log10(np.abs(x))
    return X

def dB2Mag(x): #get the magnitude from the dB scale
    X = np.power(10,(x/20))
    return X


def NormaliseLP(X): # normalise the LP spectrum

    X_norm = np.clip((X - hparams.min_level_db) / hparams.norm_scale_db, 0, None)

    # plt.figure(17)
    # plt.subplot(211)
    # temp = np.reshape(X, (np.product(X.shape),))
    # plt.hist(temp, density=True, bins=100)
    # plt.ylabel('Probability')
    # plt.subplot(212)
    # temp = np.reshape(X_norm, (np.product(X.shape),))
    # plt.hist(temp, density=True, bins=100)
    # plt.ylabel('Probability')
    # plt.show()

    return X_norm

def DenormaliseLP(X_norm): # denormalise the LP spectrum

    X = X_norm*hparams.norm_scale_db + hparams.min_level_db

    return X


def NormLP2Spec(x, input_angle_exp): # from denormalised LP spectrum to the original complex STFT spectrum
    # x                     ndarray T x NFFT/2
    # input_angle_exp       ndarray NFFT/2+1 x T        exp(1j*input_angle)
    zeros_end = np.zeros((x.shape[0], 1), dtype='float32')

    x = np.clip(x, 0, None)
    x = DenormaliseLP(x)
    x = dB2Mag(x)
    x = np.concatenate((x, zeros_end), axis=1).T
    x_complex = x * input_angle_exp
    return x_complex


def ApplySTFT(sig):
    [_, _, SIG] = sg.stft(sig, hparams.sample_rate, 'hann', hparams.fft_size, hparams.overlap)
    SIG = SIG[:-1].astype(np.complex64) # keep only the first NFFT/2 bins
    return SIG

def STFTtensor(inputTensor):
    # STFT
    P0 = ApplySTFT(inputTensor[0])
    Gx = ApplySTFT(inputTensor[1])
    Gy = ApplySTFT(inputTensor[2])

    # fill tensor
    X = np.stack([P0, Gx, Gy], axis=0)

    return X


def angles_dist(P0GxGy):
    Y = (np.conj(P0GxGy[0]) * P0GxGy[2]).real
    X = (np.conj(P0GxGy[0]) * P0GxGy[1]).real
    theta = np.arctan2(Y, X) # in the range [-pi, pi]

    return theta


def ExtractFeatureOneMixture(featureFlag, outputFlag, sourceNames, RIRs, mus, chooseIndexNormalised, batchLen):
# def ExtractFeatureOneMixture(featureFlag, outputFlag, sourceNames, RIRs, mus, chooseIndexNormalised, batchLen, saveIndex): # only for saving mixtures and sources
    try:
        Nsrc = len(sourceNames)
        sourceList = []
        L = 0
        for i in range(Nsrc):
            source_i, sr = librosa.load(sourceNames[i], sr=None)
            if source_i is None:
                print("Data {} loading fail".fomat(sourceNames[i]))
            L = max(L,len(source_i))
            sourceList.append(source_i)

        for i in range(Nsrc):
            tempSource = sourceList[i]
            source = sourceList[i]
            while len(source) < L:
                source = np.concatenate((source,tempSource), axis=0)
            sourceList[i] = source[:L]

        # plt.figure(10)
        # plt.subplot(211)
        # plt.plot(sourceList[0])
        # plt.subplot(212)
        # plt.plot(sourceList[1],'r')

    except:
        print('Check you code for loading the data!')
        print(sourceNames)
        # sys.exit()


    try:
        # Generate the mixture
        # the RIRs for the target and the interference respectively
        mix_s_all = []

        for source_i in range(Nsrc):
            source = sourceList[source_i]
            p0 = sg.fftconvolve(source, RIRs[source_i][0])
            vel_x = sg.fftconvolve(source, RIRs[source_i][1])
            vel_y = sg.fftconvolve(source, RIRs[source_i][2])

            mix_s = np.stack([p0, vel_x, vel_y], axis=0)
            mix_s = mix_s[:, :L]

            mix_s_all.append(mix_s)


            if source_i==0:
                mix = mix_s.copy()
            else:
                mix += mix_s

            # plt.figure()
            # plt.plot(mix_s_all[source_i][0])
            #
            # scale = 1 / np.max(np.abs(mix_s[0])) * hparams.rescaling_max
            # dst_wav_path = "./aaaaa.wav"
            # librosa.output.write_wav(dst_wav_path, mix_s_all[0][0] * scale, sr=hparams.sample_rate)
            # dst_wav_path = "./bbbbb.wav"
            # librosa.output.write_wav(dst_wav_path, mix_s[0] * scale, sr=hparams.sample_rate)

        # plt.figure(19)
        # plt.subplot(211)
        # plt.plot(RIRs[0].T)
        # plt.subplot(212)
        # plt.plot(RIRs[1].T)
        #
        # plt.figure(20)
        # plt.subplot(211)
        # plt.plot(sourceList[0])
        # plt.subplot(212)
        # plt.plot(sourceList[1], 'r')
        #
        # plt.figure(21)
        # plt.subplot(211)
        # plt.subplot(211)
        # plt.plot(mix_s_all[0][0])
        # plt.subplot(212)
        # plt.plot(mix_s_all[1][0], 'r')

        # scale = 1 / np.max(np.abs(mix[0])) * hparams.rescaling_max
        # dst_wav_path = "./aaaaa.wav"
        # librosa.output.write_wav(dst_wav_path, mix_s_all[0][0]*scale, sr=hparams.sample_rate)
        # dst_wav_path = "./bbbbb.wav"
        # librosa.output.write_wav(dst_wav_path, mix_s_all[1][0]*scale, sr=hparams.sample_rate)
        # dst_wav_path = "./mix.wav"
        # librosa.output.write_wav(dst_wav_path, mix[0] * scale, sr=hparams.sample_rate)

        # scale = 1 / np.max(np.abs(mix[0])) * hparams.rescaling_max
        # for source_i in range(Nsrc):
        #     dst_wav_path = "./TempResults/Mixture{}/Ind_{}_src{}.wav".format(Nsrc,saveIndex,source_i)
        #     librosa.output.write_wav(dst_wav_path, mix_s_all[source_i][0] * scale, sr=hparams.sample_rate)
        # dst_wav_path = "./TempResults/Mixture{}/Ind_{}_mix.wav".format(Nsrc, saveIndex)
        # librosa.output.write_wav(dst_wav_path, mix[0] * scale, sr=hparams.sample_rate)

        # print the input SNR
        inputSNR = 20*np.log10(np.linalg.norm(mix_s_all[0][0])/np.linalg.norm(mix_s_all[1][0]))
        # print("The input SNR is {} dB".format(inputSNR))

        # Do the normalisation based on the mixture p0
        scale = 1 / np.max(np.abs(mix[0])) * hparams.rescaling_max
        mix *= scale
        for source_i in range(Nsrc):
            mix_s_all[source_i] *= scale


        ###########################
        MIX = STFTtensor(mix)

        # The normalised spectrum from the mixture [0, around 1]
        MIX_LP_norm = NormaliseLP(STFT2dB(MIX[0]))
        Input_angle = np.angle(MIX[0]) # the input angles of the mixture

        # plt.figure(100)
        # plt.pcolor(MIX_LP_norm)

        # The angle distance from the mixture [-pi pi]
        MIX_angle = angles_dist(MIX)

        Mix_angle_shift = np.tile(MIX_angle,[Nsrc,1,1])

        for source_i in range(Nsrc):
            temp = Mix_angle_shift[source_i]
            temp -= mus[source_i]
            temp = (temp + np.pi) % (2 * np.pi) - np.pi
            # nonlinearly convert the angle distance
            temp = np.power(temp, 2)
            temp = np.exp(-temp / hparams.dspatial_sigma2)
            Mix_angle_shift[source_i] = temp

        ###########################
        # get the spectrum of all source contributions
        specs = []
        for source_i in range(Nsrc):
            specs.append(np.abs(ApplySTFT(mix_s_all[source_i][0])))
        specs = np.asarray(specs)
        # Y = np.zeros_like(SS) # (N,W,T)

        # First get the groundtruth one-hot IBM labels
        Y = np.zeros(specs.shape[1::]+(specs.shape[0],)).astype(np.float32)  # (W,T,2)
        vals = np.argmax(specs, axis=0)  # (W,T)
        for source_i in range(Nsrc):
            t = np.zeros(Nsrc).astype(np.float32)
            t[source_i] = 1
            Y[vals == source_i] = t

        # Create mask for zeroing out gradients from silence components
        t = np.zeros(Nsrc).astype(np.float32)
        silenceIndex = MIX_LP_norm < hparams.silence_threshold
        Y[silenceIndex] = t

        # ideal ratio mask generation
        specs2 = np.power(specs, 2)
        IRM = (specs2 / np.sum(specs2, axis=0, keepdims=True)).transpose([1,2,0])
        IRM[silenceIndex] = t


        # plt.figure(80)
        # temp = np.reshape(MIX_LP_norm, (np.product(MIX_LP_norm.shape),))
        # plt.hist(temp, normed=True, bins=50)
        # plt.ylabel('Probability')
        #
        # plt.figure(81)
        # plt.subplot(211)
        # temp = np.reshape(specs[0], (np.product(specs[0].shape),))
        # plt.hist(temp, normed=True, bins=200)
        # plt.subplot(212)
        # temp = np.reshape(specs[1], (np.product(specs[1].shape),))
        # plt.hist(temp, normed=True, bins=200)
        #
        # plt.figure(82)
        # plt.subplot(211)
        # plt.pcolor(Y[:,:,0])
        # plt.subplot(212)
        # plt.pcolor(Y[:,:,1])
        #
        # plt.figure(83)
        # plt.subplot(211)
        # plt.pcolor(IRM[:, :, 0])
        # plt.subplot(212)
        # plt.pcolor(IRM[:, :, 1])





        #############################
        # Extract several blocks from the sequence
        # If you want to repeat your results, use this one
        if chooseIndexNormalised is None:  # consecutive blocks
            chooseIndex = np.arange(0, MIX_LP_norm.shape[1], batchLen, dtype=int)
            chooseIndex[-1] = MIX_LP_norm.shape[1] - batchLen
        else:
            N = int(round(MIX_LP_norm.shape[1] / batchLen))
            chooseIndex = (chooseIndexNormalised * (MIX_LP_norm.shape[1] - batchLen)).astype(int)
            if len(chooseIndex) >= N:
                chooseIndex = chooseIndex[:N]

        N = len(chooseIndex)
        # concatenate the feature as the input
        Index1 = (np.tile(range(0, batchLen), (N, 1))).T
        Index2 = np.tile(chooseIndex, (batchLen, 1))
        Index = Index1 + Index2


        ############################# DNN INPUT ############################
        MIX_LP_norm_blocks = MIX_LP_norm[:, Index]  # (W,T)--->(W,100,N)
        MIX_LP_norm_blocks = MIX_LP_norm_blocks.transpose([2, 1, 0])  # (W,100,N)--->(N,100,W)

        MIX_angle_blocks = MIX_angle[:, Index]
        MIX_angle_blocks = MIX_angle_blocks.transpose([2, 1, 0])

        MIX_angle_shift_all_blocks = np.zeros(MIX_angle_blocks.shape+(Nsrc,)).astype(np.float32)
        for source_i in range(Nsrc):
            temp = Mix_angle_shift[source_i]
            MIX_angle_shift_blocks = temp[:, Index]
            MIX_angle_shift_blocks = MIX_angle_shift_blocks.transpose([2, 1, 0])
            MIX_angle_shift_all_blocks[:,:,:,source_i] = MIX_angle_shift_blocks


        ############################# DNN OUTPUT ############################
        label_blocks = Y.transpose([0,2,1]) #(W,T,Nsrc)---> (W,Nsrc,T) one-hot IBM labels
        label_blocks = label_blocks[:, :, Index]  #(W,Nsrc,T)---> (W,Nsrc,100,N)
        label_blocks = label_blocks.transpose([3, 2, 0, 1]) # (W,Nsrc,100,N)--->(N,100,W,Nsrc)

        IRM_blocks = IRM.transpose([0, 2, 1])  # (W,T,Nsrc)---> (W,Nsrc,T)
        IRM_blocks = IRM_blocks[:, :, Index]  # (W,Nsrc,T)---> (W,Nsrc,100,N)
        IRM_blocks = IRM_blocks.transpose([3, 2, 0, 1]) # (W,Nsrc,100,N)--->(N,100,W,Nsrc)

        if featureFlag==0:
            # currentSubBatchIn = np.concatenate((np.expand_dims(MIX_LP_norm_blocks, -1), MIX_angle_shift_all_blocks), axis=-1)
            currentSubBatchIn = np.concatenate((MIX_LP_norm_blocks[:,:,:,None], MIX_angle_shift_all_blocks), axis=-1)
        elif featureFlag==1:
            currentSubBatchIn = np.stack([MIX_LP_norm_blocks, MIX_angle_blocks], axis=-1)
        elif featureFlag == 2:
            currentSubBatchIn = MIX_LP_norm_blocks[:, :, :, None]

        if outputFlag==0: # direct IBM labels and IRM
            currentSubBatchOut1 = label_blocks
            currentSubBatchOut2 = IRM_blocks
        elif outputFlag==1: # IBM and IRM followed by input spectrum for perceptual evaluation
            currentSubBatchOut1 = np.concatenate([label_blocks, MIX_LP_norm_blocks[:,:,:,None]], axis=-1)
            currentSubBatchOut2 = np.concatenate([IRM_blocks, MIX_LP_norm_blocks[:,:,:,None]], axis=-1)

        # index = 2
        # plt.figure(99)
        # plt.title('Input')
        # ax1 = plt.subplot(411)
        # im1 = ax1.pcolor(MIX_LP_norm_blocks[index].T)
        # plt.colorbar(im1)
        # ax2 = plt.subplot(412, sharex=ax1)
        # im2 = ax2.pcolor(MIX_angle_blocks[index].T)
        # plt.colorbar(im2)
        # ax3 = plt.subplot(413, sharex=ax1)
        # im3 = ax3.pcolor(MIX_angle_shift_all_blocks[index,:,:,0].T)
        # plt.colorbar(im3)
        # ax4 = plt.subplot(414, sharex=ax1)
        # im4 = ax4.pcolor(MIX_angle_shift_all_blocks[index,:,:,1].T)
        # plt.colorbar(im4)
        # plt.show()

        # index = 2
        # plt.figure(100)
        # plt.title('Output')
        # ax1 = plt.subplot(411)
        # im1 = ax1.pcolor(label_blocks[index, :, :, 0].T)
        # plt.colorbar(im1)
        # ax2 = plt.subplot(412, sharex=ax1)
        # im2 = ax2.pcolor(label_blocks[index, :, :, 1].T)
        # plt.colorbar(im2)
        # ax3 = plt.subplot(413, sharex=ax1)
        # im3 = ax3.pcolor(IRM_blocks[index, :, :, 0].T)
        # plt.colorbar(im3)
        # ax4 = plt.subplot(414, sharex=ax1)
        # im4 = ax4.pcolor(IRM_blocks[index, :, :, 1].T)
        # plt.colorbar(im4)
        # plt.show()

    except:
        print('Check your code for feature extraction!\n')
        currentSubBatchIn = np.array([0])
        currentSubBatchOut1 = np.array([0])
        currentSubBatchOut2 = np.array([0])
        chooseIndex = np.array([0])
        Input_angle = np.array([0])  # the input angles of the mixture
        MIX_angle_blocks = np.array([0])

    return (currentSubBatchIn, currentSubBatchOut1, currentSubBatchOut2, chooseIndex, [Input_angle, MIX_angle_blocks])



class dataGen:

    def __init__(self, seedNum = 123456789, verbose = False, verboseDebugTime = True):
        # self.seedNum = seedNum
        self.verbose = verbose
        self.verboseDebugTime = verboseDebugTime

        num_workers = min(cpu_count()-2,4) # parallel at most num_workers threads
        self.parallelN = num_workers
        self.executor = ProcessPoolExecutor(max_workers=4)


        # self.executorTrain = ProcessPoolExecutor(max_workers=2)
        # self.executorValid = ProcessPoolExecutor(max_workers=2)

        self.BATCH_SIZE_Train = 32 #32 # 128 #4096  # mini-batch size
        # self.batchSeqN_Train = self.BATCH_SIZE_Train

        self.BATCH_SIZE_Valid = 32  # 32 # 128 #4096  # mini-batch size
        # self.batchSeqN_Valid = self.BATCH_SIZE_Valid

        # self.EpochBatchNum = [200,20]
        self.EpochBatchNum = [200, 20] # [100*32*100*256/20050 ~ 1.1 hour, ~15 minutes]

        self.batchLen = 64 # 2^N for ease of model design
        self.halfNFFT = int(hparams.fft_size/2)  # instead of nfft/2+1, we keep only 2^N for ease of model design

        self.target_train_i = 0
        self.target_valid_i = 0
        self.target_test_i = 0

        # load the RIRs for mixture simulations
        self.rirs = sio.loadmat('./RIRs/Data/B_format_RIRs_12BB01_Alfredo_S3A_16k.mat')['rirs'].astype(np.float32)
        self.thetaNorm = sio.loadmat('./RIRs/Data/B_format_RIRs_SpatialFeatureNorm.mat')['ResultMean'].astype(np.float32)

        # load the training, validation, testing data from the TIMIT dataset
        with open('./TIMITtrainingSignalList.pkl', 'rb') as f:  # Python 2, open (...)  Python 3: open(..., 'rb')
            self.targetListTrain, self.targetListValid, self.targetListTest = pickle.load(f)

        random.seed(seedNum)
        shuffle(self.targetListTrain)
        shuffle(self.targetListValid)
        shuffle(self.targetListTest)


    # From a list of sequences, randomly chose two that are from different speakers
    def random2seq(self, sequenceList):
        name1 = random.choice(sequenceList)
        name2 = random.choice(sequenceList)
        spk1 = name1.split(os.sep)[-2]
        spk2 = name2.split(os.sep)[-2]
        while spk1==spk2:
            # ind1 = random.randint(0, len(sequenceList) - 1)
            # ind2 = random.randint(0, len(sequenceList) - 1)
            name1 = random.choice(sequenceList)
            name2 = random.choice(sequenceList)
            spk1 = name1.split(os.sep)[-2]
            spk2 = name2.split(os.sep)[-2]

        return [name1,name2]

    def randomNseq(self, sequenceList,N):

        spkList = []
        seqList = []
        for i in range(N):
            tempName = random.choice(sequenceList)
            tempSpk = tempName.split(os.sep)[-2]
            while tempSpk in spkList: # this speaker is already chosen
                tempName = random.choice(sequenceList)
                tempSpk = tempName.split(os.sep)[-2]
            spkList.append(tempSpk)
            seqList.append(tempName)
        return seqList

    def random2ang(self):
        azi1 = random.choice(hparams.angle_list)
        azi2 = random.choice(hparams.angle_list)
        diff = azi1-azi2
        diff = (diff + 180) % (360) - 180
        while np.abs(diff)<40:
            azi1 = random.choice(hparams.angle_list)
            azi2 = random.choice(hparams.angle_list)
            diff = azi1 - azi2
            diff = (diff + 180) % (360) - 180

        return [azi1, azi2]

    def randomNangOld(self, N):
        # we constraint the source numuber N = 2 or 3
        azi1 = 0 # one source is fixed at right front, i.e. 0 degree
        if N == 2:
            tempAzi2 = random.choice(hparams.angle_list[3:34])# they need to be at least 30 degrees apart
            return [azi1, tempAzi2]
        elif N==3: # the other source is at the other side
            tempAzi2 = random.choice(hparams.angle_list[3:19])
            tempAzi3 = random.choice(hparams.angle_list[19:34])
            diff = tempAzi3 - tempAzi2
            diff = (diff + 180) % (360) - 180
            while np.abs(diff) < 30:
                tempAzi3 = random.choice(hparams.angle_list[19:34])
                diff = tempAzi3 - tempAzi2
                diff = (diff + 180) % (360) - 180
            return [azi1, tempAzi2, tempAzi3]

    def randomNang(self, N):
        # we constraint the source numuber N = 2 or 3
        tempAziList = [random.choice(hparams.angle_list)]


        for n in range(1,N):
            nextFlag = True
            if N==2:
                threshold = 30
            elif N==3:
                threshold = 50
            while (nextFlag):
                tempAzi = random.choice(hparams.angle_list)
                diffAll = []
                for i in range(n):
                    diff = tempAzi-tempAziList[i]
                    diff = (diff + 180) % (360) - 180
                    diffAll.append(diff)
                if all(np.abs(diffAll) > threshold):
                    nextFlag = False

            tempAziList.append(tempAzi)

        return tempAziList




    def myDataGenerator(self, numSrc = 2, dataFlag=0, featureFlag = 0, outputFlag = 0):
        # numSrc, number of hidden sources in the mixtures
        # dataFlag, 0 for training, 1 for valid
        # featureFlag, 0: spectral+ numSrc shifted DOA spatial features
        #              1: spectral + DOA spatial features
        #              2: spectral features
        # outputFlag, 0: IBM from the DC head and IRM labels from the regression head
        #             1: IBM and IRM, followed by the mixture LP for perceptual weighting

        if numSrc not in [2,3]:
            print("There should only be 2 or 3 sources, while you input {}, it is reset to 2!".format(numSrc))
            numSrc = 2

        cin = [1+numSrc, 2, 1][featureFlag]
        cout = [numSrc,1+numSrc][outputFlag]
        batchSize = [self.BATCH_SIZE_Train, self.BATCH_SIZE_Valid][dataFlag]

        BatchDataIn = np.zeros((batchSize, self.batchLen, self.halfNFFT, cin), dtype='f')  # 100 x 512 x 3
        BatchDataOut1 = np.zeros((batchSize, self.batchLen, self.halfNFFT, cout), dtype='f') # IBM labels
        BatchDataOut2 = np.zeros((batchSize, self.batchLen, self.halfNFFT, cout), dtype='f') # IRM labels

        # save for the next batch
        BatchDataInNext = np.zeros((batchSize, self.batchLen, self.halfNFFT, cin), dtype='f')  # 100 x 512 x 3
        BatchDataOut1Next = np.zeros((batchSize, self.batchLen, self.halfNFFT, cout), dtype='f') # IBM labels
        BatchDataOut2Next = np.zeros((batchSize, self.batchLen, self.halfNFFT, cout), dtype='f') # IRM labels

        batchNumber = 0
        availableN = 0 # number of unused samples generated from the previous round of parallel executor

        while True:

            if self.verbose & (dataFlag==0):
                print('\n Now collect a mini batch {} for {}'.format(batchNumber+1,['training','validataion'][dataFlag]))

            # time_collect_start = datetime.datetime.now()
            NinCurrentBatch=0

            if availableN>0:
                # print('\n Grab {} samples from the previou round of parallel processing'.format(tempAvailableN))
                BatchDataIn[:availableN] = BatchDataInNext[:availableN]
                BatchDataOut1[:availableN] = BatchDataOut1Next[:availableN]
                BatchDataOut2[:availableN] = BatchDataOut2Next[:availableN]
                NinCurrentBatch += availableN
                availableN = 0 # clear the buffer

            while NinCurrentBatch < batchSize:

                # if self.verbose & (dataFlag==0):
                #     print('...{} in {} in batch {} for {}'.format(NinCurrentBatch,batchSize,batchNumber+1,['Training','Validation','Testing'][dataFlag]))

                if self.verbose & (dataFlag==0):
                    print('=================Parallel processing start')
                futuresthread = []
                tempResults = []
                for para_i in range(self.parallelN):  # parallel 4 processes

                    if self.verbose & (dataFlag==0):
                        print('---****---'*(para_i+1))


                    # randomly find two sequences from N speakers
                    chosenSeqs = self.randomNseq([self.targetListTrain,self.targetListValid,self.targetListTest][dataFlag],numSrc)

                    # randomly choose N angles to simulate the mixture
                    # for simplicity, we constrain one target at 0 azimuth degree, the either one or two at different sides
                    chosenAzis = self.randomNang(numSrc)
                    chosenAzis.sort()
                    # chosenAzis = self.randomNang(3)
                    # chosenAzis.sort(reverse=True)
                    chosenAziIndex = (np.asarray(chosenAzis)/10).astype(int)
                    RIRsUse = self.rirs[chosenAziIndex]
                    musUse = self.thetaNorm[chosenAziIndex]

                    # [azi1,azi2] = self.random2ang()
                    # RIRsUse = [self.rirs[int(azi1/10)],self.rirs[int(azi2/10)]]
                    # musUse = [self.thetaNorm[int(azi1/10)],self.thetaNorm[int(azi2/10)]]

                    # Each block takes (256/16000)*64 ~ 1 second long, so we randomly generate 10 numbers,
                    # i.e. to extract at most 10 blocks from each sequence
                    # since the average training data is only a few seconds long.
                    # chooseIndexNormalised = None
                    chooseIndexNormalised = np.asarray([random.random() for i in range(10)])  # extract N samples


                    if multiProcFlag:
                        futuresthread.append(self.executor.submit(
                            partial(ExtractFeatureOneMixture,
                                    featureFlag, outputFlag, chosenSeqs, RIRsUse, musUse, chooseIndexNormalised,
                                    self.batchLen)))
                    else:
                        tempResult = ExtractFeatureOneMixture(featureFlag, outputFlag, chosenSeqs, RIRsUse, musUse, chooseIndexNormalised, self.batchLen)
                        tempResults.append(tempResult)



                if self.verbose & (dataFlag==0):
                    print('=================Parallel processing end')

                # try:
                #     # tempResults = [future.result() for future in
                #     #                concurrent.futures.as_completed(futuresthread)]
                #     tempResults = [future.result() for future in concurrent.futures.as_completed(futuresthread,timeout=5)] # 5 seconds time limit
                # except:
                #     print('some thing wrong here\n')
                #     tempResults = []

                if multiProcFlag:
                    tempResults = [future.result() for future in futuresthread]

                if self.verbose & (dataFlag==0):
                    print('=================Parallel processing end{}'.format(len(tempResults)))

                if len(tempResults)>0:
                    for (currentSubBatchIn, currentSubBatchOut1, currentSubBatchOut2, _, _) in tempResults:
                        if currentSubBatchIn.size >1:
                            N = len(currentSubBatchIn)
                            useN = min(batchSize-NinCurrentBatch,N)
                            if self.verbose & (dataFlag==0):
                                print(useN)
                            if useN>0:
                                BatchDataIn[NinCurrentBatch:NinCurrentBatch + useN] = currentSubBatchIn[:useN]
                                BatchDataOut1[NinCurrentBatch:NinCurrentBatch + useN] = currentSubBatchOut1[:useN]
                                BatchDataOut2[NinCurrentBatch:NinCurrentBatch + useN] = currentSubBatchOut2[:useN]
                                NinCurrentBatch += useN

                            # too many data in the current parallel processing, save them for the next batch
                            reuseableN = min(N-useN,batchSize-availableN)
                            BatchDataInNext[availableN:availableN + reuseableN] = currentSubBatchIn[useN:useN+reuseableN]
                            BatchDataOut1Next[availableN:availableN + reuseableN] = currentSubBatchOut1[useN:useN+reuseableN]
                            BatchDataOut2Next[availableN:availableN + reuseableN] = currentSubBatchOut2[useN:useN+reuseableN]
                            availableN += reuseableN


                if self.verbose & (dataFlag==0):
                    print('...{} in {} in batch {} for {}'.format(NinCurrentBatch,batchSize,batchNumber+1,['Training','Validation','Testing'][dataFlag]))



            batchNumber += 1

            if self.verbose & (dataFlag==0):
                # time_collect_end = datetime.datetime.now()
                print("\t {} batch {} successuflly collected ".format(['Training','Validation','Testing'][dataFlag],batchNumber))
                # print("\t The total time to collect the current batch of data is ", time_collect_end - time_collect_start)

            yield [BatchDataIn], [BatchDataOut1, BatchDataOut2]





if __name__=="__main__":

    print('Test data generator')
    dg = dataGen(seedNum = 123456789, verbose = True, verboseDebugTime=False)

    # change yield to return to debug the generator
    featureFlag = 0
    outputFlag = 1
    # dg.myDataGenerator(numSrc=3, dataFlag=0, featureFlag=featureFlag, outputFlag=outputFlag) # comment out yield

    # change yield to return to debug the generator
    [aa],[bb1,bb2] = dg.myDataGenerator(numSrc=3,dataFlag=0,featureFlag=featureFlag,outputFlag=outputFlag)

    index = 4
    plt.figure(99)
    plt.title('Input')
    ax1 = plt.subplot(411)
    im1 = ax1.pcolor(aa[index,:,:,0].T)
    plt.colorbar(im1)
    ax2 = plt.subplot(412, sharex=ax1)
    im2 = ax2.pcolor(aa[index,:,:,1].T)
    plt.colorbar(im2)
    ax3 = plt.subplot(413, sharex=ax1)
    im3 = ax3.pcolor(aa[index,:,:,2].T)
    plt.colorbar(im3)
    ax4 = plt.subplot(414, sharex=ax1)
    im4 = ax4.pcolor(aa[index, :, :, 3].T)
    plt.colorbar(im4)

    plt.figure(100,figsize=(6, 8))
    plt.title('Output')
    ax1 = plt.subplot(611)
    im1 = ax1.pcolor(bb1[index, :, :, 0].T)
    plt.colorbar(im1)
    ax2 = plt.subplot(612, sharex=ax1)
    im2 = ax2.pcolor(bb1[index, :, :, 1].T)
    plt.colorbar(im2)
    ax3 = plt.subplot(613, sharex=ax1)
    im3 = ax3.pcolor(bb1[index, :, :, 2].T)
    plt.colorbar(im3)


    ax4 = plt.subplot(614, sharex=ax1)
    im4 = ax4.pcolor(bb2[index, :, :, 0].T)
    plt.colorbar(im4)
    ax5 = plt.subplot(615, sharex=ax1)
    im5 = ax5.pcolor(bb2[index, :, :, 1].T)
    plt.colorbar(im4)
    ax6 = plt.subplot(616, sharex=ax1)
    im6 = ax6.pcolor(bb2[index, :, :, 2].T)
    plt.colorbar(im6)
    plt.show()

    a = 0


