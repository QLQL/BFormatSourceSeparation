# -*- coding: utf-8 -*-
"""
QL
"""

from keras import backend as K
from keras import optimizers
from keras import callbacks
from keras.utils import plot_model
from keras.models import load_model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import re
import tensorflow as tf

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

K.set_learning_phase(1) #set learning phase
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
import keras
print('The keras version is ',keras.__version__)
print('The TF version is ',tf.__version__)


# setting the hyper parameters
# import os
import argparse

# setting the hyper parameters
parser = argparse.ArgumentParser(description="Encoder")

parser.add_argument('--continueToTrainFlag', default=False, type=bool)
parser.add_argument('--srcNum', default=2, type=int, help="Number of sources")
parser.add_argument('--featureFlag', default=1, type=int)  # featureFlag (0 spectrum shift1 shift2, 1 spectrum spat, 2 spectrum)
parser.add_argument('--outputFlag', default=1, type=int)  # outputFlag (0 mse, 1 perceptual weighting)

parser.add_argument('--newSeedNum', default=418571351248, type=int)  # new seeds for shuffle data when continue training 418571351248
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
parser.add_argument('--save_dir', default='./TrainedModels')
# parser.add_argument('--save_dir', default='./DeleteLater')
parser.add_argument('--is_training', default=1, type=int)
parser.add_argument('-w', '--weights', default=None, help="The path of the saved weights. Should be specified when testing")
parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.98, type=float, help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
args = parser.parse_args()
print(args)


############################################################
from GenerateData import dataGen
# dg = dataGenBig()

if args.continueToTrainFlag:
    dg = dataGen(seedNum=args.newSeedNum, verbose=False, verboseDebugTime=False)
    # dg = dataGenBig(seedNum=123456789, verbose=False, verboseDebugTime=False)
else:
    dg = dataGen(seedNum = 123456789, verbose = False, verboseDebugTime=False)
# [aa,bb] = dg.myDataGenerator(0)# change yield to return to debug the generator


if args.outputFlag==0:
    from LossFunctions import my_loss_DC_Nsrc as LossA, my_loss_MSE_Nsrc as LossB
elif args.outputFlag==1:
    from LossFunctions import weight_loss_DC_Nsrc as LossA, weight_loss_MSE_Nsrc as LossB
    # if args.srcNum==2:
    #     from LossFunctions import weight_loss_DC as Loss1, weight_loss_MSE as Loss2
    # else:
    #     from LossFunctions import weight_loss_DC_3 as Loss1, weight_loss_MSE_3 as Loss2

from GenerateModels import ChimeraNet as GenerateModel
tag = 'Feature{}Nsrc{}{}'.format(['Proposed','SpecSpat','SpecOnly'][args.featureFlag],args.srcNum,['','Wei'][args.outputFlag])

print(tag)
if args.save_dir is None:
    save_dir = './Models/{}'.format(tag)
else:
    save_dir = args.save_dir
modelDirectory = '{}/{}'.format(save_dir,tag)


# Load the model
train_model = GenerateModel(Nsrc = args.srcNum, inputFeature = args.featureFlag)
print(train_model.summary())
# plot_model(train_model, to_file='Model.png')



initial_epoch = 0
if args.continueToTrainFlag:
    savedModelDirectory = '{}/{}'.format('/Directory2TrainedModels',tag)
    try:
        savedModels = [filename for path, subdirs, files in os.walk(savedModelDirectory)
                          for filename in files if filename.endswith(".h5")]

        if len(savedModels)>0:
            index = []
            for saveModelName in savedModels:
                temp = re.findall(r'(?<=model-)(\d+).h5', saveModelName)
                if len(temp)>0:
                    index.append(int(temp[0]))
            # index = [int(re.findall(r'(?<=weights-)(\d+).h5', saveModelName)[0]) for saveModelName in savedModels
            #          if re.findall(r'(?<=weights-)(\d+).h5', saveModelName) is not None]
            if len(index)>0:
                initial_epoch = max(index)
            if args.continueToTrainFlag:
            # if initial_epoch >= 1 and args.continueToTrainFlag:
                # since the optimiser status is not saved, save-weight only is not applicable for retraining
                # train_model.load_weights("{}/weights-{}.h5".format(save_dir,initial_epoch))
                del train_model
                # from keras.utils.generic_utils import get_custom_objects
                # get_custom_objects().update({"my_loss": customLoss})
                # train_model = load_model("{}/model-{:02d}.h5".format(save_dir, initial_epoch))
                #
                # train_model = load_model("{}/model-{:02d}.h5".format(save_dir, initial_epoch),
                #                          custom_objects={'my_loss': customLoss})
                # modelName = "{}/model-{:02d}.h5".format(modelDirectory, initial_epoch)
                modelName = "{}/{}".format(savedModelDirectory, savedModels[-1])
                if args.outputFlag==0:
                    train_model = load_model(modelName,
                                             custom_objects={'my_loss_DC_Nsrc_Ref': LossA(Nsrc=args.srcNum),
                                                             'my_loss_MSE_Nsrc_Ref': LossB(Nsrc=args.srcNum)})
                elif args.outputFlag==1:
                    train_model = load_model(modelName,
                                             custom_objects={'weight_loss_DC_Nsrc_Ref': LossA(Nsrc=args.srcNum),
                                                             'weight_loss_MSE_Nsrc_Ref': LossB(Nsrc=args.srcNum)})
                print('\nA pre-trained model {} has been loaded, continue training\n'.format(modelName))
                print(train_model.summary())
                # Initial_lr = Initial_lr*(lr_decay**initial_epoch)
            else:
                initial_epoch = 0
    except:
        train_model = GenerateModel()
        print('\nCould not find a pre-trained model, start a refresh training\n')



if not os.path.exists(modelDirectory):
    os.makedirs(modelDirectory)


# compile the model
# if args.continueToTrainFlag and initial_epoch>=1:
if args.continueToTrainFlag:
    # will also take care of compiling the model using the saved training configuration
    # (unless the model was never compiled in the first place).
    pass
else:
    # Note the key words loss and losses are very different! made a stupid error here and wasted a whole day
    # train_model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[customLoss], metrics=[customLoss])
    # train_model.compile(optimizer=optimizers.Adam(lr=args.lr),
    #                     loss={'embedding': Loss1, 'mask': Loss2}, metrics={'embedding': Loss1, 'mask': Loss2})
    train_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                        loss={'embedding': LossA(Nsrc=args.srcNum), 'mask': LossB(Nsrc=args.srcNum)},
                        metrics={'embedding': LossA(Nsrc=args.srcNum), 'mask': LossB(Nsrc=args.srcNum)})


log = callbacks.CSVLogger(modelDirectory + '/log.csv')
# tensorboard does not work with data generator, you have to write your summaries :(. Not bothered....
# tb = callbacks.TensorBoard(log_dir=modelDirectory + '/tensorboard-logs', batch_size=dg.BATCH_SIZE_Train, histogram_freq=args.debug)
# checkpoint = callbacks.ModelCheckpoint(modelDirectory + '/model-{epoch:02d}.h5', monitor='val_loss',
#                                        save_best_only=True, save_weights_only=False, verbose=1, period=10)
checkpoint = callbacks.ModelCheckpoint(modelDirectory + '/modelsave.h5', monitor='val_loss',
                                       save_best_only=True, save_weights_only=False, verbose=1, period=2)

lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: max(args.lr * (args.lr_decay ** epoch),0.000001))
# lr_decay = callbacks.ReduceLROnPlateau(monitor='val_loss',
#                                       factor=0.8,
#                                       patience=3, # monitor every # epochs
#                                       verbose=1,
#                                       mode='auto',
#                                       min_delta=0.01,
#                                       cooldown=5, # add an extra # epochs for next monitor
#                                       min_lr=0.000001)

train_model.fit_generator(generator=dg.myDataGenerator(numSrc = args.srcNum, dataFlag=0, featureFlag=args.featureFlag, outputFlag=args.outputFlag), # 0 training batch, 1 validation batch
                    steps_per_epoch=dg.EpochBatchNum[0],
                    epochs=args.epochs,
                    initial_epoch = initial_epoch,
                    validation_data=dg.myDataGenerator(numSrc = args.srcNum, dataFlag=1,featureFlag=args.featureFlag, outputFlag=args.outputFlag),
                    validation_steps=dg.EpochBatchNum[1],
                    callbacks=[log, checkpoint, lr_decay])
# End: Training with data augmentation -----------------------------------------------------------------------#
#
# train_model.save_weights(save_dir + '/trained_model_weights.h5') # saves only the weight
train_model.save(modelDirectory + '/trained_model.h5') # the weight, the architecture, the optimiser status, for re-training
print('Trained model saved to \'%s/trained_model.h5\'' % modelDirectory)