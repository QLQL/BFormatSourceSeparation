# import tensorflow as tf
# I should have also put the directories of date and models here, so you don't need to manually input them several times at other source code. 
import tensorflow.contrib.training as tftrain
import numpy as np

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
# hparams = tf.contrib.training.HParams(
hparams = tftrain.HParams(
    name="BformatSS",
    # Audio:
    sample_rate=16000,
    fft_size=1024,
    fre_bin_num = 512,
    hop_size=256,
    overlap = 768,
    angle_list = np.arange(0,360,10),
    rescaling_max = 0.2,

    min_level_db = -125,
    norm_scale_db = 80,
    silence_threshold = 0.05, # silence threshold for normalised LP features

    dspatial_sigma2 = 3, # 2*sigma^2 for spatial conversion of the angle distance


)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
