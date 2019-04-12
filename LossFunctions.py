from keras import backend as K

def nonlinearTFweight(x):
    """
    x: each item in the range of [0,1]
    """
    # return K.pow(x, 2)
    return K.sqrt(x)

# https://github.com/keras-team/keras/issues/2121
# customerised loss with other arguments other than (y_true, y_pred)

def my_loss_DC_Nsrc(Nsrc):

    def my_loss_DC_Nsrc_Ref(Y, V): # y_true, y_pred
        # Y size [BATCH_SIZE, FrameN, FrequencyN, Nsrc]
        # V size [BATCH_SIZE, FrameN, FrequencyN, EMBEDDINGS_DIMENSION] normalised

        def norm(tensor):
            square_tensor = K.square(tensor)
            frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
            return frobenius_norm2

        def dot(x, y):
            return K.batch_dot(x, y, axes=(2, 1))

        def T(x):
            return K.permute_dimensions(x, [0, 2, 1])

        EMBEDDINGS_DIMENSION = 20  # change this later
        Nframe = 64
        FrequencyN = 512

        Y = K.reshape(Y, [-1, Nframe * FrequencyN, Nsrc])

        silence_mask = K.sum(Y, axis=2, keepdims=True)
        V = K.reshape(V, [-1, Nframe * FrequencyN, EMBEDDINGS_DIMENSION])
        V = silence_mask * V

        # aa = dot(T(V), V)

        lossSum = norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))

        # N2 = (Nframe * FrequencyN) ** 2
        N2 = K.square(K.sum(Y, axis=(1, 2)))

        return lossSum / N2

    return my_loss_DC_Nsrc_Ref


def my_loss_MSE_Nsrc(Nsrc):

    def my_loss_MSE_Nsrc_Ref(Y, V): # y_true, y_pred
        # Y size [BATCH_SIZE, FrameN, FrequencyN, Nsrc] Groundtruth IRM with silence mask [0 0]
        # V size [BATCH_SIZE, FrameN, FrequencyN, Nsrc] normalised

        def norm(tensor):
            square_tensor = K.square(tensor)
            frobenius_norm2 = K.sum(square_tensor, axis=(1, 2, 3))
            return frobenius_norm2

        diff = Y - V
        silence_mask = K.sum(Y, axis=3, keepdims=True)
        diff = silence_mask * diff

        lossSum = norm(diff)

        N = K.sum(Y, axis=(1, 2, 3))

        return lossSum / N

    return my_loss_MSE_Nsrc_Ref




# https://github.com/keras-team/keras/issues/2121
# customerised loss with other arguments other than (y_true, y_pred)
def weight_loss_DC_Nsrc(Nsrc):
    # the embedded function name will be passed to the saved model, so make it unique to avoid confusions
    def weight_loss_DC_Nsrc_Ref(YpM, V): # y_true, y_pred
        # YpM size [BATCH_SIZE, FrameN, FrequencyN, Nsrc+1] for Nsrc signal situations, the third one is the normalised mixture LP
        # V size [BATCH_SIZE, FrameN, FrequencyN, EMBEDDINGS_DIMENSION] normalised

        def norm(tensor):
            square_tensor = K.square(tensor)
            frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
            return frobenius_norm2

        def dot(x, y):
            return K.batch_dot(x, y, axes=(2, 1))

        def T(x):
            return K.permute_dimensions(x, [0, 2, 1])

        EMBEDDINGS_DIMENSION = 20  # change this later
        Nframe = 64
        FrequencyN = 512

        Y = YpM[:, :, :, 0:Nsrc]
        Mixture = YpM[:, :, :, Nsrc]

        weight = nonlinearTFweight(Mixture)
        silence_mask = K.sum(Y, axis=3)
        weight = silence_mask * weight
        weight = K.reshape(weight, [-1, Nframe * FrequencyN, 1])


        Y = K.reshape(Y, [-1, Nframe * FrequencyN, Nsrc])
        V = K.reshape(V, [-1, Nframe * FrequencyN, EMBEDDINGS_DIMENSION])

        Y = weight * Y
        V = weight * V

        # aa = dot(T(V), V)

        lossSum = norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))

        # N2 = (Nframe * FrequencyN) ** 2
        # N2 = K.square(K.sum(Y,axis=(1,2)))

        De = norm(dot(T(weight), weight))

        return lossSum / De

    return weight_loss_DC_Nsrc_Ref


def weight_loss_MSE_Nsrc(Nsrc):
    # the embedded function name will be passed to the saved model, so make it unique to avoid confusions
    def weight_loss_MSE_Nsrc_Ref(YpM, V):  # y_true, y_pred
        # YpM size [BATCH_SIZE, FrameN, FrequencyN, Nsrc+1] Groundtruth IRM with silence mask [0 0] or [0 0 0]
        # V size [BATCH_SIZE, FrameN, FrequencyN, Nsrc] estimated IRM

        def norm(tensor):
            square_tensor = K.square(tensor)
            frobenius_norm2 = K.sum(square_tensor, axis=(1, 2, 3))
            return frobenius_norm2

        Y = YpM[:, :, :, 0:Nsrc]
        Mixture = YpM[:, :, :, Nsrc:Nsrc + 1]  # use Nsrc:Nsrc+1 instead of Nsrc to keep the last dimension

        weight = nonlinearTFweight(Mixture)
        silence_mask = K.sum(Y, axis=3, keepdims=True)
        weight = silence_mask*weight

        diff = Y - V
        diff = weight * diff

        lossSum = norm(diff)

        De = norm(weight) * Nsrc

        return lossSum / De

    return weight_loss_MSE_Nsrc_Ref






# def my_loss_DC(Y, V):
#     # Y size [BATCH_SIZE, FrameN, FrequencyN, 2] for two signal situations
#     # V size [BATCH_SIZE, FrameN, FrequencyN, EMBEDDINGS_DIMENSION] normalised
#
#     def norm(tensor):
#         square_tensor = K.square(tensor)
#         frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
#         return frobenius_norm2
#
#     def dot(x, y):
#         return K.batch_dot(x, y, axes=(2, 1))
#
#     def T(x):
#         return K.permute_dimensions(x, [0, 2, 1])
#
#     MAX_MIX = 2  # either target or inteference
#     EMBEDDINGS_DIMENSION = 20  # change this later
#     Nframe = 64
#     FrequencyN = 512
#
#     Y = K.reshape(Y, [-1, Nframe * FrequencyN, MAX_MIX])
#
#     silence_mask = K.sum(Y, axis=2, keepdims=True)
#     V = K.reshape(V, [-1, Nframe * FrequencyN, EMBEDDINGS_DIMENSION])
#     V = silence_mask * V
#
#     # aa = dot(T(V), V)
#
#     lossSum = norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))
#
#     # N2 = (Nframe * FrequencyN) ** 2
#     N2 = K.square(K.sum(Y,axis=(1,2)))
#
#     return lossSum / N2
#
#
# def my_loss_MSE(Y, V):
#     # Y size [BATCH_SIZE, FrameN, FrequencyN, 2] Groundtruth IRM with silence mask [0 0]
#     # V size [BATCH_SIZE, FrameN, FrequencyN, 2] normalised
#
#     def norm(tensor):
#         square_tensor = K.square(tensor)
#         frobenius_norm2 = K.sum(square_tensor, axis=(1, 2, 3))
#         return frobenius_norm2
#
#
#     # MAX_MIX = 2  # either target or inteference
#     # Nframe = 64
#     # FrequencyN = 512
#
#     diff = Y-V
#     silence_mask = K.sum(Y, axis=3, keepdims=True)
#     diff = silence_mask * diff
#
#     lossSum = norm(diff)
#
#
#     N = K.sum(Y,axis=(1,2,3))
#
#     return lossSum / N




# def weight_loss_DC(YpM, V):
#     # YpM size [BATCH_SIZE, FrameN, FrequencyN, 3] for two signal situations, the third one is the normalised mixture LP
#     # V size [BATCH_SIZE, FrameN, FrequencyN, EMBEDDINGS_DIMENSION] normalised
#
#     def norm(tensor):
#         square_tensor = K.square(tensor)
#         frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
#         return frobenius_norm2
#
#     def dot(x, y):
#         return K.batch_dot(x, y, axes=(2, 1))
#
#     def T(x):
#         return K.permute_dimensions(x, [0, 2, 1])
#
#     MAX_MIX = 2  # either target or inteference
#     EMBEDDINGS_DIMENSION = 20  # change this later
#     Nframe = 64
#     FrequencyN = 512
#
#     Y = YpM[:,:,:,0:2]
#     Mixture = YpM[:,:,:,2]
#
#     weight = nonlinearTFweight(Mixture)
#     weight= K.reshape(weight, [-1, Nframe * FrequencyN, 1])
#
#     Y = K.reshape(Y, [-1, Nframe * FrequencyN, MAX_MIX])
#
#     # silence_mask = K.sum(Y, axis=2, keepdims=True)
#     # weight = silence_mask*weight
#
#     V = K.reshape(V, [-1, Nframe * FrequencyN, EMBEDDINGS_DIMENSION])
#     # V = silence_mask * V
#
#     Y = weight*Y
#     V = weight*V
#
#
#
#     # aa = dot(T(V), V)
#
#     lossSum = norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))
#
#     # N2 = (Nframe * FrequencyN) ** 2
#     # N2 = K.square(K.sum(Y,axis=(1,2)))
#
#     N2 = norm(dot(T(weight), weight))
#
#     return lossSum / N2
#
#
# def weight_loss_MSE(YpM, V):
#     # YpM size [BATCH_SIZE, FrameN, FrequencyN, 3] Groundtruth IRM with silence mask [0 0]
#     # V size [BATCH_SIZE, FrameN, FrequencyN, 2] normalised
#
#     def norm(tensor):
#         square_tensor = K.square(tensor)
#         frobenius_norm2 = K.sum(square_tensor, axis=(1, 2, 3))
#         return frobenius_norm2
#
#     Y = YpM[:, :, :, 0:2]
#     Mixture = YpM[:, :, :, 2:3]
#
#     weight = nonlinearTFweight(Mixture)
#
#     # MAX_MIX = 2  # either target or inteference
#     # Nframe = 64
#     # FrequencyN = 512
#
#     diff = Y-V
#
#     # silence_mask = K.sum(Y, axis=3, keepdims=True)
#     # weight = silence_mask*weight
#
#     diff = weight * diff
#
#     lossSum = norm(diff)
#
#
#     # N = K.sum(Y,axis=(1,2,3))
#     N = K.sum(weight, axis=(1, 2))
#
#     return lossSum / N
#
#
# def weight_loss_DC_3(YpM, V):
#     # YpM size [BATCH_SIZE, FrameN, FrequencyN, Nsrc+1] for Nsrc signal situations, the third one is the normalised mixture LP
#     # V size [BATCH_SIZE, FrameN, FrequencyN, EMBEDDINGS_DIMENSION] normalised
#
#     def norm(tensor):
#         square_tensor = K.square(tensor)
#         frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
#         return frobenius_norm2
#
#     def dot(x, y):
#         return K.batch_dot(x, y, axes=(2, 1))
#
#     def T(x):
#         return K.permute_dimensions(x, [0, 2, 1])
#
#     Nsrc = 3
#     EMBEDDINGS_DIMENSION = 20  # change this later
#     Nframe = 64
#     FrequencyN = 512
#
#     Y = YpM[:,:,:,0:Nsrc]
#     Mixture = YpM[:,:,:,Nsrc]
#
#     weight = nonlinearTFweight(Mixture)
#     weight= K.reshape(weight, [-1, Nframe * FrequencyN, 1])
#
#     Y = K.reshape(Y, [-1, Nframe * FrequencyN, Nsrc])
#
#     # silence_mask = K.sum(Y, axis=2, keepdims=True)
#     # weight = silence_mask*weight
#
#     V = K.reshape(V, [-1, Nframe * FrequencyN, EMBEDDINGS_DIMENSION])
#     # V = silence_mask * V
#
#     Y = weight*Y
#     V = weight*V
#
#
#
#     # aa = dot(T(V), V)
#
#     lossSum = norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))
#
#     # N2 = (Nframe * FrequencyN) ** 2
#     # N2 = K.square(K.sum(Y,axis=(1,2)))
#
#     N2 = norm(dot(T(weight), weight))
#
#     return lossSum / N2
#
#
# def weight_loss_MSE_3(YpM, V):
#     # YpM size [BATCH_SIZE, FrameN, FrequencyN, Nsrc+1] Groundtruth IRM with silence mask [0 0] or [0 0 0]
#     # V size [BATCH_SIZE, FrameN, FrequencyN, Nsrc] estimated IRM
#
#     def norm(tensor):
#         square_tensor = K.square(tensor)
#         frobenius_norm2 = K.sum(square_tensor, axis=(1, 2, 3))
#         return frobenius_norm2
#
#     Nsrc = 3  #
#     Y = YpM[:, :, :, 0:Nsrc]
#     Mixture = YpM[:, :, :, Nsrc:Nsrc+1] # use Nsrc:Nsrc+1 instead of Nsrc to keep the last dimension
#
#     weight = nonlinearTFweight(Mixture)
#
#     diff = Y-V
#
#     # silence_mask = K.sum(Y, axis=3, keepdims=True)
#     # weight = silence_mask*weight
#
#     diff = weight * diff
#
#     lossSum = norm(diff)
#
#
#     # N = K.sum(Y,axis=(1,2,3))
#     N = K.sum(weight, axis=(1, 2))
#
#     return lossSum / N
