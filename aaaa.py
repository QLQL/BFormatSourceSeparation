#
# # import numpy as np
# # label = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,])
# # best_path = [2,0,1]
# #
# # def simplyMap(x, best_path):
# #     return best_path[x]
# #
# # vfunc = np.vectorize(simplyMap)
# # #https://stackoverflow.com/questions/34065412/np-vectorize-giving-me-indexerror-invalid-index-to-scalar-variable
# # vfunc.excluded.add(1)
# # newlabel = vfunc(label,best_path)
# # a = 0
#
#
# ###########
# # The following is to prove how the weighted loss DC works
# import numpy as np
# N = 100
#
# #########################DC
# A = np.abs(np.random.rand(N,10))
# B = np.abs(np.random.rand(N,2))
#
#
# A_M = np.matmul(A,A.T)
# B_M = np.matmul(B,B.T)
#
# Diff = np.sum(np.square(A_M-B_M))
#
# De = np.sum(np.ones(shape=(N,N)))
# Diff /= De
#
# Diff2 = np.sum(np.square(np.matmul(A.T, A))) \
#         - np.sum(np.square(np.matmul(A.T, B))) * 2 \
#         + np.sum(np.square(np.matmul(B.T, B)))
#
# Diff2 /= De
#
#
# print('Normalised loss are---{} and {}'.format(Diff, Diff2))
#
#
# weight = np.random.uniform(size=(N,1))
# # weight = np.ones(shape=(100,1))*0.5
#
# A_w = A*weight
# B_w = B*weight
#
# A_w_M = np.matmul(A_w,A_w.T)
# B_w_M = np.matmul(B_w,B_w.T)
#
# De_w = np.sum(np.square(np.matmul(weight,weight.T)))
# De_w2 = np.sum(np.square(np.matmul(weight.T,weight)))
#
# Diff_w = np.sum(np.square(A_w_M-B_w_M))
#
# Diff_w /= De_w
#
# Diff2_w = np.sum(np.square(np.matmul(A_w.T, A_w))) \
#         - np.sum(np.square(np.matmul(A_w.T, B_w))) * 2 \
#         + np.sum(np.square(np.matmul(B_w.T, B_w)))
#
# Diff2_w /= De_w
#
# print('Normalised loss are---{} and {}'.format(Diff_w, Diff2_w))
#
#
#
# ##########################MSE
# C = np.abs(np.random.rand(N,2))
# Diff_mse = np.sum(np.square(B-C))
# De = 2*N
# Diff_mse /= De
#
# B_w = B*weight
# C_w = C*weight
#
# Diff2_mse_w = np.sum(np.square(B_w-C_w))
# De_w = np.sum(np.square(weight))*2
# Diff2_mse_w /= De_w
#
# print('Normalised loss are---{} and {}'.format(Diff_mse, Diff2_mse_w))
#
#
# a = 0


# import random
# import numpy as np
#
# def randomNang(N):
#         # we constraint the source numuber N = 2 or 3
#         angle_list = np.arange(0,360,10).tolist()
#         tempAziList = [random.choice(angle_list)]
#
#         for n in range(1,N):
#             nextFlag = True
#             if N==2:
#                 threshold=30
#             elif N==3:
#                 threshold = 50
#             while (nextFlag):
#                 tempAzi = random.choice(angle_list)
#                 diffAll = []
#                 for i in range(n):
#                     diff = tempAzi-tempAziList[i]
#                     diff = (diff + 180) % (360) - 180
#                     diffAll.append(diff)
#                 if all(np.abs(diffAll) > threshold):
#                     print(np.abs(diffAll))
#                     nextFlag = False
#
#             tempAziList.append(tempAzi)
#
#         return tempAziList
#
# for i in range(100):
#         # a = randomNang(2)
#         b = randomNang(3)
#         print(b)


# import numpy as np
# a = np.array([0])
# print(a.size) # equivalent to np.prod(a.shape)


import pickle
# Getting back the objects:
with open('TIMITtestingSignalList.pkl', 'rb') as f:
    chosenSeqsList, chosenAngsList, chosenSeqsList3, chosenAngsList3, rirs, thetaNorm, batchLen = pickle.load(f)
N = len(chosenSeqsList)

with open('TestingSignalList.txt', 'w') as f:
    for item in chosenSeqsList:
        f.write("%s\n" % item)
    for item in chosenSeqsList3:
        f.write("%s\n" % item)

file_name_save = "./TestingSignalList.mat"
import scipy.io as sio
sio.savemat(file_name_save, mdict={'chosenAngsList':chosenAngsList,'chosenAngsList3':chosenAngsList3, 'rirs':rirs, 'thetaNorm':thetaNorm, 'batchLen':batchLen})