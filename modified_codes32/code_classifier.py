import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn import svm


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
eval_data = mnist.test.images # Returns np.array
labels = np.asarray(mnist.test.labels, dtype=np.int32)


codes = np.load(os.path.join('./results3/mnist_waedec_ncenc4_lbd10_nquant2/','code_data.npy'))
#codes = np.load(os.path.join('./results3/mnist_wae_nz8/','code_data.npy'))
print('code shape' +str(codes.shape))
X,Y =shuffle(codes,labels, random_state=2)
xtrain = X[0:8000 , :]
xtest = X[8000:-1 , :]
ytrain = Y[0:8000]
ytest = Y[8000:-1]
print(xtest.shape)


clf = svm.LinearSVC()
clf.fit(xtrain, ytrain)  
print("training is complete")
estimated_labels = clf.predict(xtest)
print(estimated_labels.shape)
accuracy = 100* np.sum(estimated_labels == ytest)/len(ytest)
print("Accuracy with linear classifier : %"+ str(accuracy) )


levels = 2**xtrain.shape[1]
number_of_codes = xtrain.shape[0]

tmp = np.logspace(0,xtrain.shape[1]-1, xtrain.shape[1], base = 2 )
xtrain = np.array(np.matmul(xtrain,tmp),dtype = int)
xtest = np.array(np.matmul(xtest,tmp),dtype = int)

#xtrain = np.squeeze(np.packbits(np.array(xtrain,dtype = bool),axis = 1))
#xtest = np.squeeze(np.packbits(np.array(xtest , dtype = bool) , axis = 1))
A = np.zeros([levels , 10])
for i in range (number_of_codes):
    selected_level = xtrain[i]
    selected_label = ytrain[i]
    A[selected_level , selected_label] += 1
R = np.argmax(A, axis = 1)
estimated_labels = np.zeros([xtest.shape[0]])
for j in range(xtest.shape[0]):
    estimated_labels[j] = R[xtest[j]]
accuracy = 100* np.sum(estimated_labels == ytest)/len(ytest)
print("Accuracy with: %"+ str(accuracy) )

