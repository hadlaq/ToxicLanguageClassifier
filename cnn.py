import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, Concatenate, Flatten
from data_utils import *
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import random
import decimal


def get_coefs(word,*arr): 
	return word, np.asarray(arr, dtype='float32')
	
EMBEDDING_FILE= './data/glove.6B.100d.txt'
TRAIN_DATA_FILE= './data/train.csv'
TEST_DATA_FILE= './data/test.csv'

X_train, Y_train = readRankOneTrainData()
X_test, Y_test, actualComments, commentsLabels = readRankOneTestData()

hyperParamXTrain = X_train[:10000]
hyperParamYTrain = Y_train[:10000]

hyperParamXTest = X_test[:5000]
hyperParamYTest = Y_test[:5000]


bestAuc = 0
batchSizes = [16,32,64,128,256,512]
filter_sizes = list(range(3,12))
optimizers = ['adam', 'rmsprop']

bestBatchSize = 32
bestFilterSize = 5
bestOptimizer = 'adam'
'''
embed_size = 100
max_features = 40000 
maxlen = 100
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
list_tokenized_train = tokenizer.texts_to_sequences(hyperParamXTrain)
list_tokenized_test = tokenizer.texts_to_sequences(hyperParamXTest)
X_Train = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_Test = pad_sequences(list_tokenized_test, maxlen=maxlen)


embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

print(np.shape(embedding_matrix))
max_features = np.shape(embedding_matrix)[0]
for word, i in word_index.items():
    if i >= max_features: 
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector

for currIter in range(100):

    currBatchSize = random.choice(batchSizes)
    currFilterSize = random.choice(filter_sizes)
    currOpt = random.choice(optimizers)

    print("Using batch size " + str(currBatchSize) + ", filter size " + str(currFilterSize) + ", and optimizer " + currOpt)
    embedding_layer = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = True)
    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    covs = []
        

    #filter size 3
    for j in range(2, currFilterSize+1):

        convs.append(Conv1D(filters=128,kernel_size=j,activation='relu')(embedded_sequences))
        covs.append(MaxPooling1D(5)(convs[-1]))


    l_merge = Concatenate(axis=1)(convs)
    l_cov1=  Conv1D(128, 5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_flat = Flatten()(l_pool1)
    output = Dense(6, activation="sigmoid")(l_flat)    
    model = Model(sequence_input, output)


    model.compile(loss='binary_crossentropy',
                  optimizer=currOpt,
                  metrics=['accuracy'])

    final_train_X, final_test_X, final_train_y, final_test_y = train_test_split(X_Train, hyperParamYTrain, train_size= 0.9)
    model.fit(final_train_X, final_train_y, batch_size=currBatchSize,verbose = 1, epochs=1, validation_data=(final_test_X, final_test_y))
    # evaluate the model
    Y_pred = model.predict(X_Test)
    names = ["Insult", "Toxic", "Identity Hate", "Severe Toxic", "Obscene", "Threat"]

    #writeErrorsToFile("cnn_errors.txt", Y_pred, Y_test, actualComments, commentsLabels)

    avgAuc = 0
    for i in range(len(names)):
        y_pred_keras = Y_pred[:, i]
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(hyperParamYTest[:, i], y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
        avgAuc += auc_keras
    avgAuc /= (1.0*len(names))
    if avgAuc > bestAuc:
        bestBatchSize = currBatchSize
        bestFilterSize = currFilterSize
        bestOpt = currOpt
    print("Iteration " + str(currIter) + " found auc of " + str(avgAuc))
'''
##### ACTUAL NEW FINAL TEST (ABOVE IS HYPERPARAMETER TUNING) HERE #############


embed_size = 100
max_features = 40000 
maxlen = 100
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
list_tokenized_train = tokenizer.texts_to_sequences(X_train)
list_tokenized_test = tokenizer.texts_to_sequences(X_test)
X_Train = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_Test = pad_sequences(list_tokenized_test, maxlen=maxlen)

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))


print("BEST PARAMS: Using batch size " + str(bestBatchSize) + ", filter size " + str(bestFilterSize) + ", and optimizer " + bestOptimizer)
embedding_layer = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = True)
sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
convs = []
covs = []
    

#filter size 3
for j in range(3, bestFilterSize+1):

    convs.append(Conv1D(filters=128,kernel_size=j,activation='relu')(embedded_sequences))
    covs.append(MaxPooling1D(5)(convs[-1]))


l_merge = Concatenate(axis=1)(convs)
l_cov1=  Conv1D(128, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_flat = Flatten()(l_pool1)
output = Dense(6, activation="sigmoid")(l_flat)    
model = Model(sequence_input, output)


model.compile(loss='binary_crossentropy',
              optimizer=bestOptimizer,
              metrics=['accuracy'])

final_train_X, final_test_X, final_train_y, final_test_y = train_test_split(X_Train, Y_train, train_size= 0.95)
model.fit(final_train_X, final_train_y, batch_size=bestBatchSize,verbose = 1, epochs=1, validation_data=(final_test_X, final_test_y))
# evaluate the model
Y_pred = model.predict(X_Test)
names = ["Insult", "Toxic", "Identity Hate", "Severe Toxic", "Obscene", "Threat"]

writeErrorsToFile("cnn_errors.txt", Y_pred, Y_test, actualComments, commentsLabels)


for i in range(len(names)):
    y_pred_keras = Y_pred[:, i]
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test[:, i], y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    print(names[i], ":", auc_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='{} AUC = {:.3f})'.format(names[i], auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('CNN ROC, hyperparam')
plt.legend(loc='best')
plt.savefig("hyper_param_cnn.png")

