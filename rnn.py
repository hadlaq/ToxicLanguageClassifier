from keras.models import Sequential
from keras.layers import Dense, LSTM
from data_utils import *
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

# create model
'''
insult_dict = dict()
toxic_dict = dict()
identity_hate_dict = dict()
severe_toxic_dict = dict()
obscene_dict = dict()
threat_dict = dict()
neither_dict = dict()
for key in ['tp', 'fp', 'fn', 'tn']:
	for dictionary in [insult_dict, toxic_dict, identity_hate_dict, severe_toxic_dict, obscene_dict, threat_dict, neither_dict]:
		dictionary[key] = 0'''


X_train, Y_train = read_expanded_train_data_glove()
X_test, Y_test = read_expanded_test_data_glove()


max_len = 100
model = Sequential()
lstm = LSTM(100, input_shape=(max_len, 100))
model.add(lstm)
# model.add(Dense(25, input_dim=100, activation='relu'))
# model.add(Dense(25, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])
# Fit the model

model.fit(X_train, Y_train, epochs=1, batch_size=64)

# evaluate the model
Y_pred = model.predict(X_test)
names = ["Insult", "Toxic", "Identity Hate", "Severe Toxic", "Obscene", "Threat"]

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
plt.title('3-Layer Neural Network TF-IDF Glove ROC Curve')
plt.legend(loc='best')
plt.savefig("lstm.png")

	

