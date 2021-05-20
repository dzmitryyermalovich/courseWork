import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from random import choice
data = np.load('C:\\CourseWork\\FaceRecognitionPoets\\FaceRecognitionEmbedding\\FaceRecognition\\poets-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

in_encoder = preprocessing.Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = preprocessing.LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

model = svm.SVC(kernel='linear',probability=True)
model.fit(trainX, trainy)

import pickle

#with open('C:\CourseWork\FaceRecognitionPoets\FaceModel\MyTrainModel', 'wb') as f:
#   pickle.dump(model,f)

#with open('C:\CourseWork\FaceRecognitionPoets\FaceModel\MyTrainModel', 'rb') as f:
#   new_model=pickle.load(f)

yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

score_train = metrics.accuracy_score(trainy, yhat_train)
score_test = metrics.accuracy_score(testy, yhat_test)

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
#########################################################################################################

data = np.load('C:\\CourseWork\\FaceRecognitionPoets\\FaceRecognitionProcessPhoto\\FaceRecognition\\poets-faces-dataset.npz')
testX_faces = data['arr_2']

selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

samples = np.expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])

plt.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
plt.title(title)
plt.show()