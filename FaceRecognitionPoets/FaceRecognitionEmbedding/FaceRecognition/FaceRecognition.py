import numpy as np
from keras import models
def get_embedding(model, face_pixels):

	face_pixels = face_pixels.astype('float32')

	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std

	samples = np.expand_dims(face_pixels, axis=0)

	yhat = model.predict(samples)
	return yhat[0]

data=np.load('C:\\CourseWork\\FaceRecognitionPoets\\FaceRecognitionProcessPhoto\\FaceRecognition\\poets-faces-dataset.npz')

trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

model = models.load_model('C:\\CourseWork\\FaceRecognitionPoets\\FaceModel\\facenet_keras.h5')

print('Loaded Model')

newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)

newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)

np.savez_compressed('poets-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)