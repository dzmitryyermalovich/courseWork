import keras
from keras.models import load_model
from keras import models
import numpy as np
from mtcnn.mtcnn import MTCNN
import os
import mtcnn
from PIL import Image
from IPython.display import display

def extract_face(filename,required_size=(160, 160)):

    image=Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)

    detector=MTCNN()
    results=detector.detect_faces(pixels)
    x1,y1,width,height=results[0]['box']
    x1,y1=abs(x1),abs(y1)
    x2,y2=x1+width,y1+height

    pixels_face=pixels[y1:y2,x1:x2]

    image = Image.fromarray(pixels_face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    return face_array

def load_faces(directory):
	faces = list()
	for filename in os.listdir(directory):
		path = directory + filename
		face = extract_face(path)
		faces.append(face)
	return faces

def load_dataset(directory):
	X, y = list(), list()

	for subdir in os.listdir(directory):

		path = directory + subdir+"\\"

		if not os.path.isdir(path):
			continue

		faces = load_faces(path)

		labels = [subdir for _ in range(len(faces))]

		print('>loaded %d examples for class: %s' % (len(faces), subdir))

		X.extend(faces)
		y.extend(labels)
	return np.asarray(X), np.asarray(y)

trainX, trainy = load_dataset('C:\\CourseWork\\FaceRecognitionPoets\\Data\\train\\')
print(trainX.shape, trainy.shape)

testX, testy = load_dataset('C:\\CourseWork\\FaceRecognitionPoets\\Data\\val\\')

np.savez_compressed('poets-faces-dataset.npz', trainX, trainy, testX, testy)