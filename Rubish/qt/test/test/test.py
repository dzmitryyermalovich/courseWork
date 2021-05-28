import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras import models
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QLabel
import sys
import numpy as np
from keras import models
import random
from PyQt5.QtGui import QIcon, QPixmap

maxlen=60

facenet_keras_model = models.load_model('C:\\CourseWork\\FaceRecognitionPoets\\FaceModel\\facenet_keras.h5')
import pickle
from sklearn import preprocessing
with open('C:\CourseWork\FaceRecognitionPoets\FaceModel\MyTrainModel', 'rb') as f:
   model=pickle.load(f)

data = np.load('C:\\CourseWork\\FaceRecognitionPoets\\FaceRecognitionEmbedding\\FaceRecognition\\poets-faces-embeddings.npz')
trainy=data['arr_1']
out_encoder = preprocessing.LabelEncoder()
out_encoder.fit(trainy)

model_pushkin=models.load_model('C:\\CourseWork\\GeneratePoetsText\\data\\PoetsModels\\ModelsPushkin\\model_60_pushkin.h5')
model_blok=models.load_model('C:\\CourseWork\\GeneratePoetsText\\data\\PoetsModels\\BlokModel\\model_60_blok.h5')
model_yesenin=models.load_model('C:\\CourseWork\\GeneratePoetsText\\data\\PoetsModels\\ModelEsenin\\model_60_Esenin.h5')
model_tyutchev=models.load_model('C:\\CourseWork\\GeneratePoetsText\\data\\PoetsModels\\TyutchevModel\\model_60_Tyutchev.h5')

with open('C:\\CourseWork\\GeneratePoetsText\\data\\TextData\\pushkinPoems.txt',encoding="UTF-8") as fp:
    dataPushkin=fp.read()
with open('C:\\machine learning\\ProcessPushkinData\\EseninPoems.txt',encoding="UTF-8") as fp:
    dataYesenin=fp.read()
with open('C:\\machine learning\\ProcessPushkinData\\TyutchevPoems.txt',encoding="UTF-8") as fp:
    dataTyutchev=fp.read()
with open('C:\\machine learning\\ProcessPushkinData\\BlokPoems.txt',encoding="UTF-8") as fp:
    dataBlok=fp.read()

dataPushkin=dataPushkin.lower()
dataYesenin=dataYesenin.lower()
dataTyutchev=dataTyutchev.lower()
dataBlok=dataBlok.lower()

glagolitsa_Pushkin = sorted(list(set(dataPushkin)))
glagolitsa_Yesenin = sorted(list(set(dataYesenin)))
glagolitsa_Tyutchev = sorted(list(set(dataTyutchev)))
glagolitsa_Blok = sorted(list(set(dataBlok)))

char_indices_pushkin = dict((char, glagolitsa_Pushkin.index(char)) for char in glagolitsa_Pushkin)
indices_char_pushkin = dict((glagolitsa_Pushkin.index(char),char) for char in glagolitsa_Pushkin)

char_indices_yesenin = dict((char, glagolitsa_Yesenin.index(char)) for char in glagolitsa_Yesenin)
indices_char_yesenin = dict((glagolitsa_Yesenin.index(char),char) for char in glagolitsa_Yesenin)

char_indices_tyutchev = dict((char, glagolitsa_Tyutchev.index(char)) for char in glagolitsa_Tyutchev)
indices_char_tyutchev = dict((glagolitsa_Tyutchev.index(char),char) for char in glagolitsa_Tyutchev)

char_indices_blok = dict((char, glagolitsa_Blok.index(char)) for char in glagolitsa_Blok)
indices_char_blok = dict((glagolitsa_Blok.index(char),char) for char in glagolitsa_Blok)


class Window(QMainWindow):
    def __init__(self):
        super(Window,self).__init__()

        self.setWindowTitle("S1mple program")
        self.setGeometry(250,200,650,400)

        self.btn1=QtWidgets.QPushButton(self)
        self.btn1.move(300,380)
        self.btn1.setText("open")
        self.btn1.clicked.connect(self.IdentifyThePoet)

        self.btn2=QtWidgets.QPushButton(self)
        self.btn2.move(300,400)
        self.btn2.setText("print")
        self.btn2.clicked.connect(self.print)

        self.btn3=QtWidgets.QPushButton(self)
        self.btn3.move(300,420)
        self.btn3.setText("generate")
        self.btn3.clicked.connect(self.WritePoem)

        self.btn4=QtWidgets.QPushButton(self)
        self.btn4.move(300,440)
        self.btn4.setText("print Photo")
        self.btn4.clicked.connect(self.WritePoem)

    def IdentifyThePoet(self):
        self.path = QFileDialog.getOpenFileName()[0]
        self.face_array=self.extract_face(self.path)
        self.face_emb=self.get_embedding(facenet_keras_model,self.face_array)
        self.samples = np.expand_dims(self.face_emb, axis=0)
        self.poet_name=self.DetectFace()
        return self.poet_name

    def extract_face(self,filename,required_size=(160, 160)):
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

    def get_embedding(self,model,face_pixels):
	    face_pixels = face_pixels.astype('float32')

	    mean, std = face_pixels.mean(), face_pixels.std()
	    face_pixels = (face_pixels - mean) / std

	    samples = np.expand_dims(face_pixels, axis=0)

	    yhat = model.predict(samples)
	    return yhat[0]

    def DetectFace(self):
        yhat_class = model.predict(self.samples)
        yhat_prob = model.predict_proba(self.samples)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        return predict_names[0]

    def print(self):
        print(self.poet_name)
        return self.poet_name

    def sample(self,preds, temperature=1.0):
         preds = np.asarray(preds).astype('float64')
         preds = np.log(preds) / temperature
         exp_preds = np.exp(preds)
         preds = exp_preds / np.sum(exp_preds)
         probas = np.random.multinomial(1, preds, 1)
         return np.argmax(probas)

    def generate_sentence(self,model,sentense_gen,char_indices,indices_char,glagolitsa):
        generated_text=sentense_gen
        for i in range(200):
            sampled=np.zeros(shape=(1,maxlen,len(glagolitsa)))
            for j,chars in enumerate(sentense_gen):
                sampled[0,j,char_indices.get(chars)]=1
            preds=model.predict(sampled,verbose=0)[0]
            #next_index = sample(preds) 
            next_index=np.argmax(preds)
            next_char = indices_char[next_index]
            sentense_gen += next_char
            generated_text += next_char
            sentense_gen = sentense_gen[1:]
        
        return generated_text

    def GeneratePoem(self,poet_name):
        if poet_name=="Pushkin":
            start_index = random.randint(0, len(dataPushkin) - maxlen - 1)
            generated_text = dataPushkin[start_index: start_index + maxlen] 
            self.poem=self.generate_sentence(model_pushkin,generated_text,char_indices_pushkin,indices_char_pushkin,glagolitsa_Pushkin)
        elif poet_name=="Blok":
            start_index = random.randint(0, len(dataBlok) - maxlen - 1) 
            generated_text = dataBlok[start_index: start_index + maxlen] 
            self.poem=self.generate_sentence(model_blok,generated_text,char_indices_blok,indices_char_blok,glagolitsa_Blok)
        elif poet_name=="Yesenin":
            start_index = random.randint(0, len(dataYesenin) - maxlen - 1) 
            generated_text = dataYesenin[start_index: start_index + maxlen] 
            self.poem=self.generate_sentence(model_yesenin,generated_text,char_indices_yesenin,indices_char_yesenin,glagolitsa_Yesenin)
        elif poet_name=="Tyutchev":
            start_index = random.randint(0, len(dataTyutchev) - maxlen - 1)
            generated_text = dataTyutchev[start_index: start_index + maxlen] 
            self.poem=self.generate_sentence(model_tyutchev,generated_text,char_indices_tyutchev,indices_char_tyutchev,glagolitsa_Tyutchev)
        return self.poem

    def WritePoem(self):
        self.GeneratePoem(self.poet_name)
        print(self.poem)
        return self.poem

    def printPhot(self):
        label = QLabel(self)
        pixmap = QPixmap(self.path)
        self.resize(pixmap.width(),pixmap.height())
        label.setPixmap(pixmap)
        self.show()

def application():
    app=QApplication(sys.argv)
    window=Window()

    window.show()
    sys.exit(app.exec_())

application()