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
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
size = 400, 400

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


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 607)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(410, 300, 361, 251))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.verticalLayoutWidget)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayout.addWidget(self.plainTextEdit)
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(190, 190, 190);")
        self.pushButton.setObjectName("pushButton")

        self.pushButton.clicked.connect(self.WritePoem)

        self.verticalLayout.addWidget(self.pushButton)
        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_2.setGeometry(QtCore.QRect(410, 40, 361, 251))
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(300, 520, 91, 31))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(410, 0, 361, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(143, 143, 143);")
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menuOperation = QtWidgets.QMenu(self.menubar)
        self.menuOperation.setObjectName("menuOperation")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menuOperation.addAction("Import_Poet",self.IdentifyThePoet)

        self.menubar.addAction(self.menuOperation.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Generate"))
        self.label.setText(_translate("MainWindow", "Biography"))
        self.menuOperation.setTitle(_translate("MainWindow", "Operation"))

    def IdentifyThePoet(self):
        self.path = QFileDialog.getOpenFileName()[0]
        self.face_array=self.extract_face(self.path)
        self.face_emb=self.get_embedding(facenet_keras_model,self.face_array)
        self.samples = np.expand_dims(self.face_emb, axis=0)
        self.poet_name=self.DetectFace()

        im = Image.open(self.path)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save('C:\CourseWork\MyApplication\out.jpg', "JPEG")
        self.lbl = QtWidgets.QLabel(MainWindow)
        self.pix = QtGui.QPixmap('C:\\CourseWork\\MyApplication\\out.jpg')
        print(self.pix.isNull())
        self.lbl.setPixmap(self.pix)
        self.lbl.resize(500, 500)
        self.lbl.move(20, 20)
        self.lbl.show() 
        self.plainTextEdit_2.setPlainText("This is "+self.poet_name);

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
        mytext = self.textEdit.toPlainText()
        if mytext!="":
            n=int(mytext)
        else:
            n=200
    
        generated_text=sentense_gen
        for i in range(n):
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
        #print(self.poem)
        self.plainTextEdit.setPlainText(self.poem);
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

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
