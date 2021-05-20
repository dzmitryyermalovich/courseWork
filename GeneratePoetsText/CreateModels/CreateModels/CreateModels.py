import numpy as np
import keras


with open('C:\\CourseWork\\GeneratePoetsText\\data\\TextData\\EseninPoems.txt',encoding="UTF-8") as fp:
    dataInputString=fp.read()


dataInputString=dataInputString.lower()


glagolitsa = sorted(list(set(dataInputString)))


char_indices = dict((char, glagolitsa.index(char)) for char in glagolitsa)


indices_char = dict((glagolitsa.index(char),char) for char in glagolitsa)


step=3
maxlen=60
sentenseInput=[]
next_char_index=[]
for i in range(0,len(dataInputString)- maxlen,step):
    sentenseInput.append(dataInputString[i:i+maxlen])
    next_char_index.append(dataInputString[i+maxlen])



InputData=np.zeros(shape=(len(sentenseInput),maxlen,len(glagolitsa)),dtype=np.bool)
OutputData=np.zeros(shape=(len(sentenseInput),len(glagolitsa)),dtype=np.bool)



for i,sentense in enumerate(sentenseInput):
    for j,chars in enumerate(sentense):
       InputData[i,j,char_indices.get(chars)]=1
    OutputData[i,char_indices.get(next_char_index[i])]=1

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import layers
from keras import models

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen,len(glagolitsa))))
model.add(layers.Dense(len(glagolitsa), activation='softmax'))
model.summary()


print(InputData.shape)
print(OutputData.shape)


optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


for epoch in range(62): 
     model.fit(InputData,OutputData,epochs=1,batch_size=128)
     if epoch==20:
            model.save('C:\CourseWork\GeneratePoetsText\data\PoetsModels\ModelEsenin\model_20_Esenin.h5')
     if epoch==40:
            model.save('C:\CourseWork\GeneratePoetsText\data\PoetsModels\ModelEsenin\model_40_Esenin.h5')
     if epoch==60:
            model.save('C:\CourseWork\GeneratePoetsText\data\PoetsModels\ModelEsenin\model_60_Esenin.h5')       
