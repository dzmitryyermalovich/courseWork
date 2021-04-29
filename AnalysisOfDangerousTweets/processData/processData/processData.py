import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import preprocessing
import os
import keras


with open('C:\\CourseWork\\data\\AnalysisOfDangerousTweets\\trainToLower.csv', newline='',encoding='cp437') as f:
    reader = csv.reader(f)
    data = list(reader)

dataArray=np.array(data)
trainText=dataArray[:,3]
trainLabels=dataArray[:,4]
maxlen=15
max_words=10000
tokenizer=Tokenizer(num_words=10000)
tokenizer.fit_on_texts(trainText)
sequences=tokenizer.texts_to_sequences(trainText)
word_index = tokenizer.word_index

sequences= preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
trainLabels=trainLabels.reshape((sequences.shape[0], 1))
trainLabels=trainLabels.astype('int32')
sequences=sequences.astype('float32')

with open('C:\\CourseWork\\data\\AnalysisOfDangerousTweets\\testLower.csv', newline='',encoding='cp437') as f:
    reader2 = csv.reader(f)
    data2 = list(reader2)

dataArrayTest=np.array(data2)

testText=dataArrayTest[2000:dataArrayTest.shape[0],3]
testLabels=np.zeros(shape=(dataArrayTest.shape[0]-2000,1))

valText=dataArrayTest[0:2000,3]
valLabels=np.zeros(shape=(2000,1))

tokenizer.fit_on_texts(testText)
sequencesTest=tokenizer.texts_to_sequences(testText)
sequencesTest= preprocessing.sequence.pad_sequences(sequencesTest, maxlen=maxlen)
sequencesTest=sequencesTest.astype('float32')
testLabels=testLabels.astype('int32')

tokenizer.fit_on_texts(valText)
sequencesVal=tokenizer.texts_to_sequences(valText)
sequencesVal= preprocessing.sequence.pad_sequences(sequencesVal, maxlen=maxlen)
sequencesVal=sequencesVal.astype('float32')
valLabels=valLabels.astype('int32')

embeddings_index = {}
f = open(os.path.join("C:\\machine learning\\data", 'glove.6B.100d.txt'),encoding='cp437')
for line in f:
 values = line.split()
 word = values[0]
 coefs = np.asarray(values[1:], dtype='float32')
 embeddings_index[word] = coefs
f.close()

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
 if i < max_words:
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
   embedding_matrix[i] = embedding_vector

callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.7, patience=5,),keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,)]

from keras import layers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.layers import GRU
model = Sequential()
model.add(Embedding(10000,embedding_dim,input_length=maxlen))
model.add(layers.GRU(64,dropout=0.5,recurrent_dropout=0.5,return_sequences=True))
model.add(layers.GRU(64,dropout=0.2,recurrent_dropout=0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(sequences, trainLabels,epochs=30,batch_size=128,callbacks=callbacks_list,validation_data=(sequencesVal, valLabels))

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['acc']
val_loss_values = history_dict['val_acc']
plt.plot(loss_values, 'bo', label='Training loss')
plt.plot(val_loss_values, 'b', label='Training loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.evaluate(sequencesTest,testLabels)