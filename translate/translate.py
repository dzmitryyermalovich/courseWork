# In[1]:


from keras.models import Model
from keras.layers import Input, LSTM, Dense
import csv
import numpy as np
import codecs
import json
import string
import keras


# In[10]:


characters = string.printable
for i in characters:
    if(ord(i)>64 and ord(i)<91):
      characters=characters.replace(i,"")
token_eng_index = dict(zip(characters, range(1, len(characters) + 1)))

glagolitsa = "А,Б,В,Г,Д,Е,Ё,Ж,З,И,Й,К,Л,М,Н,О,П,Р,С,Т,У,Ф,Х,Ц,Ч,Ш,Щ,Ъ,Ы,Ь,Э,Ю,Я"
glagolitsa = glagolitsa.lower().split(',')
for i in glagolitsa:
    characters=characters+i
token_rus_index = dict(zip(characters, range(1, len(characters) + 1)))


# In[11]:


with open('C:\\machine learning\\data\\rusLower.csv', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    data = list(reader)


# In[12]:


dataArray=np.array(data)
samplesEng=dataArray[:,0]
samplesEng=samplesEng.reshape((samplesEng.shape[0],1))

samples = samplesEng

maxlen=50

encoder_input_data=np.zeros((len(samples),maxlen,len(token_eng_index)+1))

for i,sample in enumerate(samples):
    for word in sample:
        for j,chars in enumerate(word):
            if(j==maxlen):
                break;
            index=token_eng_index.get(chars,0)
            encoder_input_data[i,j,index]=1


# In[15]:


samplesRus=dataArray[:,1]
samplesRus=samplesRus.reshape((samplesRus.shape[0],1))
samples = samplesRus

maxlen=50

decoder_input_data=np.zeros((len(samples),maxlen+1,len(token_rus_index)+1))

for i,sample in enumerate(samples):
    for word in sample:
        for j,chars in enumerate(word):
            if(j==maxlen):
                break;
            index=token_rus_index.get(chars,0)
            decoder_input_data[i,j+1,index]=1

decoder_target_data=np.zeros((len(samples),maxlen+1,len(token_rus_index)+1))
decoder_target_data[:,0:50,:]=decoder_input_data[:,1:,:]


# In[17]:


lstm_dim=64


# In[19]:


encoder_input=Input(shape=(None,maxlen))
encoder=LSTM(lstm_dim,return_state=True)
encoder_outputs,state_h,state_c=encoder(encoder_input)
encoder_states=[state_h,state_c]


# In[28]:


decoder_input=Input(shape=(None,maxlen+1))
decoder=LSTM(lstm_dim,return_sequences=True,return_state=True)
decoder_outputs, _, _=decoder(decoder_input,initial_state=encoder_states)
decoder_dense=Dense(len(token_rus_index)+1,activation='softmax')
decoder_output=decoder_dense(decoder_outputs)


# In[29]:


model = Model([encoder_input, decoder_input], decoder_output)


# In[30]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# In[31]:


batch_size=128
epochs=10


# In[32]:


model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# In[ ]: