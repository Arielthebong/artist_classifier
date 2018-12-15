import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from keras.utils.np_utils import to_categorical
#%matplotlib inline

only_25_artists = True

# read in CSV, replace blank lyrics with empty string
#df = pd.read_csv('lyrics.csv', converters={'lyrics' : lambda x: "" if x == np.nan else x})

df = pd.read_csv('datasets/lyrics.csv')

df.dropna(how='any', inplace=True) #drop all blank rows
#df.dropna(subset=['lyrics'])

#print(df.head())

df.drop(['index', 'song', 'year', 'genre'],axis=1,inplace=True)

artists = ['beyonce-knowles', '50-cent', 'eazy-e', 'casey-veggies', 'fetty-wap', 'flatbush-zombies', 'bas', 'frank-ocean', 'grandmaster-flash', 'childish-gambino', 'clipse', 'big-l', 'aloe-blacc', 'eminem', 'future', 'flobots', 'david-banner', '2-chainz', 'drake', 'big-sean', 'dr-dre', 'earl-sweatshirt', 'chance-the-rapper', 'common', 'asap-rocky']

if only_25_artists:
	df = df[df.artist.isin(artists)]

#df = df[:50000]

print(df.head())

print(df.info())

#sns.countplot(df.artist)
#plt.xlabel('Label')
#plt.title('Graph of songs per artist')
#plt.show()

Y = df.artist
X = df.lyrics

le = LabelEncoder()
Y = le.fit_transform(Y)
n_classes = len(le.classes_)
print('number of artists: '+str(n_classes))
#Y = np.array([y*1.0/n_classes for y in Y])
Y = Y.reshape(-1,1)

Y = to_categorical(Y)


# BY Doing this, we test on a subset of ALL artists instead of our top 25
# maybe, to reduce # of artists, have it only look at artists who have more than
# 50 songs?
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.10, shuffle=True)

print("Y_train: "+str(Y_train))
#X_val,X_test,Y_val,Y_test = train_test_split(X_testAndVal,Y_testAndVal,test_size=0.60, shuffle=True)

print("X_train: " + str(X_train))

max_words = 1000000 # this should probably be higher -- only keeps 1000000 most common words in dataset
max_len = 150
tok = Tokenizer(num_words=max_words)    
tok.fit_on_texts(X_train) # problem -- need to get rid of blank rows of lyrics (which are represented as NaN) before calling this
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(n_classes,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=25,
          validation_split=0.1)#,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

