import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
import numpy as np

class RNN:

    def __init__(self,parameters):
    
        print("We have lift off")
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences = False, dropout=0.1, recurrent_dropout=0.1, stateful = False))
        self.model.add(Dense(64, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    def train(training):
        self.model.fit(x = training, batch_size = 32, epochs = 32, verbose=1)
        self.model.save('C:\Test\testRNN.h5')
    
    def predict(self, data):
        self.model.predict(x = data, batch_size = None)