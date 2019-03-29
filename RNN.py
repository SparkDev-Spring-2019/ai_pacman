import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, RNN, Reshape, Conv1D
import numpy as np

class RNN:

    def __init__(self, params, numeps):
    
        print("We have lift off")
        self.model = Sequential()
        self.model.add(Conv1D(6, kernel_size = 1))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy') #loss = meansquare error
        
    
    def train(self, training):
        print("Training")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.fit(training, training, batch_size = 32, epochs = 32, verbose=1)
        self.model.save('C:/Users/eshou/Desktop/testRNN.h5')
    
    def predict(self, data):
        self.model.predict(x = data, batch_size = None)