import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, RNN, Reshape, Conv2D, Flatten
import numpy as np
from keras import backend as K
import os


class RNN:

    def __init__(self, params):
        self.params = params
        self.network_name = 'rnet'
        self.counter = 0
        self.model = self.build_model_Q()
        print("We have lift off")

    def getCalculations(self, q_t, y_pred, actions, terminals, rewards, discount):
        yj = np.add(rewards, np.multiply(1.0-terminals, np.multiply(discount, q_t)))
        Q_pred = np.sum(np.multiply(y_pred, actions))
        cost = np.sum(np.power(np.subtract(yj, Q_pred), 2))
        return cost

    def customLoss(self, q_t, actions, terminals, rewards, discount):
        def Loss(q_t, y_pred):
            return self.getCalculations(q_t, y_pred, actions, terminals, rewards, discount)
        return Loss

        
    def build_model_Q(self):
        model = Sequential()
        model.add(Conv2D(input_shape = (20,11,6), filters =16, kernel_size = 3, strides =  (1,1), padding='SAME', activation = 'relu'))
        model.add(Conv2D( filters = 32, kernel_size = 3,strides = (1,1), padding='SAME', activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(4))
        
        model.compile(optimizer = 'adam', loss = 'mse')
        return model

    def train(self,params,bat_s,bat_a,bat_t,bat_n,bat_r):
        #model_target = self.build_model_Q()
        #loss_target = self.customLoss(np.zeros(bat_n.shape[0]), bat_a, bat_t, bat_r, self.params['discount'])
        #model_target.compile(optimizer = 'adam', loss = loss_target)
        
        if(os. path. isfile('C:/Users/eshou/Desktop/testRNN.h5')):
            self.model.load_weights('C:/Users/eshou/Desktop/testRNN.h5')
            
        #model_target.train_on_batch(bat_n, np.zeros(bat_n.shape[0]))
        q_t_preds = self.model.predict(bat_n)
        q_t = np.amax(q_t_preds, axis=1)
        
        #loss_target = self.customLoss(q_t, bat_a, bat_t, bat_r, self.params['discount'])
        #self.model.compile(optimizer = 'adam', loss = loss_target)
        Q_values = bat_r + self.params['discount'] * q_t
        self.model.fit(bat_s, bat_a*Q_values[:,None], nb_epoch=1, batch_size=len(bat_s), verbose=1)
        
        
        print(self.counter)
        self.counter = self.counter + 1
     
        if self.counter % 10 == 0:
            print("Saving")
            self.model.save_weights('C:/Users/eshou/Desktop/testRNN.h5')
        
    
    def make_prediction(self, input):
        #q_model = self.build_model_Q()
        if(os. path. isfile('C:/Users/eshou/Desktop/testRNN.h5')):
            self.model.load_weights('C:/Users/eshou/Desktop/testRNN.h5')
        #q_model.compile(optimizer ='adam', loss = 'MSE')
        return self.model.predict(input)