import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, RNN, Reshape, Conv2D
import numpy as np
from keras import backend as K

class RNN:

    def __init__(self, params):
        self.params = params
        self.network_name = 'rnet'
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

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape = (20,11,6), filters =16, kernel_size = 3, strides =  (1,1), padding='SAME', activation = 'relu'))
        model.add(Conv2D( filters = 32, kernel_size = 3,strides = (1,1), padding='SAME', activation = 'relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(4))

        return model

    def train(self,params,bat_s,bat_a,bat_t,bat_n,bat_r):
        #feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        model = self.build_model()

        loss = self.customLoss(np.zeros(bat_n.shape[0]), bat_a, bat_t, bat_r, self.params['discount'])
        model.compile(optimizer = 'adam', loss = loss)
        #q_t = self.sess.run(self.y, feed_dict = feed_dict)
        q_t = model.train_on_batch(x=bat_n, y= np.zeros(bat_n.shape[0]))
        q_t = np.amax(q_t, axis=1)
        print("QT")
        print(q_t)
        print(q_t.shape)
        loss = self.customLoss(q_t, bat_a, bat_t, bat_r, self.params.discount)
        model.compile(optimizer = 'adam', loss = loss)
        #feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,cnt,cost = self.model.train_on_batch(x=bat_s, y= q_t)
        
        print("Training")
        
        self.model.save('C:/Users/Andy Garcia/Desktop/testRNN.h5')
        return cnt, cost
    
    #def predict(self, data):
    #    self.model.predict(x = data, batch_size = None)