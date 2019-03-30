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
        # self.x = tf.placeholder('float', [20, 11, 6],name=self.network_name + '_x')
        # self.q_t = tf.placeholder('float', [None], name=self.network_name + '_q_t')
        # self.actions = tf.placeholder("float", [None, 4], name=self.network_name + '_actions')
        # self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')
        # self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')
        # self.sess = tf.Session()
        # self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.model = Sequential()
        self.model.add(Conv2D(input_shape = (20,11,6), filters =16, kernel_size = 3, strides =  (1,1), padding='SAME', activation = 'relu'))
        self.model.add(Conv2D( filters = 32, kernel_size = 3,strides = (1,1), padding='SAME', activation = 'relu'))
        self.model.add(Dense(256, activation = 'relu'))
        self.model.add(Dense(4))
        # The discount may need to be constant
        #self.discount = self.params.discount
        #self.yj = self.model.add(self.)
        self.model.compile(optimizer='adam', loss='MSE')
        
    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        #feed_array = np.array[bat_n, np.zeros(bat_n.shape[0]), bat_a, bat_t, bat_r]
        print("BAT N")
        print(bat_n.shape)
        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        #q_t = self.model.fit(x=feed_array, y = None, batch_size = 32, epochs = 32, verbose = 1)
        q_t = self.sess.run(self.y, feed_dict = feed_dict)
        q_t = np.amax(q_t, axis=1)
        exit()
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,cnt,cost = self.model.fit(x=feed_dict, y= q_t, batch_size = 32, epochs = 32, verbose=1)
        
        print("Training")
        
        self.model.save('C:/Users/eshou/Desktop/testRNN.h5')
        return cnt, cost
    
    def predict(self, data):
        self.model.predict(x = data, batch_size = None)