import keras
from keras.datasets import mnist
from keras.layers import *
from keras.preprocessing import image
from keras.models import *
from keras.utils import to_categorical
import numpy as np
from utils import getActivationValue,layerName, hard_sigmoid
from keract import get_activations_single_layer

class mnistclass:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.imagesize = 28
        self.load_data()
        self.numAdv = 0
        self.numSamples = 0
        self.perturbations = []

    def load_data(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = self.X_train.astype('float32') / 255
        self.X_test = self.X_test.astype('float32') / 255
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)


    def load_model(self):
        self.model = load_model('models/mnist_lstm.h5')
        opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.summary()

    def layerName(self, layer):
        layerNames = [layer.name for layer in self.model.layers]
        return layerNames[layer]

    def train_model(self):
        self.load_data()
        self.model = Sequential()
        # input_shape = (batch_size, timesteps, input_dim)
        self.model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
        self.model.add(LSTM(128))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, epochs=3, validation_data=(self.X_test, self.y_test))
        self.model.save('models/mnist_lstm.h5')

    def displayInfo(self,test):
        test = test[np.newaxis, :]
        output_prob = np.squeeze(self.model.predict(test))
        output_class = np.argmax(output_prob)
        conf = np.max(output_prob)
        print("current digit: ", output_class)
        print("current confidence: %.2f\n"%(conf))
        return (output_class, conf)


    def from_array_to_image(self,test):
        test = test*255
        test = test.astype(int)
        test = test.reshape((28, 28, 1))
        pred_img = image.array_to_img(test)
        pred_img.save('output.jpg')

    def image_plot(self,test):
        img_class = self.model.predict_classes(test)
        classname = img_class[0]
        # # show image in matplot
        plt.imshow(test)
        plt.title(classname)
        plt.show()

    def updateSample(self,label2,label1,m,o):
        if label2 != label1 and o == True:
            self.numAdv += 1
            self.perturbations.append(m)
        self.numSamples += 1
        self.displaySuccessRate()

    def displaySamples(self):
        print("%s samples are considered" % (self.numSamples))

    def displaySuccessRate(self):
        print("%s samples, within which there are %s adversarial examples" % (self.numSamples, self.numAdv))
        print("the rate of adversarial examples is %.2f\n" % (self.numAdv / self.numSamples))

    def displayPerturbations(self):
        if self.numAdv > 0:
            print("the average perturbation of the adversarial examples is %s" % (sum(self.perturbations) / self.numAdv))
            print("the smallest perturbation of the adversarial examples is %s" % (min(self.perturbations)))

    # calculate the lstm hidden state and cell state manually (no dropout)
    # activation function is tanh
    def cal_hidden_state(self, test, layernum):
        if layernum == 0:
            acx = test
        else:
            acx = get_activations_single_layer(self.model, np.array([test]), self.layerName(layernum-1))

        units = int(int(self.model.layers[layernum].trainable_weights[0].shape[1]) / 4)
        # print("No units: ", units)
        # lstm_layer = model.layers[1]
        W = self.model.layers[layernum].get_weights()[0]
        U = self.model.layers[layernum].get_weights()[1]
        b = self.model.layers[layernum].get_weights()[2]

        W_i = W[:, :units]
        W_f = W[:, units: units * 2]
        W_c = W[:, units * 2: units * 3]
        W_o = W[:, units * 3:]

        U_i = U[:, :units]
        U_f = U[:, units: units * 2]
        U_c = U[:, units * 2: units * 3]
        U_o = U[:, units * 3:]

        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]

        # calculate the hidden state value
        h_t = np.zeros((self.imagesize, units))
        c_t = np.zeros((self.imagesize, units))
        f_t = np.zeros((self.imagesize, units))
        h_t0 = np.zeros((1, units))
        c_t0 = np.zeros((1, units))

        for i in range(0, self.imagesize):
            f_gate = hard_sigmoid(np.dot(acx[i, :], W_f) + np.dot(h_t0, U_f) + b_f)
            i_gate = hard_sigmoid(np.dot(acx[i, :], W_i) + np.dot(h_t0, U_i) + b_i)
            o_gate = hard_sigmoid(np.dot(acx[i, :], W_o) + np.dot(h_t0, U_o) + b_o)
            new_C = np.tanh(np.dot(acx[i, :], W_c) + np.dot(h_t0, U_c) + b_c)
            c_t0 = f_gate * c_t0 + i_gate * new_C
            h_t0 = o_gate * np.tanh(c_t0)
            c_t[i, :] = c_t0
            h_t[i, :] = h_t0
            f_t[i, :] = f_gate

        return h_t, c_t, f_t


