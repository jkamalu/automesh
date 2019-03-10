import numpy as np

from keras import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Flatten, Reshape
from keras.optimizers import Adam, RMSprop, SGD
from keras.initializers import TruncatedNormal
from keras.layers import ReLU, LeakyReLU, Activation

import keras.backend as K

def accuracy(y_true, y_pred):
    pred_greater = K.greater(y_pred, 0.5)
    true_greater = K.greater(y_true, 0.5)
    return K.sum(K.cast(K.equal(pred_greater, true_greater), "float32")) / K.cast(K.shape(y_true)[0], "float32")

class CGAN:
    
    def __init__(self, params):
        self.params = params
        self.D = self.build_D(strides=(2, 2))
        self.G = self.build_G(strides=(2, 2))
        self.GD = self.build_GD()
        
    def __repr__(self):
        string = ""
        string += "Parameters:\n"
        string += str(vars(self.params))
        string += "\n" * 2
        string += "Discriminator:\n"
        D_lines = []
        self.D.summary(print_fn=lambda x: D_lines.append(x))
        string += "\n".join(D_lines)
        string += "\n" * 2
        string += "Generator:\n"
        G_lines = []
        self.G.summary(print_fn=lambda x: G_lines.append(x))
        string += "\n".join(G_lines)
        string += "\n" * 2
        string += "Adversarial:\n"
        GD_lines = []
        self.GD.summary(print_fn=lambda x: GD_lines.append(x))
        string += "\n".join(GD_lines)
        return string

    def build_D(self, strides=(1, 1)):
        d_input = Input((self.params.W, self.params.H, self.params.n_channels))
        
        # Convolve and dropout
        d_layer = Conv2D(self.params.n_filters * 2, (4, 4), 
                         strides=strides, 
                         padding="same", 
                         kernel_initializer=RandomNormal(stddev=0.02))(d_input)
        d_layer = LeakyReLU(alpha=0.2)(d_layer)
        if strides == (1, 1):
            d_layer = MaxPooling2D()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)
        
        # Convolve and dropout
        d_layer = Conv2D(self.params.n_filters * 4, (4, 4), 
                         strides=strides, 
                         padding="same", 
                         kernel_initializer=RandomNormal(stddev=0.02))(d_layer)
        d_layer = BatchNormalization()(d_layer)
        d_layer = LeakyReLU(alpha=0.2)(d_layer)
        if strides == (1, 1):
            d_layer = MaxPooling2D()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)
        
        # Convolve and dropout
        d_layer = Conv2D(self.params.n_filters * 4, (4, 4), 
                         strides=strides, 
                         padding="same", 
                         kernel_initializer=RandomNormal(stddev=0.02))(d_layer)
        d_layer = BatchNormalization()(d_layer)        
        d_layer = LeakyReLU(alpha=0.2)(d_layer)
        if strides == (1, 1):
            d_layer = MaxPooling2D()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)
        
        # Convolve and dropout
        d_layer = Conv2D(self.params.n_filters * 4, (4, 4), 
                         strides=strides, padding="same", 
                         kernel_initializer=RandomNormal(stddev=0.02))(d_layer)
        d_layer = BatchNormalization()(d_layer)
        d_layer = LeakyReLU(alpha=0.2)(d_layer)
        if strides == (1, 1):
            d_layer = MaxPooling2D()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)     
        
        # Flatten and predict
        d_layer = Flatten()(d_layer)
        d_layer = Dense(1, kernel_initializer=RandomNormal(stddev=0.02))(d_layer)
        d_layer = Activation("sigmoid")(d_layer)
        
        self.D = Model(inputs=d_input, outputs=d_layer)
        
        self.D.compile(optimizer=Adam(lr=self.params.lr_D, beta_1=0.5), loss="binary_crossentropy", metrics=[accuracy])
        
        return self.D
        
    def build_G(self, strides=(1, 1)):
        g_input = Input((self.params.n_rand + self.params.n_cond, ))
        
        # Project the data into higher dimensional space
        g_layer = Dense((self.params.W // 8) * (self.params.H // 8) * self.params.n_filters * 16, kernel_initializer=RandomNormal(stddev=0.02))(g_input)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)
        
        # Reshape the data for convolution
        g_layer = Reshape((self.params.W // 8, self.params.H // 8, self.params.n_filters * 16))(g_layer)
        
        # Upsample with many filters
        if strides == (1, 1):
            g_layer = UpSampling2D()(g_layer)
        g_layer = Conv2DTranspose(self.params.n_filters * 16, (4, 4), strides=strides, padding="same", kernel_initializer=RandomNormal(stddev=0.02))(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)
        
        # Upsample
        if strides == (1, 1):
            g_layer = UpSampling2D()(g_layer)
        g_layer = Conv2DTranspose(self.params.n_filters * 8, (4, 4), strides=strides, padding="same", kernel_initializer=RandomNormal(stddev=0.02))(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)        
        
        # Upsample with fewer filters
        if strides == (1, 1):
            g_layer = UpSampling2D()(g_layer)
        g_layer = Conv2DTranspose(self.params.n_filters * 4, (4, 4), strides=strides, padding="same", kernel_initializer=RandomNormal(stddev=0.02))(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)
        
        # Convolve with fewer filters
        g_layer = Conv2DTranspose(self.params.n_filters * 2, (4, 4), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)        
        
        # Convolve to n_channels filters
        g_layer = Conv2DTranspose(self.params.n_channels, (4, 4), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(g_layer)
        g_layer = Activation("tanh")(g_layer)
        
        self.G = Model(inputs=g_input, outputs=g_layer)
        
        return self.G
        
    def build_GD(self):
        self.GD = Sequential()
        
        self.GD.add(self.G)
        
        self.D.trainable = False
        self.GD.add(self.D)
        
        self.GD.compile(optimizer=Adam(lr=self.params.lr_GD, beta_1=0.5), loss="binary_crossentropy", metrics=[accuracy])
                
        return self.GD
    
if __name__ == "__main__":
    from Params import Params
    X = np.zeros((10, 88, 64, 1))
    params = Params(X)
    model = CGAN(params)
    model.D.summary()
    model.G.summary()
    model.GD.summary()
    
    