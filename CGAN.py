import numpy as np

from keras import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Flatten, Reshape
from keras.optimizers import Adam
from keras.layers import ReLU, LeakyReLU, Activation

class CGAN:
    
    def __init__(self, params):
        self.params = params
        self.build_D()
        self.build_G()
        self.build_GD()
    
    def build_D(self):
        d_input = Input((self.params.W, self.params.H, self.params.n_channels))
        
        d_layer = Conv2D(self.params.n_filters, (4, 4), strides=(2, 2), padding="same")(d_input)
        d_layer = LeakyReLU()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)
        
        d_layer = Conv2D(self.params.n_filters * 2, (4, 4), strides=(2, 2), padding="same")(d_layer)
        d_layer = BatchNormalization()(d_layer)
        d_layer = LeakyReLU()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)
        
        d_layer = Conv2D(self.params.n_filters * 4, (4, 4), strides=(2, 2), padding="same")(d_layer)
        d_layer = BatchNormalization()(d_layer)
        d_layer = LeakyReLU()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)
        
        d_layer = Conv2D(self.params.n_filters * 8, (4, 4), strides=(2, 2), padding="same")(d_layer)
        d_layer = BatchNormalization()(d_layer)
        d_layer = LeakyReLU()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)        
        
        d_layer = Flatten()(d_layer)
        d_layer = Dense(1, activation="sigmoid")(d_layer)
        
        self.D = Model(inputs=d_input, outputs=d_layer)
        
        self.D.compile(optimizer=Adam(lr=self.params.lr_D, beta_1=self.params.beta_1), loss="binary_crossentropy", metrics=["accuracy"])
        
        return self.D
        
    def build_G(self):
        g_input = Input((self.params.n_rand + self.params.n_cond, ))
        
        g_layer = Dense((self.params.W // 8) * (self.params.H // 8) * self.params.n_filters * 8)(g_input)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)
        
        g_layer = Reshape((self.params.W // 8, self.params.H // 8, self.params.n_filters * 8))(g_layer)
        
        g_layer = Conv2DTranspose(self.params.n_filters * 4, (4, 4), strides=(2, 2), padding="same")(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)
        
        g_layer = Conv2DTranspose(self.params.n_filters * 2, (4, 4), strides=(2, 2), padding="same")(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)        
        
        g_layer = Conv2DTranspose(self.params.n_filters, (4, 4), strides=(2, 2), padding="same")(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = ReLU()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)
        
        g_layer = Conv2DTranspose(self.params.n_channels, (4, 4), padding="same")(g_layer)
        g_layer = Activation("sigmoid")(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)
        
        self.G = Model(inputs=g_input, outputs=g_layer)
        
        return self.G
        
    def build_GD(self):
        self.GD = Sequential()
        
        self.GD.add(self.G)
        
        self.D.trainable = False
        self.GD.add(self.D)
        
        self.GD.compile(optimizer=Adam(lr=self.params.lr_GD, beta_1=self.params.beta_1), loss="binary_crossentropy", metrics=["accuracy"])
        
        self.D.trainable = True
        
        return self.GD
    
if __name__ == "__main__":
    from Params import Params
    X = np.zeros((10, 88, 64, 1))
    params = Params(X)
    model = CGAN(params)
    model.D.summary()
    model.G.summary()
    model.GD.summary()
    
    