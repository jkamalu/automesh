from keras import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Flatten, Reshape
from keras.optimizers import RMSprop

class CGAN:
    
    def __init__(self, params):
        self.params = params
        self.build_D()
        self.build_G()
        self.build_GD()
    
    def build_D(self):
        d_input = Input((self.params.W, self.params.H, self.params.n_channels))
        
        d_layer = Conv2D(self.params.n_filters, (4, 4), padding="same", activation="relu")(d_input)
        d_layer = MaxPooling2D()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)
        
        d_layer = Conv2D(self.params.n_filters * 2, (4, 4), padding="same", activation="relu")(d_layer)
        d_layer = MaxPooling2D()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)
        
        d_layer = Conv2D(self.params.n_filters * 4, (4, 4), padding="same", activation="relu")(d_layer)
        d_layer = MaxPooling2D()(d_layer)
        d_layer = Dropout(rate=self.params.drop_rate)(d_layer)        
        
        d_layer = Flatten()(d_layer)
        d_layer = Dense(1, activation="sigmoid")(d_layer)
        
        self.D = Model(inputs=d_input, outputs=d_layer)
        
        self.D.compile(optimizer=RMSprop(lr=0.0002), loss="binary_crossentropy", metrics=["accuracy"])
        
        return self.D
        
    def build_G(self):
        g_input = Input((self.params.n_rand + self.params.n_cond, ))
        
        g_layer = Dense((self.params.W // 8) * 
                        (self.params.H // 8) *
                        self.params.n_filters * 8, activation="relu")(g_input)
        g_layer = BatchNormalization()(g_layer)
        g_layer = Reshape((self.params.W // 8, 
                          self.params.H // 8,
                          self.params.n_filters * 8))(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)
        
        g_layer = UpSampling2D()(g_layer)
        g_layer = Conv2DTranspose(self.params.n_filters * 4, (4, 4), padding="same", activation="relu")(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)
        
        g_layer = UpSampling2D()(g_layer)
        g_layer = Conv2DTranspose(self.params.n_filters * 2, (4, 4), padding="same", activation="relu")(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)        
        
        g_layer = UpSampling2D()(g_layer)
        g_layer = Conv2DTranspose(self.params.n_filters, (4, 4), padding="same", activation="relu")(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)        
        
        g_layer = Conv2DTranspose(self.params.n_channels, (4, 4), padding="same", activation="sigmoid")(g_layer)
        g_layer = BatchNormalization()(g_layer)
        g_layer = Dropout(rate=self.params.drop_rate)(g_layer)
        
        g_image = g_layer
        
        self.G = Model(inputs=g_input, outputs=g_image)
        
        return self.G
        
    def build_GD(self):
        self.GD = Sequential()
        
        self.GD.add(self.G)
        
        self.D.trainable = False
        self.GD.add(self.D)
        
        self.GD.compile(optimizer=RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
        
        return self.GD