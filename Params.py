class Params:
    
    def __init__(self, X, X_cond=None):
        _, self.W, self.H, self.n_channels = X.shape
        _, self.n_cond = X_cond.shape if X_cond is not None else (None, 0)
        
        self.drop_rate = 0.5
        self.n_filters = 64
        self.n_rand = 100
        self.batch_size = 32
        self.lr_D =  0.0002
        self.lr_GD = 0.0002
        self.real_l = 0.9
        self.fake_l = 0
        self.steps_D = 1
        self.steps_GD = 1