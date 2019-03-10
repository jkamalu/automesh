class Params:

    drop_rate = 0.4
    n_filters = 32
    n_rand = 100
    batch_size = 64
    lr_D =  0.0001
    lr_GD = 0.0002
    real_l = 1
    fake_l = 0
    steps_D = 1
    steps_GD = 2
    beta_1 = 0.5
    
    def __init__(self, X, X_cond=None):
        _, self.W, self.H, self.n_channels = X.shape
        _, self.n_cond = X_cond.shape if X_cond else (None, 0)