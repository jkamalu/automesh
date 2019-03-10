class Params:

    drop_rate = 0.5
    n_filters = 32
    n_rand = 100
    batch_size = 32
    lr_D =  0.0002
    lr_GD = 0.0002
    real_l = 0.9
    fake_l = 0
    steps_D = 1
    steps_GD = 1
    
    def __init__(self, X, X_cond=None):
        _, self.W, self.H, self.n_channels = X.shape
        _, self.n_cond = X_cond.shape if X_cond is not None else (None, 0)