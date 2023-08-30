from torch.nn import L1Loss, MSELoss


class L2:
    def __init__(self):
        super(L2, self).__init__()
        self.loss_ = MSELoss()

    def __call__(self, x, x_recon, z=None):
        return self.loss_(x, x_recon)


class L1:
    def __init__(self):
        super(L1, self).__init__()
        self.loss_ = L1Loss()

    def __call__(self, x, x_recon, z=None):
        return self.loss_(x, x_recon)