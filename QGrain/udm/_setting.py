class UDMResolverSetting:
    def __init__(self,
                 device="cpu",
                 pretrain_epochs=400,
                 min_epochs=200,
                 max_epochs=2000,
                 min_delta_loss=1e-4,
                 lr=1e-2,
                 betas=(0.8, 0.5),
                 constraint_weight=1000.0):
        assert isinstance(device, str)
        assert isinstance(pretrain_epochs, int)
        assert isinstance(min_epochs, int)
        assert isinstance(max_epochs, int)
        assert isinstance(min_delta_loss, float)
        assert isinstance(lr, float)
        assert isinstance(betas, tuple)
        assert len(betas) == 2
        beta1, beta2 = betas
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)
        assert isinstance(constraint_weight, (int, float))
        assert pretrain_epochs >= 0
        assert min_epochs > 0
        assert max_epochs > 0
        assert 0.0 < min_delta_loss < 1.0
        assert lr > 0.0
        assert 0.0 < beta1 < 1.0
        assert 0.0 < beta2 < 1.0
        assert constraint_weight > 0.0
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.min_delta_loss = min_delta_loss
        self.lr = lr
        self.betas = betas
        self.constraint_weight = constraint_weight

    def __str__(self) -> str:
        return self.__dict__.__str__()
