class UDMAlgorithmSetting:
    def __init__(self,
                 device="cpu",
                 pretrain_epochs=400,
                 min_epochs=200,
                 max_epochs=2000,
                 precision=4.0,
                 learning_rate=1e-2,
                 betas=(0.8, 0.5),
                 constraint_level=3.0):
        assert isinstance(device, str)
        assert isinstance(pretrain_epochs, int)
        assert isinstance(min_epochs, int)
        assert isinstance(max_epochs, int)
        assert isinstance(precision, (int, float))
        assert isinstance(learning_rate, (int, float))
        assert isinstance(betas, tuple)
        assert len(betas) == 2
        beta1, beta2 = betas
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)
        assert isinstance(constraint_level, (int, float))
        assert pretrain_epochs >= 0
        assert min_epochs > 0
        assert max_epochs > 0
        assert 1.0 < precision < 100.0
        assert learning_rate > 0.0
        assert 0.0 < beta1 < 1.0
        assert 0.0 < beta2 < 1.0
        assert -20 < constraint_level < 20.0
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.precision = precision
        self.learning_rate = learning_rate
        self.betas = betas
        self.constraint_level = constraint_level

    def __str__(self) -> str:
        return self.__dict__.__str__()
