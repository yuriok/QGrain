built_in_distances = ("1-norm", "2-norm", "3-norm", "4-norm", "MSE", "log10MSE", "angular")

def check_distance(distance):
    assert isinstance(distance, str)
    in_set = False
    for d in built_in_distances:
        if distance == d:
            in_set = True
            break
    assert in_set

class EMMAAlgorithmSetting:
    def __init__(self,
                 device="cpu",
                 distance="log10MSE",
                 pretrain_epochs=0,
                 min_epochs=100,
                 max_epochs=10000,
                 precision=6,
                 learning_rate=5e-3,
                 betas=(0.8, 0.5)):
        assert isinstance(device, str)
        check_distance(distance)
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
        assert pretrain_epochs >= 0
        assert min_epochs > 0
        assert max_epochs > 0
        assert 1.0 < precision < 100.0
        assert learning_rate > 0.0
        assert 0.0 < beta1 < 1.0
        assert 0.0 < beta2 < 1.0
        self.device = device
        self.distance = distance
        self.pretrain_epochs = pretrain_epochs
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.precision = precision
        self.learning_rate = learning_rate
        self.betas = betas

    def __str__(self) -> str:
        return self.__dict__.__str__()
