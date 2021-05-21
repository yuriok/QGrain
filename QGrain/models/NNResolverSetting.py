built_in_distances = ("1-norm", "2-norm", "3-norm", "4-norm", "MSE", "log10MSE", "angular")

def check_distance(distance):
    assert isinstance(distance, str)
    in_set = False
    for d in built_in_distances:
        if distance == d:
            in_set = True
            break
    assert in_set

class NNResolverSetting:
    def __init__(self,
                 device="cpu",
                 distance="log10MSE",
                 min_niter=100,
                 max_niter=10000,
                 tol=1e-8,
                 ftol=1e-10,
                 lr=15e-3,
                 eps=1e-8):
        assert isinstance(device, str)
        check_distance(distance)
        assert isinstance(min_niter, int)
        assert isinstance(max_niter, int)
        assert isinstance(tol, (int, float))
        assert isinstance(ftol, float)
        assert isinstance(lr, float)
        assert isinstance(eps, float)
        assert min_niter > 0
        assert max_niter > 0
        assert 0.0 < ftol < 1.0
        assert lr > 0.0
        assert 0.0 < eps < 1.0
        self.device = device
        self.distance = distance
        self.min_niter = min_niter
        self.max_niter = max_niter
        self.tol = tol
        self.ftol = ftol
        self.lr = lr
        self.eps = eps

    def __str__(self) -> str:
        return self.__dict__.__str__()
