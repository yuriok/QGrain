import typing

# "cosine" has problem
built_in_distances = ("1-norm", "2-norm", "3-norm", "4-norm", "MSE", "log10MSE", "angular")
def check_distance(distance):
    assert isinstance(distance, str)
    in_set = False
    for d in built_in_distances:
        if distance == d:
            in_set = True
            break
    assert in_set

built_in_minimizers = ("Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr")
def check_minimizer(minimizer):
    assert isinstance(minimizer, str)
    in_set = False
    for m in built_in_minimizers:
        if minimizer == m:
            in_set = True
            break
    assert in_set


class ClassicResolverSetting:
    def __init__(self,
                 distance: str="log10MSE",
                 minimizer: str="SLSQP",
                 try_GO: bool=False,
                 GO_max_niter: int=100,
                 GO_success_niter: int=5, # GO - Global Optimization
                 GO_step: float=0.1,
                 GO_minimizer_tol: float=1e-6,
                 GO_minimizer_ftol: float=1e-8,
                 GO_minimizer_max_niter: int=500,
                 FLO_tol: float=1e-8, # FLO - Final Local Optimization
                 FLO_ftol: float=1e-10,
                 FLO_max_niter: int=1000):
        # validation
        check_distance(distance)
        check_minimizer(minimizer)
        assert isinstance(try_GO, bool)
        assert isinstance(GO_max_niter, int)
        assert isinstance(GO_success_niter, int)
        assert isinstance(GO_step, float)
        assert isinstance(GO_minimizer_tol, (int, float))
        assert isinstance(GO_minimizer_ftol, float)
        assert isinstance(GO_minimizer_max_niter, int)
        assert isinstance(FLO_tol, (int, float))
        assert isinstance(FLO_ftol, float)
        assert isinstance(FLO_max_niter, int)
        assert GO_max_niter > 0
        assert GO_success_niter > 0
        assert GO_step > 0.0
        # when distance="log10MSE"
        # the loss function may < 0
        # assert GO_minimizer_tol > 0.0
        assert 0.0 < GO_minimizer_ftol < 1.0
        assert GO_minimizer_max_niter > 0
        # assert FLO_tol > 0.0
        assert 0.0 < FLO_ftol < 1.0
        assert FLO_max_niter > 0

        self.distance = distance
        self.minimizer = minimizer
        self.try_GO = try_GO
        self.GO_max_niter = GO_max_niter
        self.GO_success_niter = GO_success_niter
        self.GO_step = GO_step
        self.GO_minimizer_tol = GO_minimizer_tol
        self.GO_minimizer_ftol = GO_minimizer_ftol
        self.GO_minimizer_max_niter = GO_minimizer_max_niter
        self.FLO_tol = FLO_tol
        self.FLO_ftol = FLO_ftol
        self.FLO_max_niter = FLO_max_niter

    def __str__(self) -> str:
        return self.__dict__.__str__()
