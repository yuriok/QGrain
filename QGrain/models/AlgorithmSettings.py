import typing


class AlgorithmSettings:
    def __init__(self, global_optimization_maximum_iteration: int = 100,
                 global_optimization_success_iteration: int = 3,
                 global_optimization_step_size: typing.Union[int, float] = 1.0,
                 global_optimization_minimizer_tolerance_level: typing.Union[int, float] = 8,
                 global_optimization_minimizer_maximum_iteration: int = 500,
                 final_optimization_minimizer_tolerance_level=100,
                 final_optimization_minimizer_maximum_iteration=1000):

        # validation
        assert isinstance(global_optimization_maximum_iteration, int)
        assert isinstance(global_optimization_success_iteration, int)
        assert isinstance(global_optimization_step_size, (int, float))
        assert isinstance(
            global_optimization_minimizer_tolerance_level, (int, float))
        assert isinstance(global_optimization_minimizer_maximum_iteration, int)
        assert isinstance(
            final_optimization_minimizer_tolerance_level, (int, float))
        assert isinstance(final_optimization_minimizer_maximum_iteration, int)
        assert global_optimization_maximum_iteration > 0
        assert global_optimization_success_iteration > 0
        assert global_optimization_step_size > 0.0
        assert global_optimization_minimizer_tolerance_level > 0.0
        assert global_optimization_minimizer_maximum_iteration > 0
        assert final_optimization_minimizer_tolerance_level > 0.0
        assert final_optimization_minimizer_maximum_iteration > 0

        self.__global_optimization_maximum_iteration = global_optimization_maximum_iteration
        self.__global_optimization_success_iteration = global_optimization_success_iteration
        self.__global_optimization_step_size = global_optimization_step_size
        self.__global_optimization_minimizer_tolerance_level = global_optimization_minimizer_tolerance_level
        self.__global_optimization_minimizer_maximum_iteration = global_optimization_minimizer_maximum_iteration
        self.__final_optimization_minimizer_tolerance_level = final_optimization_minimizer_tolerance_level
        self.__final_optimization_minimizer_maximum_iteration = final_optimization_minimizer_maximum_iteration

    @property
    def global_optimization_maximum_iteration(self):
        return self.__global_optimization_maximum_iteration

    @property
    def global_optimization_success_iteration(self):
        return self.__global_optimization_success_iteration

    @property
    def global_optimization_step_size(self):
        return self.__global_optimization_step_size

    @property
    def global_optimization_minimizer_tolerance_level(self):
        return self.__global_optimization_minimizer_tolerance_level

    @property
    def global_optimization_minimizer_maximum_iteration(self):
        return self.__global_optimization_minimizer_maximum_iteration

    @property
    def final_optimization_minimizer_tolerance_level(self):
        return self.__final_optimization_minimizer_tolerance_level

    @property
    def final_optimization_minimizer_maximum_iteration(self):
        return self.__final_optimization_minimizer_maximum_iteration
