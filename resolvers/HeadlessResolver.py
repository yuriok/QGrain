import numpy as np

from algorithms import DistributionType
from models.SampleData import SampleData
from resolvers.Resolver import Resolver


class FittingTask:
    def __init__(self, sample: SampleData,
                 distribution_type=DistributionType.Weibull,
                 component_number=2,
                 algorithm_settings=None):
        self.sample = sample
        self.component_number = component_number
        self.distribution_type = distribution_type
        self.algorithm_settings = algorithm_settings


class HeadlessResolver(Resolver):
    def __init__(self):
        super().__init__()
        self.current_task = None
        self.current_result = None

    def on_fitting_succeeded(self, algorithm_result):
        result = self.get_fitting_result(algorithm_result.x)
        sample_id = self.current_task.sample.uuid
        self.current_result = (sample_id, result)

    def on_exception_raised_while_fitting(self, exception: Exception):
        print(exception)

    def execute_task(self, task: FittingTask):
        self.current_task = task
        self.distribution_type = task.distribution_type
        self.component_number = task.component_number
        if task.algorithm_settings is not None:
            self.change_settings(**task.algorithm_settings)
        self.feed_data(task.sample.name, task.sample.classes, task.sample.distribution)
        self.try_fit()

        if self.current_result is None or \
                self.current_result[0] != self.current_task.sample.uuid:
            return False, self.current_task, None
        else:
            return True, self.current_task, self.current_result[1]
