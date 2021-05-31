import time
import typing
from enum import Enum, unique
from multiprocessing import Manager, Queue, freeze_support, set_start_method
from multiprocessing.managers import DictProxy, ListProxy, Value

from QGrain.algorithms.BaseDistribution import BaseDistribution
from QGrain.algorithms.DistributionType import DistributionType
from QGrain.models.FittingResult import SSUResult
from QGrain.models.FittingTask import SSUTask
from QGrain.models.ProcessedSampleData import ProcessedSampleData
from QGrain.resolvers.BaseResolver import BaseResolver, FittingState
from scipy.optimize import OptimizeResult


@unique
class FittingCommand(Enum):
    Normal = 0,
    Suspend = 1,
    Break = 2

class HeadlessResolver(BaseResolver):
    def __init__(self, task_queue: Queue,
                 task_state: DictProxy,
                 failed_queue: Queue,
                 succeeded_results: DictProxy,
                 command_value: Value):
        default_distribution = BaseDistribution.get_distribution(DistributionType.Normal, 3)
        super().__init__(default_distribution)
        self.task_queue = task_queue
        self.task_state = task_state
        self.failed_queue = failed_queue
        self.succeeded_results = succeeded_results
        self.command_value = command_value
        self.__current_task = None

    # hooks
    def on_data_fed(self, processed_sample: ProcessedSampleData):
        pass

    def on_data_not_prepared(self):
        self.task_state[self.__current_task.uuid] = FittingState.DataNotAvailable

    def on_fitting_started(self):
        self.task_state[self.__current_task.uuid] = FittingState.Fitting

    def on_fitting_finished(self):
        pass

    def on_global_fitting_failed(self, algorithm_result: OptimizeResult):
        self.task_state[self.__current_task.uuid] = FittingState.GlobalOptimizationFailed

    def on_global_fitting_succeeded(self, algorithm_result: OptimizeResult):
        pass

    def on_final_fitting_failed(self, algorithm_result: OptimizeResult):
        self.task_state[self.__current_task.uuid] = FittingState.FinalOptimizationFailed

    def on_exception_raised_while_fitting(self, exception: Exception):
        self.task_state[self.__current_task.uuid] = FittingState.UnknownError

    def local_iteration_callback(self, fitted_params: typing.Iterable[float]):
        pass

    def global_iteration_callback(self, fitted_params: typing.Iterable[float], function_value: float, accept: bool):
        pass

    def on_fitting_succeeded(self, algorithm_result: OptimizeResult, fitting_result: SSUResult):
        self.task_state[self.__current_task.uuid] = FittingState.Succeeded

    def execute_task(self, task: SSUTask):
        self.__current_task = task
        self.change_settings(task.algorithm_settings)
        self._distribution = BaseDistribution.get_distribution(task.distribution_type, task.component_number)
        self.feed_data(task.sample)
        state, result = self.try_fit(reference_params=task.reference_params, mean_values=task.mean_values)
        if state == FittingState.Succeeded:
            self.succeeded_results[task.uuid] = result
        else:
            print(state, result)
            self.failed_queue.put(task)

    def start_mainloop(self):
        while True:
            command = self.command_value.get()
            assert isinstance(command, FittingCommand)
            if command == FittingCommand.Normal:
                task_to_execute = self.task_queue.get()
                self.execute_task(task_to_execute)
            elif command == FittingCommand.Suspend:
                time.sleep(0.1)
            elif command == FittingCommand.Break:
                break

def process_setup(task_queue: Queue,
           task_state: DictProxy,
           failed_queue: Queue,
           succeeded_results: DictProxy,
           command_value: Value):
    freeze_support()
    resolver = HeadlessResolver(task_queue, task_state, failed_queue, succeeded_results, command_value)
    resolver.start_mainloop()


if __name__ == "__main__":
    freeze_support()
    # set_start_method('spawn')
    import sys
    from multiprocessing import Process

    from PySide2.QtWidgets import QApplication
    from QGrain.models.GrainSizeSample import GrainSizeSample
    from QGrain.ui.SampleGenerateSettingWidget import \
        RandomGeneratorWidget
    app = QApplication(sys.argv)
    generator = RandomGeneratorWidget()

    m = Manager()

    task_queue = m.Queue()
    task_state = m.dict()
    failed_queue = m.Queue()
    succeeded_results = m.dict()
    command_value = m.Value(FittingCommand, FittingCommand.Normal)

    task_count = 0

    def add_task():
        artificial_sample = generator.get_random_sample()
        sample_data = GrainSizeSample("Generated Sample", artificial_sample.classes_μm, artificial_sample.classes_φ, artificial_sample.distribution)
        task = SSUTask(sample_data, DistributionType.Normal, 3)
        task_queue.put(task)
        global task_count
        task_count += 1

    p_list = []
    for i in range(8):
        p = Process(target=process_setup, args=(task_queue, task_state, failed_queue, succeeded_results, command_value))
        p.start()
        p_list.append(p)

    for i in range(10):
        add_task()

    while True:
        time.sleep(1)
        print(f"Task Count: {task_count}; Not Obtained Task Number: {task_queue.qsize()}; Finished Task Number: {len(task_state)}")
        # for task_id, result in succeeded_results.items():
        #     print(f"Task [{task_id}] spent {result.time_spent} seconds")

        if task_queue.qsize() == 0:
            command_value.set(FittingCommand.Suspend)
            for i in range(10):
                add_task()

