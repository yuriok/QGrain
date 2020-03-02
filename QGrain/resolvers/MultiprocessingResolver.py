__all__ = ["MultiProcessingResolver"]

import logging
import time
from multiprocessing import Pool, cpu_count
from typing import List

from PySide2.QtCore import QObject, Signal

from QGrain.algorithms import DistributionType
from QGrain.models.SampleDataset import SampleDataset
from QGrain.resolvers.HeadlessResolver import FittingTask, HeadlessResolver


def run_task(task):
    global resolver
    results = resolver.execute_task(task)
    return results

def setup_process(*args):
    global resolver
    resolver = HeadlessResolver()
    for distribution_type, component_number in args:
        resolver.distribution_type = distribution_type
        resolver.component_number = component_number

class MultiProcessingResolver(QObject):
    sigTaskStateUpdated = Signal(tuple)
    sigTaskFinished = Signal(list, list)
    sigTaskInitialized = Signal(list)
    logger = logging.getLogger(name="root.resolvers.MultiProcessingResolver")
    STATE_CHECK_TIME_INTERVAL = 0.1

    def __init__(self):
        super().__init__()
        self.component_number = 3
        self.distribution_type = DistributionType.GeneralWeibull
        self.algorithm_settings = None

        self.grain_size_data = None # type: SampleDataset
        self.tasks = None # type: List[FittingTask]


    def on_component_number_changed(self, component_number: int):
        self.component_number = component_number
        self.logger.info("Component number has been changed to [%d].", self.component_number)

    def on_distribution_type_changed(self, distribution_type: DistributionType):
        self.distribution_type = DistributionType(distribution_type)
        self.logger.info("Distribution type has been changed to [%s].", self.distribution_type)

    def on_algorithm_settings_changed(self, settings: dict):
        self.algorithm_settings = settings
        self.logger.info("Algorithm settings have been changed to [%s].", settings)

    def on_data_loaded(self, data: SampleDataset):
        if data is None:
            return
        elif not data.has_data:
            return

        self.grain_size_data = data

    def init_tasks(self):
        tasks = []
        for sample in self.grain_size_data.samples:
            task = FittingTask(
                sample,
                component_number=self.component_number,
                distribution_type=self.distribution_type,
                algorithm_settings=self.algorithm_settings)
            tasks.append(task)
        self.tasks = tasks
        self.sigTaskInitialized.emit(tasks)

    def execute_tasks(self):
        if self.grain_size_data is None:
            return
        self.init_tasks()

        async_results = [(task.sample.uuid, self.pool.apply_async(run_task, args=(task,))) for task in self.tasks]

        while True:
            time.sleep(self.STATE_CHECK_TIME_INTERVAL)
            task_states = []
            for sample_id, result in async_results:
                task_states.append((sample_id, result.ready()))
            all_ready = True
            for sample_id, ready in task_states:
                if not ready:
                    all_ready = False
                    break
            self.sigTaskStateUpdated.emit(task_states)
            if all_ready:
                break

        succeeded_results = []
        failed_tasks = []
        for sample_id, r in async_results:
            flag, task, fitting_result = r.get()
            if flag:
                succeeded_results.append(fitting_result)
            else:
                failed_tasks.append(task)

        self.sigTaskFinished.emit(succeeded_results, failed_tasks)


    def setup_all(self):
        suggested_params = [(DistributionType.GeneralWeibull, 1), (DistributionType.GeneralWeibull, 2),
                            (DistributionType.GeneralWeibull, 3), (DistributionType.GeneralWeibull, 4)]
        self.pool = Pool(cpu_count(), setup_process, suggested_params)

    def cleanup_all(self):
        self.pool.terminate()
        self.pool.join()
