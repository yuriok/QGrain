__all__ = ["MultiProcessingResolver"]

import logging
import time
from enum import Enum, unique
from multiprocessing import Pool, cpu_count
from typing import Dict, List
from uuid import UUID, uuid4

import numpy as np
from PySide2.QtCore import QMutex, QObject, Signal

from QGrain.algorithms import DistributionType
from QGrain.models.AlgorithmSettings import AlgorithmSettings
from QGrain.models.FittingResult import FittingResult
from QGrain.resolvers.HeadlessResolver import FittingTask, HeadlessResolver


@unique
class ProcessState(Enum):
    NotStarted = 0
    Succeeded = 1
    Failed = 2

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
    task_state_updated = Signal(list, dict, dict)
    logger = logging.getLogger(name="root.resolvers.MultiProcessingResolver")
    STATE_CHECK_TIME_INTERVAL = 0.20
    def __init__(self):
        super().__init__()
        self.tasks = [] # type: List[FittingTask]
        self.states = {} # type: Dict[UUID, ProcessState]
        self.succeeded_results = {} # type: Dict[UUID, FittingResult]

        self.__pause_flag = False
        self.__cancel_flag = False
        self.__pause_mutex = QMutex()

    def on_task_generated(self, tasks: List[FittingTask]):
        assert tasks is not None
        states = {}
        for task in tasks:
            states[task.uuid] = ProcessState.NotStarted
        self.tasks.extend(tasks)
        self.states.update(states)

    def execute_tasks(self):
        assert self.tasks is not None
        assert self.states is not None
        assert self.succeeded_results is not None
        # setup the thread pool
        suggested_params = [(DistributionType.GeneralWeibull, 1), (DistributionType.GeneralWeibull, 2),
                            (DistributionType.GeneralWeibull, 3), (DistributionType.GeneralWeibull, 4)]
        pool = Pool(cpu_count(), setup_process, suggested_params)
        async_results = {task.uuid: pool.apply_async(run_task, args=(task,))
                         for task in self.tasks
                         if self.states[task.uuid] == ProcessState.NotStarted}
        def check_states():
            for task_id, result in async_results.items():
                if result.ready():
                    flag, task, fitting_result = result.get()
                    if flag:
                        self.states[task_id] = ProcessState.Succeeded
                        self.succeeded_results[task_id] = fitting_result
                    else:
                        self.states[task_id] = ProcessState.Failed
                else:
                    self.states[task_id] = ProcessState.NotStarted
            self.task_state_updated.emit(self.tasks, self.states, self.succeeded_results)

        while True:
            self.__pause_mutex.lock()
            # handle the pause request
            if self.__pause_flag:
                self.__pause_flag = False
                self.__pause_mutex.unlock()
                check_states()
                pool.terminate()
                pool.join()
                break
            else:
                self.__pause_mutex.unlock()
                time.sleep(self.STATE_CHECK_TIME_INTERVAL)
                check_states()
                # check if all tasks are finished
                if np.alltrue([state!=ProcessState.NotStarted for state in self.states.values()]):
                    pool.terminate()
                    pool.join()
                    break


    def pause_task(self):
        self.__pause_mutex.lock()
        self.__pause_flag = True
        self.__pause_mutex.unlock()


    def cleanup(self):
        self.tasks = []
        self.states = {}
        self.succeeded_results = {}

    def setup_all(self):
        pass

    def cleanup_all(self):
        pass
