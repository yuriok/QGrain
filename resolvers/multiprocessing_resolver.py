
import time
from multiprocessing import Pool, cpu_count
from typing import List

from PySide2.QtCore import QObject, Signal

from data import GrainSizeData
from resolvers import FittingTask, HeadlessResolver


def run_task(task):
    resolver = HeadlessResolver()
    return resolver.execute_task(task)


class MultiProcessingResolver(QObject):
    sigTaskStateUpdated = Signal(tuple)
    sigTaskFinished = Signal(list, list)
    sigTaskInitialized = Signal(list)

    STATE_CHECK_TIME_INTERVAL = 0.1

    def __init__(self):
        super().__init__()
        self.grain_size_data = None # type: GrainSizeData
        self.tasks = None # type: List[FittingTask]
        self.pool = Pool(cpu_count())

    def on_data_loaded(self, data: GrainSizeData):
        if data is None:
            return
        elif not data.is_valid:
            return
        
        self.grain_size_data = data

    def init_tasks(self):
        tasks = []
        for i, sample_data in enumerate(self.grain_size_data.sample_data_list):
            task = FittingTask(i, sample_data.name, self.grain_size_data.classes, sample_data.distribution)
            tasks.append(task)
        self.tasks = tasks
        self.sigTaskInitialized.emit(tasks)

    def execute_tasks(self):
        if self.grain_size_data is None:
            return
        self.init_tasks()
        
        async_results = [(task.sample_id, self.pool.apply_async(run_task, args=(task,))) for task in self.tasks]
        
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
            flag, task, fitted_data = r.get()
            if flag:
                succeeded_results.append(fitted_data)
            else:
                failed_tasks.append(task)

        self.sigTaskFinished.emit(succeeded_results, failed_tasks)
