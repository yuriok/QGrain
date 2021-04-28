import traceback

from PySide2.QtCore import QObject, QThread, Signal, Slot
from QGrain.algorithms import FittingState
from QGrain.algorithms.ClassicResolver import ClassicResolver
from QGrain.algorithms.NNResolver import NNResolver
from QGrain.models.FittingResult import FittingResult
from QGrain.models.FittingTask import FittingTask


class BackgroundWorker(QObject):
    task_succeeded = Signal(FittingResult)
    task_failed = Signal(str, FittingTask)

    def __init__(self):
        super().__init__()

    def on_task_started(self, task: FittingTask):
        if task.resolver == "classic":
            resolver = ClassicResolver()
        elif task.resolver == "neural":
            resolver = NNResolver()
        else:
            raise NotImplementedError(task.resolver)
        try:
            state, result = resolver.try_fit(task)
            if state == FittingState.Succeeded:
                self.task_succeeded.emit(result)
            else:
                self.task_failed.emit(f"Fitting Failed, error details:\n{result.__str__()}", task)
        except Exception as e:
            self.task_failed.emit(f"Unknown Exception Raised: {type(e)}, {e.__str__()}, {traceback.format_exc()}", task)

class AsyncFittingWorker(QObject):
    task_started = Signal(FittingTask)

    def __init__(self):
        super().__init__()
        self.background_worker = BackgroundWorker()
        self.working_thread = QThread()
        self.background_worker.moveToThread(self.working_thread)
        self.task_started.connect(self.background_worker.on_task_started)
        self.background_worker.task_failed.connect(self.on_task_failed)
        self.background_worker.task_succeeded.connect(self.on_task_succeeded)
        self.working_thread.start()

    def on_task_succeeded(self, fitting_result: FittingResult):
        pass

    def on_task_failed(self, failed_info, task):
        pass

    @Slot()
    def execute_task(self, task: FittingTask):
        self.task_started.emit(task)

if __name__ == "__main__":
    import sys

    from PySide2.QtWidgets import QApplication
    from QGrain.algorithms import DistributionType
    from QGrain.charts.MixedDistributionChart import MixedDistributionChart
    from QGrain.models.artificial import get_random_sample
    from QGrain.models.NNResolverSetting import NNResolverSetting
    app = QApplication(sys.argv)
    canvas = MixedDistributionChart()
    canvas.show_demo()

    worker = AsyncFittingWorker()
    def show(result: FittingResult):
        print(result.sample.name)
        print(result.n_iterations)
        # canvas.show_models(result.view_models, repeat=False)
        canvas.show_model(result.view_model)

    worker.background_worker.task_succeeded.connect(show)
    reference = [dict(mean=10, std=1.0, skewness=0.0),
                 dict(mean=7, std=1.0, skewness=0.0),
                 dict(mean=5, std=1.0, skewness=0.0)]
    # reference = None
    from PySide2.QtCore import QTimer
    timer = QTimer()
    timer.setSingleShot(True)
    def do_test():
        sample = get_random_sample()
        task = FittingTask(sample.sample_to_fit, DistributionType.Normal, 3,
                           resolver_setting=NNResolverSetting(min_niter=1000, max_niter=3000),
                           reference=reference, resolver="neural")
        worker.execute_task(task)
        timer.start(1000)
    timer.timeout.connect(do_test)
    timer.start(100)
    canvas.show()
    sys.exit(app.exec_())
