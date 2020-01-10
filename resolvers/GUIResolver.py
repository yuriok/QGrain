import logging
import time

import numpy as np
from PySide2.QtCore import QMutex, QObject, Signal, Slot
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult

from algorithms import DistributionType
from models.FittingResult import FittingResult
from models.SampleData import SampleData
from resolvers.Resolver import Resolver


class CancelError(Exception):
    pass


class GUIResolver(QObject, Resolver):
    sigSingleIterationFinished = Signal(int, FittingResult)
    sigFittingEpochSucceeded = Signal(FittingResult)
    sigWidgetsEnable = Signal(bool)
    sigFittingFailed = Signal(str) # emit hint text
    logger = logging.getLogger(name="root.resolvers.GUIResolver")

    def __init__(self, inherit_params=True, emit_iteration=False, time_interval=0.05):
        super().__init__()
        Resolver.__init__(self)
        # settings
        self.expected_params = None
        self.inherit_params = inherit_params
        self.last_succeeded_params = None
        self.emit_iteration = emit_iteration
        self.time_interval = time_interval

        self.current_iteration = 0

        self.cancel_flag = False
        self.cancel_mutex = QMutex()

    def on_component_number_changed(self, component_number: int):
        self.component_number = component_number
        self.logger.info("Component Number has been changed to [%d].", self.component_number)

    def on_distribution_type_changed(self, distribution_type: DistributionType):
        self.distribution_type = DistributionType(distribution_type)
        self.logger.info("Distribution type has been changed to [%s].", self.distribution_type)
        # clear if type changed
        self.last_succeeded_params = None

    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            setattr(self, setting, value)
            self.logger.info("Setting [%s] have been changed to [%s].", setting, value)

    def on_algorithm_settings_changed(self, settings: dict):
        self.change_settings(**settings)
        self.logger.info("Algorithm settings have been changed to [%s].", settings)

    def on_target_data_changed(self, sample: SampleData):
        self.logger.debug("Target data has been changed to [%s].", sample.name)
        self.feed_data(sample)

    def on_data_fed(self, sample_name: str):
        self.logger.debug("Sample [%s] has been fed.", sample_name)

    def on_data_not_prepared(self):
        self.sigFittingFailed.emit(self.tr("There is no valid data to fit."))
        self.logger.error("There is no valid data to fit.")

    def on_fitting_started(self):
        self.current_iteration = 0
        if self.inherit_params and \
            self.last_succeeded_params is not None and \
            len(self.last_succeeded_params) == len(self.algorithm_data.defaults):
            self.initial_guess = self.last_succeeded_params
        else:
            self.initial_guess = self.algorithm_data.defaults

        if self.expected_params is not None:
            self.initial_guess = self.expected_params

        self.sigWidgetsEnable.emit(False)
        self.logger.debug("Fitting progress started.")

    def on_fitting_finished(self):
        self.current_iteration = 0
        self.expected_params = None
        self.sigWidgetsEnable.emit(True)
        self.logger.debug("Fitting progress finished.")

    def on_global_fitting_failed(self, algorithm_result: OptimizeResult):
        self.sigFittingFailed.emit(self.tr("Fitting failed during global fitting progress."))
        self.logger.error("Fitting failed during global fitting progress. Details: [%s].", algorithm_result.message)

    def on_global_fitting_succeeded(self, algorithm_result: OptimizeResult):
        self.logger.debug("Global fitting progress succeeded.")

    def on_final_fitting_failed(self, algorithm_result: OptimizeResult):
        self.sigFittingFailed.emit(self.tr("Fitting failed during final fitting progress."))
        self.logger.error("Fitting failed during final fitting progress. Details: [%s].", algorithm_result.message)

    def on_exception_raised_while_fitting(self, exception: Exception):
        if type(exception) == CancelError:
            self.logger.info("The fitting progress was canceled by user.")
        else:
            self.sigFittingFailed.emit(self.tr("Unknown exception raise in fitting progress."))
            self.logger.exception("Unknown exception raise in fitting progress.", stack_info=True)

    def local_iteration_callback(self, fitted_params):
        self.cancel_mutex.lock()
        if self.cancel_flag:
            self.cancel_flag = False
            self.cancel_mutex.unlock()
            # use exception to stop the fitting progress
            # need to handle the exception in `on_exception_raised_while_fitting` func
            raise CancelError()
        else:
            self.cancel_mutex.unlock()

        if self.emit_iteration:
            time.sleep(self.time_interval)
            fitting_result = self.get_fitting_result(fitted_params)
            self.sigSingleIterationFinished.emit(self.current_iteration, fitting_result)
        self.current_iteration += 1

    def global_iteration_callback(self, fitted_params, function_value, accept):
        pass

    def on_fitting_succeeded(self, fitting_result):
        if self.inherit_params:
            self.last_succeeded_params = fitting_result.x
        self.logger.info("The epoch of fitting has finished, the fitted parameters are: [%s]", fitting_result.x)
        self.sigFittingEpochSucceeded.emit(self.get_fitting_result(fitting_result.x))

    # call this func by another thread
    # so, lock is necessary
    def cancel(self):
        self.cancel_mutex.lock()
        self.cancel_flag = True
        self.cancel_mutex.unlock()

    def on_excepted_mean_value_changed(self, mean_values):
        if not self.data_prepared:
            return
        x_real_to_space = interp1d(self.real_x, self.fitting_space_x)
        converted_x = [x_real_to_space(mean).max() - self.x_offset for mean in mean_values]
        converted_x.sort()
        self.expected_params = self.algorithm_data.get_param_by_mean(converted_x)
