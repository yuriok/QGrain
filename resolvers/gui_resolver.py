import logging
import time

import numpy as np
from PySide2.QtCore import QMutex, QObject, Signal, Slot

from algorithms import DistributionType
from data import FittedData
from resolvers import DataValidationResult, Resolver


class CancelError(Exception):
    pass


class GUIResolver(QObject, Resolver):
    sigSingleIterationFinished = Signal(int, FittedData)
    sigFittingEpochSucceeded = Signal(FittedData)
    sigWidgetsEnable = Signal(bool)
    sigFittingFailed = Signal(str) # emit hint text
    logger = logging.getLogger(name="root.resolvers.GUIResolver")

    def __init__(self, inherit_params=True, emit_iteration=False, time_interval=0.05):
        super().__init__()
        Resolver.__init__(self)

        # settings
        self.inherit_params = inherit_params
        self.last_succeeded_params = None
        self.emit_iteration = emit_iteration
        self.time_interval = time_interval
        
        self.current_iteration = 0
        
        self.cancel_flag = False
        self.cancel_mutex = QMutex()

    def on_component_number_changed(self, component_number: int):
        self.component_number = component_number
        self.logger.info("Component Number has been changed to [%d].", component_number)

    def on_distribution_type_changed(self, distribution_type: str):
        if distribution_type == "normal":
            self.distribution_type = DistributionType.Normal
        elif distribution_type == "weibull":
            self.distribution_type = DistributionType.Weibull
        else:
            raise NotImplementedError(distribution_type)
        self.logger.info("Distribution type has been changed to [%s].", self.distribution_type)

    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            setattr(self, setting, value)
            self.logger.info("Setting [%s] have been changed to [%s].", setting, value)

    def on_algorithm_settings_changed(self, settings: dict):
        self.change_settings(**settings)
        self.logger.info("Algorithm settings have been changed to [%s].", settings)

    def on_target_data_changed(self, sample_name: str, x, y):
        self.logger.debug("Target data has been changed to [%s].", sample_name)
        self.feed_data(sample_name, x, y)

    def on_data_invalid(self, sample_name: str, x: np.ndarray, y:np.ndarray, result: DataValidationResult):
        if result == DataValidationResult.NameNone:
            self.sigFittingFailed(self.tr("Name of sample [%s] is None."), sample_name)
            self.logger.error("Name of sample [%s] is None.", sample_name)
        elif result == DataValidationResult.NameEmpty:
            self.sigFittingFailed(self.tr("Name of sample [%s] is empty."), sample_name)
            self.logger.error("Name of sample [%s] is empty.", sample_name)
        elif result == DataValidationResult.XNone:
            self.sigFittingFailed(self.tr("x data of sample [%s] is None."), sample_name)
            self.logger.error("x data of sample [%s] is None.", sample_name)
        elif result == DataValidationResult.YNone:
            self.sigFittingFailed(self.tr("y data of sample [%s] is None."), sample_name)
            self.logger.error("y data of sample [%s] is None.", sample_name)
        elif result == DataValidationResult.XTypeInvalid:
            self.sigFittingFailed(self.tr("The x data type of sample [%s] is invalid."), sample_name)
            self.logger.error("The x data type of sample [%s] is invalid.", sample_name)
        elif result == DataValidationResult.YTypeInvalid:
            self.sigFittingFailed(self.tr("The y data type of sample [%s] is invalid."), sample_name)
            self.logger.error("The y data type of sample [%s] is invalid.", sample_name)
        elif result == DataValidationResult.LengthNotEqual:
            self.sigFittingFailed(self.tr("The lengths of x and y data in sample [%s] are not equal."), sample_name)
            self.logger.error("The lengths of x and y data in sample [%s] are not equal.", sample_name)
        elif result == DataValidationResult.XHasNan:
            self.sigFittingFailed(self.tr("There is NaN value in x data of sample [%s]."), sample_name)
            self.logger.error("There is NaN value in x data of sample [%s].", sample_name)
        elif result == DataValidationResult.YHasNan:
            self.sigFittingFailed(self.tr("There is NaN value in y data of sample [%s]."), sample_name)
            self.logger.error("There is NaN value in y data of sample [%s].", sample_name)
        else:
            raise NotImplementedError(result)

    def on_data_feed(self, sample_name):
        self.logger.debug("Sample [%s] has been fed.", sample_name)

    def on_data_not_prepared(self):
        self.sigFittingFailed(self.tr("There is no valid data to fit."))
        self.logger.error("There is no valid data to fit.")

    def on_fitting_started(self):
        self.current_iteration = 0
        if self.inherit_params and self.last_succeeded_params is not None and len(self.last_succeeded_params) == len(self.mixed_data.defaults):
            self.initial_guess = self.last_succeeded_params
        else:
            self.initial_guess = self.mixed_data.defaults

        self.sigWidgetsEnable.emit(False)
        self.logger.debug("Fitting progress started.")

    def on_fitting_finished(self):
        self.current_iteration = 0
        self.sigWidgetsEnable.emit(True)
        self.logger.debug("Fitting progress finished.")

    def on_global_fitting_failed(self, fitted_result):
        self.sigFittingFailed(self.tr("Fitting failed during global fitting progress."))
        self.logger.error("Fitting failed during global fitting progress. Details: [%s].", fitted_result)

    def on_final_fitting_failed(self, fitted_result):
        self.sigFittingFailed(self.tr("Fitting failed during final fitting progress."))
        self.logger.error("Fitting failed during final fitting progress. Details: [%s].", fitted_result)

    def on_exception_raised_while_fitting(self, exception):
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
            self.sigSingleIterationFinished.emit(self.current_iteration, self.get_fitted_data(fitted_params))
        self.current_iteration += 1

    def global_iteration_callback(self, fitted_params, function_value, accept):
        pass

    def on_fitting_succeeded(self, fitted_result):
        if self.inherit_params:
            self.last_succeeded_params = fitted_result.x
        self.logger.info("The epoch of fitting has finished, the fitted parameters are: [%s]", fitted_result.x)
        self.sigFittingEpochSucceeded.emit(self.get_fitted_data(fitted_result.x))

    # call this func by another thread
    # so, lock is necessary
    def cancel(self):
        self.cancel_mutex.lock()
        self.cancel_flag = True
        self.cancel_mutex.unlock()
