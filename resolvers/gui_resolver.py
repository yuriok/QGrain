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
    logger = logging.getLogger(name="root.GUIResolver")

    def __init__(self, auto_fit=True, inherit_params=True, emit_iteration=False, time_interval=0.05):
        super().__init__()
        Resolver.__init__(self)

        # settings
        self.auto_fit = auto_fit
        self.inherit_params = inherit_params
        self.emit_iteration = emit_iteration
        self.time_interval = time_interval
        
        self.current_iteration = 0
        
        self.cancel_flag = False
        self.cancel_mutex = QMutex()

    def on_ncomp_changed(self, ncomp: int):
        self.ncomp = ncomp
        self.logger.debug("Component Number has been changed to [%d].", ncomp)

    def on_type_changed(self, distribution_type: DistributionType):
        self.distribution_type = distribution_type
        self.logger.debug("Distribution type has been changed to [%s].", distribution_type)

    def on_settings_changed(self, kwargs: dict):
        for setting, value in kwargs.items():
            setattr(self, setting, value)
            self.logger.debug("Setting [%s] have been changed to [%s].", setting, value)

    def on_target_data_changed(self, sample_name, x, y):
        self.logger.debug("Target data has been changed to [%s].", sample_name)

        self.feed_data(sample_name, x, y)

        if self.auto_fit:
            self.try_fit()

    def on_data_invalid(self, x, y, validation_result):
        if validation_result == DataValidationResult.NameNone:
            self.sigFittingFailed(self.tr("Name of current sample is `None`."))
            self.logger.error("Name of current sample is `None`.")
        elif validation_result == DataValidationResult.NameEmpty:
            self.sigFittingFailed(self.tr("Name of current sample is empty."))
            self.logger.error("Name of current sample is empty.")
        elif validation_result == DataValidationResult.XNone:
            self.sigFittingFailed(self.tr("x data of current sample is `None`."))
            self.logger.error("x data of current sample is `None`.")
        elif validation_result == DataValidationResult.YNone:
            self.sigFittingFailed(self.tr("y data of current sample is `None`."))
            self.logger.error("y data of current sample is `None`.")
        elif validation_result == DataValidationResult.XTypeInvalid:
            self.sigFittingFailed(self.tr("The x data type of current sample is invalid."))
            self.logger.error("The x data type of current sample is invalid.")
        elif validation_result == DataValidationResult.YTypeInvalid:
            self.sigFittingFailed(self.tr("The y data type of current sample is invalid."))
            self.logger.error("The y data type of current sample is invalid.")
        elif validation_result == DataValidationResult.LengthNotEqual:
            self.sigFittingFailed(self.tr("The lengths of x and y data are not equal."))
            self.logger.error("The lengths of x and y data are not equal.")
        elif validation_result == DataValidationResult.XHasNan:
            self.sigFittingFailed(self.tr("There is `Nan` value in x data of current sample."))
            self.logger.error("There is `Nan` value in x data of current sample.")
        elif validation_result == DataValidationResult.YHasNan:
            self.sigFittingFailed(self.tr("There is `Nan` value in y data of current sample."))
            self.logger.error("There is `Nan` value in y data of current sample.")
        else:
            raise NotImplementedError(validation_result)

    def on_data_feed(self, sample_name):
        self.logger.debug("Sample [%s] has been fed.", sample_name)

    def on_data_not_prepared(self):
        self.sigFittingFailed(self.tr("There is no valid data to fit."))
        self.logger.error("There is no valid data to fit.")

    def on_fitting_started(self):
        self.current_iteration = 0
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
            self.sigFittingFailed(self.tr("Unknown exception raise in fitting progress."))
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

    def on_fitting_succeed(self, fitted_result):
        if self.inherit_params:
            self.initial_guess = fitted_result.x
        self.logger.info("The epoch of fitting has finished, the fitted parameters are: [%s]", fitted_result.x)
        self.sigFittingEpochSucceeded.emit(self.get_fitted_data(fitted_result.x))

    # call this func by another thread
    # so, lock is necessary
    def cancel(self):
        self.cancel_mutex.lock()
        self.cancel_flag = True
        self.cancel_mutex.unlock()
