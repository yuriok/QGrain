import logging
import time
from enum import Enum, unique

import numpy as np
from PySide2.QtCore import QObject, Signal, Slot
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping, minimize

from algorithms import DistributionType
from data import FittedData
from resolvers import *


class GUIResolver(QObject, Resolver):
    sigSingleIterationFinished = Signal(FittedData)
    sigEpochFinished = Signal(FittedData)
    sigWidgetsEnable = Signal(bool)
    logger = logging.getLogger(name="root.GUIResolver")

    def __init__(self, auto_fit=True, inherit_params=True, emit_iteration=False, time_interval=0.05):
        super().__init__()
        Resolver.__init__(self)
        # use `on_target_data_changed` to modify these values
        self.sample_name = None

        self.last_fitted_params = None

        # settings
        self.auto_fit = auto_fit
        self.inherit_params = inherit_params
        self.emit_iteration = emit_iteration
        self.time_interval = time_interval



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
        self.sample_name = sample_name

        self.feed_data(x, y)

        if self.auto_fit:
            self.try_fit()


    def iteration_callback(self, fitted_params):
        if self.emit_iteration:
            time.sleep(self.time_interval)
            self.sigSingleIterationFinished.emit(self.get_fitted_data(fitted_params))

    def try_fit(self):
        if self.x_to_fit is None or self.y_to_fit is None:
            self.logger.warning("There is no valid data to fit, ignored.")
            return
        self.sigWidgetsEnable.emit(False)

        fitted_result = Resolver.try_fit(self)

        if self.inherit_params:
            self.initial_guess = fitted_result.x

        self.logger.info("The epoch of fitting has finished, the fitted parameters are: [%s]", fitted_result.x)
        self.sigEpochFinished.emit(self.get_fitted_data(fitted_result.x))
