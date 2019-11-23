import logging
import os

import numpy as np
from PySide2.QtCore import Qt, QThread, QTimer, Signal
from PySide2.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QLabel,
                             QPushButton, QRadioButton, QSizePolicy, QWidget)


import sys

sys.path.append(os.getcwd())

from data import FittedData, GrainSizeData
from resolvers import DistributionType


class ControlPanel(QWidget):
    # sigDistributionTypeChanged = Signal(DistributionType)
    sigNcompChanged = Signal(int) # ncomp
    sigFocusSampleChanged = Signal(int) # index of that sample in list
    sigResolverSettingsChanged = Signal(dict)
    sigRuningSettingsChanged = Signal(dict)
    sigDataSettingsChanged = Signal(dict)
    sigRecordFittingData = Signal(str)
    
    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.__ncomp = 2
        self.__data_index = 0
        self.__sample_names = None
        self.auto_run_timer = QTimer()
        self.auto_run_timer.setSingleShot(True)
        self.auto_run_timer.timeout.connect(self.on_auto_run_timer_timeout)
        self.auto_run_flag = False
        
        self.init_ui()
        self.connect_all()
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)


    def init_ui(self):
        
        self.main_layout = QGridLayout(self)
        
        self.distribution_type_label = QLabel("Distribution Type:")
        self.distribution_weibull_radio_button = QRadioButton("Weibull")
        self.distribution_weibull_radio_button.setChecked(True)
        self.distribution_lognormal_radio_button = QRadioButton("Lognormal")
        self.main_layout.addWidget(self.distribution_type_label, 0, 0, 1, 2)
        self.main_layout.addWidget(self.distribution_weibull_radio_button, 0, 2)
        self.main_layout.addWidget(self.distribution_lognormal_radio_button, 0, 3)
        # TODO: Move when the lognormal is added
        self.distribution_weibull_radio_button.setEnabled(False)
        self.distribution_lognormal_radio_button.setEnabled(False)

        self.ncomp_label = QLabel("Components:")
        self.ncomp_display = QLabel("Unknown")
        
        self.ncomp_add_button = QPushButton("+")
        self.ncomp_reduce_button = QPushButton("-")
        
        self.main_layout.addWidget(self.ncomp_label, 1, 0)
        self.main_layout.addWidget(self.ncomp_display, 1, 1)
        self.main_layout.addWidget(self.ncomp_add_button, 1, 2)
        self.main_layout.addWidget(self.ncomp_reduce_button, 1, 3)


        self.show_iteration_checkbox = QCheckBox("Iteration Scope")
        self.show_iteration_checkbox.setCheckState(Qt.Unchecked)
        self.inherit_params_checkbox = QCheckBox("Inherit Parameters")
        self.inherit_params_checkbox.setCheckState(Qt.Checked)
        self.main_layout.addWidget(self.show_iteration_checkbox, 2, 0, 1, 2)
        self.main_layout.addWidget(self.inherit_params_checkbox, 2, 2, 1, 2)
        self.auto_fit_checkbox = QCheckBox("Auto Fit")
        self.auto_fit_checkbox.setCheckState(Qt.Checked)
        self.auto_record_checkbox = QCheckBox("Auto Record")
        self.auto_record_checkbox.setCheckState(Qt.Checked)
        self.main_layout.addWidget(self.auto_fit_checkbox, 3, 0, 1, 2)
        self.main_layout.addWidget(self.auto_record_checkbox, 3, 2, 1, 2)

        self.data_index_label = QLabel("Current Sample:")
        self.data_index_display = QLabel("Unknown")
        self.data_index_previous_button = QPushButton("Previous")
        self.data_index_next_button = QPushButton("Next")
        self.main_layout.addWidget(self.data_index_label, 4, 0)
        self.main_layout.addWidget(self.data_index_display, 4, 1)
        self.main_layout.addWidget(self.data_index_previous_button, 4, 2)
        self.main_layout.addWidget(self.data_index_next_button, 4, 3)

        self.auto_run = QPushButton("Auto Run")
        self.cancel_run = QPushButton("Cancel")
        self.try_fit_button = QPushButton("Try Fit")
        self.record_button = QPushButton("Record")
        self.main_layout.addWidget(self.auto_run, 5, 0)
        self.main_layout.addWidget(self.cancel_run, 5, 1)
        self.main_layout.addWidget(self.try_fit_button, 5, 2)
        self.main_layout.addWidget(self.record_button, 5, 3)

    def connect_all(self):
        self.ncomp_add_button.clicked.connect(self.on_ncomp_add_clicked)
        self.ncomp_reduce_button.clicked.connect(self.on_ncomp_reduce_clicked)
        
        self.data_index_previous_button.clicked.connect(self.on_data_index_previous_clicked)
        self.data_index_next_button.clicked.connect(self.on_data_index_next_clicked)
        
        
        self.show_iteration_checkbox.stateChanged.connect(self.on_show_iteration_changed)
        self.inherit_params_checkbox.stateChanged.connect(self.on_inherit_params_changed)
        self.auto_fit_checkbox.stateChanged.connect(self.on_auto_fit_changed)
        self.auto_record_checkbox.stateChanged.connect(self.on_auto_record_changed)


        self.auto_run.clicked.connect(self.on_auto_run_clicked)
        self.cancel_run.clicked.connect(self.on_cancel_run_clicked)

        

    @property
    def ncomp(self):
        return self.__ncomp

    @ncomp.setter
    def ncomp(self, value: int):
        # check the validity
        # ncomp should be non-negative
        # and it's no need to unmix if ncomp is 1
        # TODO: change the way to generate plot styles in `FittingCanvas`, and remove the limit of <=10
        if value <= 1 or value > 10:
            logging.warning("The component number should be > 1 and <= 10.")
            return
        # update the label to display the value
        self.ncomp_display.setText(str(value))
        self.__ncomp = value
        self.sigNcompChanged.emit(value)
        if self.__sample_names is not None:
            self.data_index = self.data_index
        

    @property
    def data_index(self):
        return self.__data_index

    @data_index.setter
    def data_index(self, value: int):
        if value < 0 or value >= len(self.__sample_names):
            logging.warning("It has reached the first/last sample.")
            return
        # update the label to display the name of this sample
        self.data_index_display.setText(self.__sample_names[value])
        self.__data_index = value
        if self.auto_fit_checkbox.isChecked():
            self.on_widgets_enable_changed(False)
        self.sigFocusSampleChanged.emit(value)

    def on_ncomp_add_clicked(self):
        self.ncomp += 1
        

    def on_ncomp_reduce_clicked(self):
        self.ncomp -= 1

    def on_data_index_previous_clicked(self):
        self.data_index -= 1
        

    def on_data_index_next_clicked(self):
        self.data_index += 1

    def on_retry_clicked(self):
        self.data_index = self.data_index

    def on_show_iteration_changed(self, state):
        if state == Qt.Checked:
            self.sigResolverSettingsChanged.emit({"emit_iteration": True})
        else:
            self.sigResolverSettingsChanged.emit({"emit_iteration": False})

    def on_inherit_params_changed(self, state):
        if state == Qt.Checked:
            self.sigResolverSettingsChanged.emit({"inherit_params": True})
        else:
            self.sigResolverSettingsChanged.emit({"inherit_params": False})

    def on_auto_fit_changed(self, state):
        if state == Qt.Checked:
            self.sigResolverSettingsChanged.emit({"auto_fit": True})
        else:
            self.sigResolverSettingsChanged.emit({"auto_fit": False})

    def on_auto_record_changed(self, state):
        if state == Qt.Checked:
            self.sigDataSettingsChanged.emit({"auto_record": True})
        else:
            self.sigDataSettingsChanged.emit({"auto_record": False})

    def on_data_loaded(self, grain_size_data: GrainSizeData):
        self.__sample_names = [sample.name for sample in grain_size_data.sample_data_list]
        self.data_index = 0

    def on_data_selected(self, index):
        self.data_index = index

    def on_widgets_enable_changed(self, enable: bool):
        if self.auto_run_flag and enable:
            return
        # self.distribution_weibull_radio_button.setEnabled(enable)
        # self.distribution_lognormal_radio_button.setEnabled(enable)
        self.ncomp_add_button.setEnabled(enable)
        self.ncomp_reduce_button.setEnabled(enable)
        self.data_index_previous_button.setEnabled(enable)
        self.data_index_next_button.setEnabled(enable)
        self.show_iteration_checkbox.setEnabled(enable)
        self.inherit_params_checkbox.setEnabled(enable)
        self.auto_fit_checkbox.setEnabled(enable)
        self.auto_record_checkbox.setEnabled(enable)
        self.auto_run.setEnabled(enable)
        if not self.auto_run_flag:
            self.cancel_run.setEnabled(enable)
        self.try_fit_button.setEnabled(enable)
        self.record_button.setEnabled(enable)


    def on_record_clickedd(self):
        self.sigRecordFittingData.emit(self.__sample_names[self.data_index])


    def on_epoch_finished(self, data: FittedData):
        if self.data_index == len(self.__sample_names)-1:
            self.auto_run_flag = False
            self.on_widgets_enable_changed(True)

        if self.auto_run_flag:
            self.auto_run_timer.start(5)

    def on_auto_run_timer_timeout(self):
        self.data_index += 1

    def on_auto_run_clicked(self):
        self.auto_run_flag = True
        self.on_widgets_enable_changed(False)
        self.data_index = 0

    def on_cancel_run_clicked(self):
        if self.auto_run_flag:
            self.auto_run_flag = False
            self.on_widgets_enable_changed(True)
