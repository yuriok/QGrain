import numpy as np
from PyQt5.QtCore import QMutex, pyqtSignal
from PyQt5.QtWidgets import (QGridLayout, QLabel, QPushButton, QSizePolicy,
                             QWidget)


class ControlPanel(QWidget):
    sigNcompChanged = pyqtSignal(int)
    sigTargetDataChanged = pyqtSignal(str, np.ndarray, np.ndarray)

    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.__ncomp = 2
        self.__classes = None
        self.__data = None
        self.__data_index = 0
        self.__mutex = QMutex()
        self.init_ui()
        

    @property
    def ncomp(self):
        self.__mutex.lock()
        ncomp = self.__ncomp
        self.__mutex.unlock()
        return ncomp

    @ncomp.setter
    def ncomp(self, value:int):
        # check the validity
        # ncomp should be non-negative
        # and it's no need to unmix if ncomp is 1
        # TODO: change the way to generate plot styles in `FittingCanvas`, and remove the limit of <=10
        if value <= 1 or value > 10:
            raise ValueError(value)

        self.__mutex.lock()
        # update the label to display the value
        self.ncomp_display.setText(str(value))
        self.__ncomp = value
        self.sigNcompChanged.emit(value)
        self.__mutex.unlock()

    @property
    def data_index(self):
        self.__mutex.lock()
        data_index = self.__data_index
        self.__mutex.unlock()
        return data_index


    @data_index.setter
    def data_index(self, value:int):
        if value < 0 or value >= len(self.__data):
            raise ValueError(value)

        self.__mutex.lock()
        # update the label to display the name of this sample
        sample_id = self.__data[value]["id"]
        sample_data = self.__data[value]["data"]
        self.data_index_display.setText(sample_id)
        self.__data_index = value
        self.__mutex.unlock()
        self.sigTargetDataChanged.emit(sample_id, self.__classes, sample_data)
        


    def init_ui(self):
        self.setGeometry(300,300,300,200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout = QGridLayout(self)
        self.ncomp_label = QLabel("Component Number:")
        self.ncomp_display = QLabel()
        self.ncomp_add_button = QPushButton("Add")
        self.ncomp_reduce_button = QPushButton("Reduce")
        self.main_layout.addWidget(self.ncomp_label, 0, 0)
        self.main_layout.addWidget(self.ncomp_display, 0, 1)
        self.main_layout.addWidget(self.ncomp_add_button, 0, 2)
        self.main_layout.addWidget(self.ncomp_reduce_button, 0, 3)
        
        self.data_index_label = QLabel("Current Sample:")
        self.data_index_display = QLabel()
        self.data_index_previous_button = QPushButton("Previous")
        self.data_index_next_button = QPushButton("Next")
        self.main_layout.addWidget(self.data_index_label, 1, 0)
        self.main_layout.addWidget(self.data_index_display, 1, 1)
        self.main_layout.addWidget(self.data_index_previous_button, 1, 2)
        self.main_layout.addWidget(self.data_index_next_button, 1, 3)

        self.ncomp_add_button.clicked.connect(self.on_ncomp_add_clicked)
        self.ncomp_reduce_button.clicked.connect(self.on_ncomp_reduce_clicked)
        self.data_index_previous_button.clicked.connect(self.on_data_index_previous_clicked)
        self.data_index_next_button.clicked.connect(self.on_data_index_next_clicked)

    def on_ncomp_add_clicked(self):
        try:
            self.ncomp += 1
        except ValueError:
            return # TODO: use logging to record errors and give hints for users


    def on_ncomp_reduce_clicked(self):
        # ignore the action when the `ncomp` is 1
        try:
            self.ncomp -= 1
        except ValueError:
            return # TODO: use logging to record errors and give hints for users

    def on_data_index_previous_clicked(self):
        # ignore the action when the `data_index` is 0
        try:
            self.data_index -= 1
        except ValueError:
            return

    def on_data_index_next_clicked(self):
        # ignore the action when the `data_index` is len(data)-1
        try:
            self.data_index += 1
        except ValueError:
            return


    
    def on_data_loaded(self, classes, data):
        self.__mutex.lock()
        self.__classes = classes
        self.__data = data
        self.__mutex.unlock()


    def on_epoch_finished(self, data):
        print(data[-1])

        # self.feed_one_patch_data()

    def feed_one_patch_data(self):
        sample_id = self.data[self.data_index]["id"]
        sample_data = self.data[self.data_index]["data"]
        self.sigTargetDataChanged.emit(sample_id, self.classes, sample_data)

    def auto_run(self):
        self.feed_one_patch_data()
        
    def single_run(self):
        self.feed_one_patch_data()
