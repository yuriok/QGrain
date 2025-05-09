import logging
import os

QGRAIN_VERSION = "0.5.4.2"
QGRAIN_ROOT_PATH = os.path.dirname(__file__)

HELLO_TEXT = r"""
 _______  _______  _______  _______ _________ _       
(  ___  )(  ____ \(  ____ )(  ___  )\__   __/( (    /|
| (   ) || (    \/| (    )|| (   ) |   ) (   |  \  ( |
| |   | || |      | (____)|| (___) |   | |   |   \ | |
| |   | || | ____ |     __)|  ___  |   | |   | (\ \) |
| | /\| || | \_  )| (\ (   | (   ) |   | |   | | \   |
| (_\ \ || (___) || ) \ \__| )   ( |___) (___| )  \  |
(____\/_)(_______)|/   \__/|/     \|\_______/|/    )_)

An easy-to-use software for the analysis of grain size distributions

"""


def main():
    print(HELLO_TEXT)
    import warnings
    from PySide6 import QtCore
    from .ui import setup_app, setup_logging, create_necessary_folders
    from .ui.MainWindow import MainWindow
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    create_necessary_folders()
    settings = QtCore.QSettings(os.path.join(os.path.expanduser("~"), "QGrain", "qgrain.ini"),
                                QtCore.QSettings.Format.IniFormat)
    language = settings.value("language", "en", type=str)
    theme = settings.value("theme", "default", type=str)
    app = setup_app(language, theme)
    main_window = MainWindow()
    # load the artificial dataset to show the functions of all modules
    dataset = main_window.dataset_generator.get_random_dataset(100)
    main_window.load_dataset_dialog.dataset_loaded.emit(dataset.dataset)
    main_window.ssu_analyzer.on_try_fit_clicked()
    main_window.parameter_editor.refer_parameters(dataset.distribution_type, dataset.parameters[0])
    main_window.parameter_editor.enabled_checkbox.setChecked(True)
    main_window.emma_analyzer.on_try_fit_clicked()
    main_window.udm_analyzer.on_try_fit_clicked()
    main_window.parameter_editor.enabled_checkbox.setChecked(False)
    setup_logging(main_window.statusBar(), main_window.log_dialog)
    main_window.show()
    app.exec()
