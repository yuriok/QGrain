__all__ = ["AboutWindow"]

import os

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QGridLayout, QMainWindow, QTextBrowser, QWidget

about_md = """
# QGrain

QGrain is an easy-to-use software which integrates most analysis tools to deal with grain size distributions.

## Tools

* Statistics moments

  Calculate samples' mean, std, skewness, kurtosis, etc. The statistics formulas were referred to Blott & Pye (2001)'s work.

* End-member modelling analysis (EMMA)

  EMMA is a widely used algorithm to extract the end-members of a whole dataset.
  Here, QGrain provides a new implement which is based the basic Neural Network.

* Single Sample Unmix (SSU)

  SSU also is used to extract the end-members (i.e. components) of samples.
  Different from EMMA, it only deals with one sample at each computation.

* Principal Component Analysis (PCA)

  PCA can extract the major (which has the greatest variance) and minor signals of data.
  It also be used to reduce the dimension of data.

* Hierarchy Clustering

  Hierarchy clustering is a set of clustering algorithms.
  It can generate a hierarchy structure to reprsents the relationships of samples by determining their distances.
  Using this algorithm, we can find out the typical samples and have a overall cognition of the dataset.

## Authors

Feel free to contact the authors below, if you have some questions.

* Yuming Liu

  <a href="mailto:\\liuyuming@ieecas.cn">liuyuming@ieecas.cn</a>

"""

class AboutWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.setWindowTitle(self.tr("About"))
        self.setMinimumSize(500, 600)
        self.text = QTextBrowser()
        self.layout.addWidget(self.text, 0, 0)
        self.text.setMarkdown(about_md)
        self.text.setOpenExternalLinks(True)


if __name__ == "__main__":
    import sys

    from QGrain.entry import setup_app
    app = setup_app()
    main = AboutWindow()
    main.show()
    sys.exit(app.exec_())
