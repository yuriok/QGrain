import os
import sys
import unittest

import numpy as np
from scipy.stats import norm

sys.path.append(os.getcwd())
from resolvers.GUIResolver import *


class TestGUIResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = GUIResolver()
        self.resolver.distribution_type = DistributionType.Normal
        self.resolver.component_number = 1
        self.resolver.sigFittingEpochSucceeded.connect(self.on_fitting_finished)
        self.fitting_result = None

    def on_fitting_finished(self, fitting_result: FittingResult):
        self.fitting_result = fitting_result

    def test_valid(self):
        x = np.linspace(-10, 10, 201)
        y = norm.pdf(x, 5.45, 2.21)
        sample_data = SampleData("Sample", x, y)
        self.resolver.on_target_data_changed(sample_data)
        self.resolver.try_fit()
        self.assertIsNotNone(self.fitting_result)
        self.assertIs(self.fitting_result.real_x, x)
        self.assertIs(self.fitting_result.target_y, y)


if __name__ == "__main__":
    unittest.main()