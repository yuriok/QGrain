from QGrain.generate import SIMPLE_PRESET, random_sample
from QGrain.statistics import all_statistics

sample = random_sample(**SIMPLE_PRESET)
statistics = all_statistics(sample.classes, sample.classes_phi, sample.distribution)

{'arithmetic': {'mean': 24.416012713219324,
  'std': 28.745497203626858,
  'skewness': 2.5993653712218205,
  'kurtosis': 14.571969009591882},
 'geometric': {'mean': 11.01298368038815,
  'std': 4.457238315788177,
  'skewness': -0.6317811838318534,
  'kurtosis': 2.678393555741556,
  'std_description': 'Very poorly sorted',
  'skewness_description': 'Fine skewed',
  'kurtosis_description': 'Mesokurtic',
  'median': 14.06324632030902,
  'mean_description': 'Medium Silt',
  'mode': 28.2507508924551,
  'modes': (28.2507508924551,)},
 'logarithmic': {'mean': 6.504650807362412,
  'std': 2.1561500997915255,
  'skewness': 0.6317811838318537,
  'kurtosis': 2.678393555741555,
  'std_description': 'Very poorly sorted',
  'skewness_description': 'Fine skewed',
  'kurtosis_description': 'Mesokurtic',
  'median': 6.151926529246772,
  'mean_description': 'Medium Silt',
  'mode': 5.14556697554162,
  'modes': (5.14556697554162,)},
 'geometric_fw57': {'mean': 10.48617511007019,
  'std': 4.272953356391491,
  'skewness': -0.3091153710808348,
  'kurtosis': 0.9328536255538155,
  'std_description': 'Very poorly sorted',
  'skewness_description': 'Very fine skewed',
  'kurtosis_description': 'Mesokurtic',
  'median': 14.06324632030902,
  'mean_description': 'Medium Silt',
  'mode': 28.2507508924551,
  'modes': (28.2507508924551,)},
 'logarithmic_fw57': {'mean': 6.575367646845154,
  'std': 2.0952335686291423,
  'skewness': 0.309115371080835,
  'kurtosis': 0.9328536255538153,
  'std_description': 'Very poorly sorted',
  'skewness_description': 'Very fine skewed',
  'kurtosis_description': 'Mesokurtic',
  'median': 6.151926529246772,
  'mean_description': 'Medium Silt',
  'mode': 5.14556697554162,
  'modes': (5.14556697554162,)},
 'proportions_gsm': (0.0, 0.0969, 0.9030999999999999),
 'proportions_ssc': (0.0969, 0.7523, 0.1508),
 'proportions_bgssc': (0.0, 0.0, 0.0969, 0.7523, 0.1508),
 'proportions': {('', 'Megaclasts'): 0.0,
  ('Very large', 'Boulder'): 0.0,
  ('Large', 'Boulder'): 0.0,
  ('Medium', 'Boulder'): 0.0,
  ('Small', 'Boulder'): 0.0,
  ('Very small', 'Boulder'): 0.0,
  ('Very coarse', 'Gravel'): 0.0,
  ('Coarse', 'Gravel'): 0.0,
  ('Medium', 'Gravel'): 0.0,
  ('Fine', 'Gravel'): 0.0,
  ('Very fine', 'Gravel'): 0.0,
  ('Very coarse', 'Sand'): 0.0,
  ('Coarse', 'Sand'): 0.0,
  ('Medium', 'Sand'): 0.0008000000000000001,
  ('Fine', 'Sand'): 0.0138,
  ('Very fine', 'Sand'): 0.08230000000000001,
  ('Very coarse', 'Silt'): 0.1989,
  ('Coarse', 'Silt'): 0.211,
  ('Medium', 'Silt'): 0.1431,
  ('Fine', 'Silt'): 0.11250000000000002,
  ('Very fine', 'Silt'): 0.0868,
  ('Very coarse', 'Clay'): 0.0669,
  ('Coarse', 'Clay'): 0.052,
  ('Medium', 'Clay'): 0.0251,
  ('Fine', 'Clay'): 0.006,
  ('Very fine', 'Clay'): 0.0007999999999999999},
 'group_folk54': 'Slit',
 '_group_bp12_symbols': ['(s)', '(c)', 'SI'],
 'group_bp12_symbol': '(s)(c)SI',
 'group_bp12': 'Slightly Sandy Slightly Clayey Silt'}

from QGrain.generate import SIMPLE_PRESET, random_dataset
from QGrain.io import save_statistics

dataset = random_dataset(**SIMPLE_PRESET, n_samples=200)
save_statistics(dataset, "./Statistics.xlsx")

from QGrain.generate import SIMPLE_PRESET, random_sample
from QGrain.statistics import *

sample = random_sample(**SIMPLE_PRESET)
# statistical parameters
s = arithmetic(sample.classes, sample.distribution)
s = geometric(sample.classes, sample.distribution)
s = logarithmic(sample.classes_phi, sample.distribution)
ppf = reversed_phi_ppf(sample.classes_phi, sample.distribution)
s = geometric_fw57(ppf)
s = logarithmic_fw57(ppf)

# proportions
p = proportions_gsm(sample.classes_phi, sample.distribution)
p = proportions_ssc(sample.classes_phi, sample.distribution)
p = proportions_bgssc(sample.classes_phi, sample.distribution)
p = all_proportions(sample.classes_phi, sample.distribution)

# classification groups
g = group_folk54(sample.classes_phi, sample.distribution)
g = group_bp12(sample.classes_phi, sample.distribution)
