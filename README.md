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
