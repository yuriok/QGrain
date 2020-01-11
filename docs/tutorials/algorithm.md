# The algorithms of QGrain

## Background

Grain size distribution (GSD) data have been widely used in Earth sciences, especially Quaternary Geology, due to its convenience and reliability. However, the usages of GSD are still oversimplified. The geological information contained in GSD is very abundant, but only some simplified proxies (e.g. mean grain size) are widely used. The most important reason is that GSD data are hard to interpret and visualize directly.

To overcome this, some researchers have developed the methods to unmix the mixed multi-modal GSD to some components to make the interpretation and visualization easier. These methods can be divided into two routes. One is end-member analysis (EMA) (Weltje, 1997) which takes a batch of samples for the calculation of the end-members. Another is called single-specimen unmixing (SSU) (Sun et al., 2002) which treats each sample as an individual.

The key difference between the two routes is that whether the end-members of a batch of samples are consistent. EMA believes that the end-members between different samples are consistent, the variations of GSD are only caused by the changing of fractions of the end-members. On the contrary, SSU has no assumption on the end-members, i.e. it admits that the end-members may vary between different samples.

Some mature tools (Paterson and Heslop, 2015; Dietze and Dietze, 2019) taking the EMA route have appeared, but there is no available public and easy-to-use tool for SSU. That the reason of creating QGrain.

## Fundamental

The math principle of SSU has been described by Sun et al. (2002).

In short, the distribution of a n-components mixed sample can be indicated as:

y = f<sub>1</sub> * *d*<sub>1</sub>(x) + ... + f<sub>n</sub> * *d*<sub>n</sub>(x),

where y is the mixed distribution, f<sub>i</sub> is the fraction of component i, *d*<sub>i</sub> is the base distribution function (e.g. Normal and Weibull) of component i, x is the classes of grain size.

The question is to get the distribution paramters of *d*<sub>i</sub>.

Therefore, the unmixing problem can be coverted to an optimization problem:

minimize the error (e.g. sum of squared error) between y<sub>test</sub> and y<sub>guess</sub>.

## Data preprocess

In fact the input data of each sample are two array. One is the classes of grain size, another is the distribution. Usually, there are many 0 values in the head and tail of distribution array. These 0 values were caused by the limit of test precision. In fact, they should be close to 0 but not equal to 0. This difference will bring a constant error which is large enough to effect the fitting result. QGrain will exclude these 0 values to obtain better performance.

## Local optimization

Due to the complexity of base distribution function, the error function is non-convex. At present, there is no high-efficiency method to find the global minimum of a non-convex function. So, an alternative solution is local optimization. Local optimization can converge to a minimum rapidly, but without guarantee that the minimum is global. Optimization problem also is a core topic of machine learning. Therefore, there are many mature local optimization algorithms that meet our requirement. Here we use Sequential Least SQuares Programming (SLSQP) (Kraft, 1988) algorithm to perform local optimization.

## Global optimization

With the increase of component number, the error function will become much more complex. It's difficult to get a satisfactory result if only use local optimization.

QGrain uses a global optimization algorithm called basinhopping (Wales & Doye, 1997) to improve the robustness.

This global optimization algorithm will not search the whole space but will shift to another initial point to start a new local optimization process after one local optimization process finished. That makes it has ability to escape some loacl minimum and keep the efficiency meanwhile.

## Base distribution function

At present, QGrain supports the following distribution types:

|Distribution Type|Parameter Number|Fitting Space|Skew|
|:-:|:-:|:-:|:-:|
|Normal<sup>1<sup>|2|Bin Numbers|No|
|Weibull|2|Bin Numbers|Yes|
|Gen. Weibull<sup>2</sup>|3|Bin Numbers|Yes|

1. Normal distribution againsts bin numbers is equal to Lognormal distribution againsts grain size (μm).
2. **Gen. Weibull** is General Weibull which has an additional location parameter.

## Steps of fitting

1. Data Loading
2. Get information (e.g. distribution type and component number)
3. Generate error function
4. Data preprocess
5. Global optimization (basinhopping)
6. Final optimization (another local optimization, SLSQP)
7. Generate fitting result by the parameters of error function

## Referances

* [Weltje, G.J. End-member modeling of compositional data: Numerical-statistical algorithms for solving the explicit mixing problem. Math Geol 29, 503–549 (1997) doi:10.1007/BF02775085](https://doi.org/10.1007/BF02775085)

* Kraft, D. A software package for sequential quadratic programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace Center – Institute for Flight Mechanics, Koln, Germany.

* Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms. Journal of Physical Chemistry A, 1997, 101, 5111.
