# How to adjust algorithm

For some relatively simple samples (e.g. Loess), it's no need to adjust the algorithm settings. But for some fluvial and lacustrine deposits, the conditions of material sources and dynamics may be much more complex. In other words, the measured distribution may be the mixed result of many components (e.g. much than 4), and components may differ greatly. The complex distribution makes it's difficult to get a proper result. In order to deal with more complex situations, QGrain provides some algorithm settings.

If you are not familiar the algorithms of QGrain, click [here](./algorithm) for more information.

You can click the **Settings** menu to modify the settings of algorithm.

![Setting Window](../figures/settings_window.png)

## Paramters

* **Global Optimization Max Iteration**: Max iteration number of global optimization. If the global optimization iteration has reached the max number, fitting process will stop.
* **Global Optimization Success Iteration**: It's one of the terminal conditions of global optimization. It means the iteration number of reaching the same minimum.
* **Global Optimization Stepsize**: The stepsize of searching global minimum. Greater stepsize will jump out the local minimum easier but may miss the global minimum.
* **Global Minimizer Tolerance Level**: The tolerance level of the minimizer of global optimization. Tolerance level means the accepted minimum variation (10 ^ -level) of the target function. It controls the precision and speed of fitting. It's recommended to use ralatively lower level in global optimization process but higher leverl in final fitting.
* **Global Minimizer Max Iteration**: Max iteration number of the minimizer of global optimization.
* **Final Fitting Tolerance Level**: The tolerance level of the minimizer of final fitting.
* **Final Fitting Max Iteration**: Max iteration number of the minimizer of final fitting.
