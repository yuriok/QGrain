from scipy.stats import exponweib, lognorm

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from scipy.optimize import basinhopping

def get_valid_data_range(y):
    start_index = 0
    end_index = -1
    for i, value in enumerate(y):
        if value > 0.0:
            start_index = i
            break
    for i, value in enumerate(y[start_index+1:], start_index+1):
        if value == 0.0:
            end_index = i
            break
    return start_index, end_index

def get_processed_data(x, y):
    start_index, end_index = get_valid_data_range(y)

    x_to_fit = np.array(range(end_index-start_index)) + 1
    y_to_fit = y[start_index: end_index]
    return x_to_fit, y_to_fit


def normal(x, beta, eta):
    # return (1 / (x*eta*np.sqrt(2*np.pi))) * np.exp(-np.square((np.log(x)-beta) / eta) / 2)
    return (1 / np.sqrt(2*np.pi*eta))*np.exp(-(x-beta)**2/(2*eta**2))

def triple_normal(x, beta1, eta1, beta2, eta2, beta3, eta3, f1, f2):
    return f1 * normal(x, beta1, eta1) + f2 * normal(x, beta2, eta2) + (1-f1-f2)*normal(x, beta3, eta3)

# The pdf function of Weibull distribution
def weibull(x, beta, eta):
    return beta/eta*(x / eta)**(beta-1) * np.exp(-(x/eta)**beta)

def triple_weibull(x, a1, c1, a2, c2, a3, c3, f1, f2):
    return f1 * weibull(x, a1, c1) + f2 * weibull(x, a2, c2) + (1-f1-f2)*weibull(x, a3, c3)

import xlrd
sheet = xlrd.open_workbook("C:\\Users\\Yuri\\Desktop\\WN19 GS.xlsx").sheet_by_index(0)

classes = np.array(sheet.row_values(0)[1:])
sample_data = np.array(sheet.row_values(9)[1:]) / 100

x, y = get_processed_data(classes, sample_data)

# x = np.linspace(0.01, 10, 1000)
# # y = lognorm.pdf(x, 1.2, loc=1.2, scale=np.exp(1.4))
# y = triple_lognormal(x, 1.1, 2.1, 1.3, 1.3, 4.1, 1.5, 1.6, 3.1, 1.2, 0.3, 0.2)

# def closure(p):
#     return np.sum(np.square(triple_normal(x, *p) - y)) * 100

def closure(p):
    return np.mean(np.square(triple_weibull(x, *p) - y))*100

# res = minimize(closure, [27, 9.8, 38, 3.5, 3.2, 60.2, 0.3, 0.3], bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, 1), (0, 1)], constraints=[{'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-3:]) + 1e-100}], method="SLSQP", options={"maxiter": 1000, "disp": True, "ftol": 1e-100})

def callback(x, f, accept):
    print("Parameters are : [%s];\nFunction value is: [%0.2E];\n" % (x, f))
    return True
    # raise Exception("Something happened")

def iteration_callback(x):
    print(x)
    # raise Exception("Something happened")



# res = basinhopping(closure, [2, 10, 4, 20, 8, 30, 0.3, 0.3], callback=callback, niter_success=10, niter=100, minimizer_kwargs={"method": "SLSQP", "bounds": [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, 1), (0, 1)], "constraints": [{'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-3:]) + 1e-100}], "options":{"maxiter": 1000, "disp": True, "ftol": 1e-100}})

# res = basinhopping(closure, [2, 10, 2, 10, 2, 10, 0.33, 0.33], callback=callback, niter_success=10, niter=100, minimizer_kwargs={"method": "SLSQP", "callback": iteration_callback, "bounds": [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, 1), (0, 1)], "constraints": [{'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-3:]) + 1e-100}], "options":{"maxiter": 1000, "disp": True, "ftol": 1e-100}})
res = minimize(closure, [2, 10, 2, 10, 2, 10, 0.33, 0.33], callback=iteration_callback, method="SLSQP", bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, 1), (0, 1)], options={"maxiter": 1000, "disp": True, "ftol": 1e-100})

y_fitted = triple_weibull(x, *res.x)

print(res.x)
plt.scatter(x, y)
plt.plot(x, y_fitted)

# plt.xscale("log")

plt.show()