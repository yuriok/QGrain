import time

import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.optimize import basinhopping, minimize

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
    # x_to_fit = range(len(x))[start_index: end_index]
    y_to_fit = y[start_index: end_index]
    return x_to_fit, y_to_fit

def lognormal(x, mu, sigma):
    # return 1/((x-mu)*sigma*np.sqrt(2*np.pi))*np.exp(-np.square(np.log(x-mu))/(2*np.square(sigma)))
    return 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-np.square(np.log(x)/mu)/(2*np.square(sigma)))

def triple_lognormal(x, mu1, sigma1, mu2, sigma2, mu3, sigma3, f1, f2):
    return f1 * lognormal(x, mu1, sigma1) + f2 * lognormal(x, mu2, sigma2) + (1-f1-f2) * lognormal(x, mu3, sigma3)

def normal(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-np.square(x-mu)/(2*np.square(sigma)))

def triple_normal(x, mu1, sigma1, mu2, sigma2, mu3, sigma3, f1, f2):
    return f1 * normal(x, mu1, sigma1) + f2 * normal(x, mu2, sigma2) + (1-f1-f2) * normal(x, mu3, sigma3)

# test the implement of lognormal func
x = np.linspace(-10, 10, 20001)
plt.plot(x, lognormal(x, 1, 1))
plt.plot(x, lognormal(x, 1, 2))
plt.show()


# test the implement of normal func
# x = np.linspace(0.001, 2, 2001)
# plt.plot(x, normal(x, 0, 10))
# plt.plot(x, normal(x, 0, 3/2))
# plt.plot(x, normal(x, 0, 1))
# plt.plot(x, normal(x, 0, 1/2))
# plt.plot(x, normal(x, 0, 1/4))
# plt.plot(x, normal(x, 0, 1/8))
# plt.ylim(0, 4)
# plt.xlim(0, None)
# plt.show()






# The pdf function of Weibull distribution
def weibull(x, beta, eta):
    return beta/eta*(x / eta)**(beta-1) * np.exp(-(x/eta)**beta)

def triple_weibull(x, a1, c1, a2, c2, a3, c3, f1, f2):
    return f1 * weibull(x, a1, c1) + f2 * weibull(x, a2, c2) + (1-f1-f2)*weibull(x, a3, c3)

sheet = xlrd.open_workbook("C:\\Users\\Yuri\\Desktop\\WN19 GS.xlsx").sheet_by_index(0)

classes = np.array(sheet.row_values(0)[1:])
INFINITESIMAL = 1e-100
component_number = 3
distribution_type = 1
if distribution_type == 0: # Weibull
    target_mixed_func = triple_weibull
    target_single_func = weibull
    initial_guess = [2, 10, 2, 20, 2, 30, 0.33, 0.33]
    bounds = [(2+INFINITESIMAL, None), (INFINITESIMAL, None), (2+INFINITESIMAL, None), (INFINITESIMAL, None), (2+INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, 1), (INFINITESIMAL, 1)]
    folder = "C:\\Users\\Yuri\\Desktop\\Weibull\\"
elif distribution_type == 1: # Lognormal
    target_mixed_func = triple_lognormal
    target_single_func = lognormal
    initial_guess = [0.9, 1/2, 0.9, 1/4, 0.9, 1/8, 0.33, 0.33]
    bounds = [(None, None), (INFINITESIMAL, None), (None, None), (INFINITESIMAL, None), (None, None), (INFINITESIMAL, None), (INFINITESIMAL, 1), (INFINITESIMAL, 1)]
    folder = "C:\\Users\\Yuri\\Desktop\\Lognormal\\"
elif distribution_type == 2: # Normal
    target_mixed_func = triple_normal
    target_single_func = normal
    initial_guess = [1, 1, 8, 1, 30, 1, 0.33, 0.33]
    bounds = [(None, None), (INFINITESIMAL, None), (None, None), (INFINITESIMAL, None), (None, None), (INFINITESIMAL, None), (INFINITESIMAL, 1), (INFINITESIMAL, 1)]
    folder = "C:\\Users\\Yuri\\Desktop\\Normal\\"

constraints = [{'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-component_number:]) + INFINITESIMAL}]
minimizer_kwargs = dict(method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "disp": False, "ftol": 1e-100})


def plot(x, params, title):
    fitted_sum = target_mixed_func(x, *params)
    plt.scatter(x, y, label="Raw Data")
    plt.plot(x, fitted_sum, label="Fitted Sum")
    plt.plot(x, target_single_func(x, *params[0:2])*params[-2], label="C1")
    plt.plot(x, target_single_func(x, *params[2:4])*params[-1], label="C2")
    plt.plot(x, target_single_func(x, *params[4:6])*(1-params[-2]-params[-1]), label="C2")
    plt.legend()
    plt.title(title)
    plt.xlabel("Bin Number")
    plt.ylabel("Probability Density")
    plt.ylim(0, None)

for i in range(1, sheet.nrows):
    row_values = sheet.row_values(i)
    sample_name = row_values[0]
    sample_data = np.array(row_values[1:]) / 100
    x, y = get_processed_data(classes, sample_data)
    # x, y = classes, sample_data

    iteration = 0
    def callback(params):
        global iteration
        plt.clf()
        plot(x, params, "iteration: [%d]"%iteration)
        iteration += 1
        plt.pause(0.01)

    def closure(p):
        return np.mean(np.square(target_mixed_func(x, *p) - y))*100


    global_res = basinhopping(closure, initial_guess, niter=100, stepsize=2.0, niter_success=10, minimizer_kwargs=minimizer_kwargs)
    res = minimize(closure, global_res.x, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "disp": False, "ftol": 1e-1000})
    y_fitted = target_mixed_func(x, *res.x)

    print(res.x)
    
    plt.clf()
    plot(x, res.x, sample_name)
    # plt.xscale("log")
    plt.savefig(folder+"{0}.png".format(sample_name), dpi=300)
