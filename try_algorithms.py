import time

import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.optimize import basinhopping, minimize

def get_valid_data_range(y):
    start_index = 0
    end_index = len(y)
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
    # x_to_fit = np.array(range(end_index-start_index)) + 1
    x_to_fit = np.array(range(len(x))[start_index: end_index])+1
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


def common_gen_normal(x, xi, alpha, kappa):
    if alpha <= 0.0:
        return 0.0
    if kappa < 0.0 and x <= (xi+alpha/kappa):
        return 0.0
    elif kappa > 0.0 and x >= (xi+alpha/kappa):
        return 0.0
    elif kappa == 0.0:
        return normal((x-xi)/alpha, 0.0, 1.0)/(alpha-kappa*(x-xi))
    else:
        return normal(-1/kappa*np.log(1-(kappa*(x-xi)/alpha)), 0.0, 1.0)/(alpha-kappa*(x-xi))

gen_normal = np.frompyfunc(common_gen_normal, 4, 1)

def triple_gen_normal(x, xi1, alpha1, kappa1, xi2, alpha2, kappa2, xi3, alpha3, kappa3, f1, f2):
    return f1*gen_normal(x, xi1, alpha1, kappa1) + f2*gen_normal(x, xi2, alpha2, kappa2) + (1-f1-f2)*gen_normal(x, xi3, alpha3, kappa3)


# x = np.linspace(-10, 10, 20001)
# plt.plot(x, gen_normal(x, 0, 1, 1))
# plt.plot(x, gen_normal(x, 0, 1, 0.5))
# plt.plot(x, gen_normal(x, 0, 1, 0))
# plt.plot(x, gen_normal(x, 0, 1, -0.5))
# plt.plot(x, gen_normal(x, 0, 1, -1))
# plt.show()

# test the implement of lognormal func
# x = np.linspace(-10, 10, 20001)
# plt.plot(x, lognormal(x, 1, 1))
# plt.plot(x, lognormal(x, 1, 2))
# plt.show()

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
def common_weibull(x, beta, eta):
    if beta <= 0.0 or eta <= 0.0:
        return 0.0
    else:
        return beta/eta*(x / eta)**(beta-1) * np.exp(-(x/eta)**beta)

weibull = np.frompyfunc(common_weibull, 3, 1)

def triple_weibull(x, a1, c1, a2, c2, a3, c3, f1, f2):
    return f1 * weibull(x, a1, c1) + f2 * weibull(x, a2, c2) + (1-f1-f2)*weibull(x, a3, c3)


def common_gen_weibull(x, beta, eta, loc):
    if loc > x:
        return 0.0
    else:
        return common_weibull(x-loc, beta, eta)

gen_weibull = np.frompyfunc(common_gen_weibull, 4, 1)

def triple_gen_weibull(x, a1, c1, loc1, a2, c2, loc2, a3, c3, loc3, f1, f2):
    return f1 * gen_weibull(x, a1, c1, loc1) + f2 * gen_weibull(x, a2, c2, loc2) + (1-f1-f2)*gen_weibull(x, a3, c3, loc3)


sheet = xlrd.open_workbook("C:\\Users\\Yuri\\Desktop\\WN19 GS.xlsx").sheet_by_index(0)

classes = np.array(sheet.row_values(0)[1:])
INFINITESIMAL = 1e-100
component_number = 3
distribution_type = 4
if distribution_type == 0: # Weibull
    target_mixed_func = triple_weibull
    target_single_func = weibull
    initial_guess = [2, 10, 2, 20, 2, 30, 0.33, 0.33]
    bounds = [(INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, 1), (INFINITESIMAL, 1)]
    folder = "C:\\Users\\Yuri\\Desktop\\Results\\Weibull\\"
elif distribution_type == 1: # Lognormal
    target_mixed_func = triple_lognormal
    target_single_func = lognormal
    initial_guess = [0.9, 1/2, 0.9, 1/4, 0.9, 1/8, 0.33, 0.33]
    bounds = [(None, None), (INFINITESIMAL, None), (None, None), (INFINITESIMAL, None), (None, None), (INFINITESIMAL, None), (INFINITESIMAL, 1), (INFINITESIMAL, 1)]
    folder = "C:\\Users\\Yuri\\Desktop\\Results\\Lognormal\\"
elif distribution_type == 2: # Normal
    target_mixed_func = triple_normal
    target_single_func = normal
    initial_guess = [30, 1, 55, 1, 65, 1, 0.33, 0.33]
    bounds = [(None, None), (INFINITESIMAL, None), (None, None), (INFINITESIMAL, None), (None, None), (INFINITESIMAL, None), (INFINITESIMAL, 1), (INFINITESIMAL, 1)]
    folder = "C:\\Users\\Yuri\\Desktop\\Results\\Normal\\"
elif distribution_type == 3: # General Normal
    target_mixed_func = triple_gen_normal
    target_single_func = gen_normal
    initial_guess = [1, 1, 0, 10, 2, 0, 30, 3, 0, 0.33, 0.33]
    bounds = [(None, None), (0, None), (None, None), (None, None), (0, None), (None, None), (None, None), (0, None), (None, None), (INFINITESIMAL, 1), (INFINITESIMAL, 1)]
    folder = "C:\\Users\\Yuri\\Desktop\\Results\\GenNormal\\"
elif distribution_type == 4: # General Weibull
    target_mixed_func = triple_gen_weibull
    target_single_func = gen_weibull
    initial_guess = [2, 10, 0, 2, 20, 0, 2, 30, 0, 0.33, 0.33]
    bounds = [(INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, None), (INFINITESIMAL, 1), (INFINITESIMAL, 1)]
    folder = "C:\\Users\\Yuri\\Desktop\\Results\\GenWeibull\\"



def plot(x, params, title):
    fitted_sum = target_mixed_func(x, *params)
    plt.scatter(x, y, label="Raw Data")
    plt.plot(x, fitted_sum, label="Fitted Sum")
    if distribution_type == 3 or distribution_type == 4:
        plt.plot(x, target_single_func(x, *params[0:3])*params[-2], label="C1")
        plt.plot(x, target_single_func(x, *params[3:6])*params[-1], label="C2")
        plt.plot(x, target_single_func(x, *params[6:9])*(1-params[-2]-params[-1]), label="C2")
    else:
        plt.plot(x, target_single_func(x, *params[0:2])*params[-2], label="C1")
        plt.plot(x, target_single_func(x, *params[2:4])*params[-1], label="C2")
        plt.plot(x, target_single_func(x, *params[4:6])*(1-params[-2]-params[-1]), label="C2")
    plt.legend()
    plt.title(title)
    plt.xlabel("Bin Number")
    plt.ylabel("Probability Density")
    plt.ylim(0, None)



last_params = initial_guess
for i in range(1, sheet.nrows):
    row_values = sheet.row_values(i)
    sample_name = row_values[0]
    sample_data = np.array(row_values[1:]) / 100
    x, y = get_processed_data(classes, sample_data)
    # x, y = classes, sample_data

    iteration = 0
    def callback(params):
        # global iteration
        # plt.clf()
        # plot(x, params, "iteration: [%d]"%iteration)
        # iteration += 1
        # plt.pause(0.01)
        pass

    def closure(p):
        return np.mean(np.square(target_mixed_func(x, *p) - y))*100
    
    method = "SLSQP"
    # method = "L-BFGS-B"
    # method = "COBYLA"
    constraints = [{'type': 'ineq', 'fun': lambda args:  1 - np.sum(args[1-component_number:]) + INFINITESIMAL}]
    minimizer_kwargs = dict(method=method, callback=callback, bounds=bounds, constraints=constraints, options={"maxiter": 500, "disp": False, "ftol": 1e-50})

    global_res = basinhopping(closure, last_params, niter=100, stepsize=0.5, niter_success=1, minimizer_kwargs=minimizer_kwargs)
    res = minimize(closure, global_res.x, method=method, callback=callback, bounds=bounds, constraints=constraints, options={"maxiter": 1000, "disp": False, "ftol": 1e-100})
    # y_fitted = target_mixed_func(x, *res.x)
    last_params = res.x
    print(res.x)
    
    plt.clf()
    plot(x, res.x, sample_name)
    # plt.xscale("log")
    plt.savefig(folder+"{0}.png".format(sample_name), dpi=300)
