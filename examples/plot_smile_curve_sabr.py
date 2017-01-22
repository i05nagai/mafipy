"""
=============================
Smile curve under SABR model
=============================
This example plots smile curve under SABR model.
"""
from __future__ import division, print_function, absolute_import

from mafipy.function import analytic_formula as af
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# default parameters
UNDERLYING = 0.01
STRIKE = 0.001
STRIKE_RANGE = [STRIKE + 0.0005 * i for i in range(0, 40)]
MATURITY = 1.0
ALPHA = 0.005
ALPHA_RANGE = [0.001 * i for i in range(1, 10)]
BETA = 1.0
BETA_RANGE = [0.4 + 0.05 * i for i in range(0, 10)]
RHO = 0.0
RHO_RANGE = [-1.0 + 0.2 * i for i in range(0, 10)]
NU = 0.001
NU_RANGE = [0.001 + 0.003 * i for i in range(0, 10)]
# subplot settings
NUM_ROW = 2
NUM_COL = 2
FIG_SIZE = (10, 8)
plt.figure(figsize=FIG_SIZE)


def to_percent(array):
    return [elem * 100.0 for elem in array]


def plot_smile(title, data, this_plot, num_row=NUM_ROW, num_col=NUM_COL):
    strikes, vols, label = data[0]
    plt.subplot(num_row, num_col, this_plot)
    plt.title(title, size=16)
    strikes = to_percent(strikes)
    plt.xlim(min(strikes), max(strikes))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f %%'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f %%'))
    for strikes, vols, label in data:
        strikes = to_percent(strikes)
        vols = to_percent(vols)
        plt.plot(strikes, vols, label=label)
        plt.legend(prop={'size': 6})


# underlying, maturity, alpha, beta, rho, nu
def gen_dict(u, m, a, b, r, n):
    return {
        "underlying": u,
        "maturity": m,
        "alpha": a,
        "beta": b,
        "rho": r,
        "nu": n,
    }


def gen_label(param_type, val):
    if param_type == "alpha":
        return r"$\alpha$" + " = {0:.3f}".format(val)
    elif param_type == "beta":
        return r"$\beta$" + "={0:.3f}".format(val)
    elif param_type == "rho":
        return r"$\rho$" + " = {0:.3f}".format(val)
    elif param_type == "nu":
        return r"$\nu$" + " = {0:.3f}".format(val)
    else:
        raise ValueError("{0} is not parameter".format(param_type))


def gen_dict_and_label(u, m, a, b, r, n, param_type):
    param = gen_dict(u, m, a, b, r, n)
    label = gen_label(param_type, param[param_type])
    return param, label


# underlying, maturity, alpha, beta, rho, nu
def gen_param_label_list_alpha(u, m, a, b, r, n):
    return [
        gen_dict_and_label(u, m, a_i, b, r, n, "alpha") for a_i in ALPHA_RANGE
    ]


# underlying, maturity, alpha, beta, rho, nu
def gen_param_label_list_beta(u, m, a, b, r, n):
    return [
        gen_dict_and_label(u, m, a, b_i, r, n, "beta") for b_i in BETA_RANGE
    ]


# underlying, maturity, alpha, beta, rho, nu
def gen_param_label_list_rho(u, m, a, b, r, n):
    return [
        gen_dict_and_label(u, m, a, b, r_i, n, "rho") for r_i in RHO_RANGE
    ]


# underlying, maturity, alpha, beta, rho, nu
def gen_param_label_list_nu(u, m, a, b, r, n):
    return [
        gen_dict_and_label(u, m, a, b, r, n_i, "nu") for n_i in NU_RANGE
    ]


def gen_strikes(strike):
    return STRIKE_RANGE


# underlying, maturity, alpha, beta, rho, nu, strikes
def calc_smile(strikes, **param):
    u = param["underlying"]
    m = param["maturity"]
    a = param["alpha"]
    b = param["beta"]
    r = param["rho"]
    n = param["nu"]
    return [
        af.sabr_implied_vol_hagan(
            u, strike, m, a, b, r, n) for strike in strikes
    ]


# params: array of parameters
def calc_smile_set(strikes, param_label_list):
    """calc_smile_set

    :param array strikes:
    :param array params: array of dict
    """
    return [
        (strikes, calc_smile(strikes, **param), label)
        for param, label in param_label_list
    ]


def main():
    # params
    underlying = UNDERLYING
    strike = STRIKE
    maturity = MATURITY
    alpha = ALPHA
    beta = BETA
    rho = RHO
    nu = NU
    # strikes
    strikes = gen_strikes(strike)
    # alpha
    param_label_list_alpha = gen_param_label_list_alpha(
        underlying, maturity, alpha, beta, rho, nu)
    smile_set_alpha = calc_smile_set(strikes, param_label_list_alpha)
    plot_smile("alpha", smile_set_alpha, 1)
    # beta
    param_label_list_beta = gen_param_label_list_beta(
        underlying, maturity, alpha, beta, rho, nu)
    smile_set_beta = calc_smile_set(strikes, param_label_list_beta)
    plot_smile("beta", smile_set_beta, 2)
    # rho
    param_label_list_rho = gen_param_label_list_rho(
        underlying, maturity, alpha, beta, rho, nu)
    smile_set_rho = calc_smile_set(strikes, param_label_list_rho)
    plot_smile("rho", smile_set_rho, 3)
    # nu
    param_label_list_nu = gen_param_label_list_nu(
        underlying, maturity, alpha, beta, rho, nu)
    smile_set_nu = calc_smile_set(strikes, param_label_list_nu)
    plot_smile("nu", smile_set_nu, 4)
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    main()
