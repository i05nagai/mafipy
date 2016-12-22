"""
=================================================
European vanilla option under Black scholes model
=================================================
This example calculates call and put options under black scholes model.
p.d.f. and c.d.f. of the unerlying are also drawn.
"""

from __future__ import division

from mafipy import analytic_formula as af
from mafipy import payoff
from mafipy import pricer_quanto_cms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# default parameters
UNDERLYING = 0.01
UNDERLYING_RANGE = [0.001 + 0.0005 * i for i in range(0, 40)]
STRIKE = 0.01
STRIKE_RANGE = [0.001 + 0.0005 * i for i in range(0, 40)]
RATE = 0.05
MATURITY = 1.0
VOL = 0.25

# subplot settings
NUM_ROW = 3
NUM_COL = 1
FIG_SIZE = (10, 8)
plt.figure(figsize=FIG_SIZE)


def to_percent(array):
    return [elem * 100.0 for elem in array]


def plot_curve(label, xs, ys):
    plt.plot(xs, ys, label=label)
    plt.legend(prop={'size': 9})


def set_plot(plot_num,
             title,
             x_min=0.0,
             x_max=1.0,
             x_unit="",
             y_unit="",
             num_row=NUM_ROW,
             num_col=NUM_COL):
    plt.subplot(num_row, num_col, plot_num)
    plt.title(title, size=16)
    plt.xlim(x_min, x_max)

    if x_unit == "percent":
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f %%'))
    if y_unit == "percent":
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f %%'))


def plot_price_curves(title,
                      underlyings,
                      prices_call,
                      prices_put,
                      payoffs_call,
                      payoffs_put):
    underlyings = to_percent(underlyings)
    set_plot(1, title, min(underlyings), max(underlyings),
             x_unit="percent", y_unit="")

    plot_curve("price call", underlyings, prices_call)
    plot_curve("price put", underlyings, prices_put)
    plot_curve("payoff call", underlyings, payoffs_call)
    plot_curve("payoff call", underlyings, payoffs_put)


def plot_density_curve(title, underlyings, call_densities):
    underlyings = to_percent(underlyings)
    set_plot(2, title, min(underlyings), max(underlyings),
             x_unit="percent", y_unit="")

    plot_curve("call", underlyings, call_densities)


def plot_cumulative_density_curve(title, underlyings, call_densities):
    underlyings = to_percent(underlyings)
    set_plot(3, title, min(underlyings), max(underlyings),
             x_unit="percent", y_unit="")

    plot_curve("call", underlyings, call_densities)


def main():
    underlyings = UNDERLYING_RANGE
    rate = RATE
    maturity = MATURITY
    vol = VOL
    strike = STRIKE

    # price
    prices_call = [af.calc_black_scholes_call_value(
        underlying, strike, rate, maturity, vol) for underlying in underlyings]
    prices_put = [af.calc_black_scholes_put_value(
        underlying, strike, rate, maturity, vol) for underlying in underlyings]
    payoffs_call = [payoff.payoff_call(
        underlying, strike) for underlying in underlyings]
    payoffs_put = [payoff.payoff_put(
        underlying, strike) for underlying in underlyings]
    plot_price_curves("price",
                      underlyings,
                      prices_call,
                      prices_put,
                      payoffs_call,
                      payoffs_put)
    # p.d.f.
    pdf_func = pricer_quanto_cms.make_pdf_black_scholes(
        strike, rate, maturity, vol)
    densities = [pdf_func(underlying) for underlying in underlyings]
    plot_density_curve("pdf", underlyings, densities)
    # c.d.f.
    cdf_func = pricer_quanto_cms.make_cdf_black_scholes(
        strike, rate, maturity, vol)
    cumulative_densities = [cdf_func(underlying) for underlying in underlyings]
    plot_cumulative_density_curve("cdf", underlyings, cumulative_densities)

    plt.show()


if __name__ == '__main__':
    print(__doc__)
    main()
