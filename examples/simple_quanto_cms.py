"""
==================================================
European bull-spread quanto CMS option with Copula
==================================================
This example calculates bull spread option of quanto CMS option.
The model is based on
Andersen, Leif BG, and Vladimir V. Piterbarg.
Interest rate modeling. Vol. 1.
London: Atlantic Financial Press, 2010.
"""
from __future__ import division

from mafipy import analytic_formula
from mafipy import pricer_quanto_cms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# swap rate
INIT_SWAP_RATE = 0.018654
SWAP_ANNUITY = 1.0
SWAP_FIXING_DATE = (365 * 2 - 14.0) / 365.0
VOL_SWAP_RATE = 0.39
VOL_SWAP_RATE_RANGE = [0.05 * i for i in range(1, 30)]
# forward fx
VOL_FORWARD_FX = 0.13
VOL_FORWARD_FX_RANGE = [0.05 * i for i in range(1, 30)]
CORR_FORWARD_FX = 0.3
CORR_FORWARD_FX_RANGE = [0.1 * i for i in range(-10, 10)]
# payoff
CMS_GEARING = 1.0 / 0.018654
CMS_LOWER_STRIKE = 1e-10
CMS_UPPER_STRIKE = 0.018654
# annuity mapping func
ALPHA0 = (0.97990869 / SWAP_ANNUITY - 0.5) / SWAP_ANNUITY - 0.5
ALPHA1 = 0.5
# subplot settings
NUM_ROW = 2
NUM_COL = 2
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


def plot(title, label, xs, ys, plot_num, x_unit=""):
    xs = to_percent(xs)
    set_plot(plot_num, title, min(xs), max(xs),
             x_unit=x_unit, y_unit="")

    plot_curve(label, xs, ys)


def linear_annuity_bull_spread_pricer(
        init_swap_rate,
        swap_annuity,
        vol_swap_rate,
        swap_fixing_date,
        vol_forward_fx,
        corr_forward_fx,
        cms_lower_strike,
        cms_upper_strike,
        cms_gearing,
        alpha0,
        alpha1):

    payoff_params = {
        "lower_strike": cms_lower_strike,
        "upper_strike": cms_upper_strike,
        "gearing": cms_gearing,
    }
    swap_rate_cdf = pricer_quanto_cms.make_cdf_black_swaption(
        init_swap_rate=init_swap_rate,
        swap_annuity=swap_annuity,
        option_maturity=swap_fixing_date,
        vol=vol_swap_rate)
    swap_rate_pdf = pricer_quanto_cms.make_pdf_black_swaption(
        init_swap_rate=init_swap_rate,
        swap_annuity=swap_annuity,
        option_maturity=swap_fixing_date,
        vol=vol_swap_rate)
    swap_rate_pdf_fprime = pricer_quanto_cms.make_pdf_fprime_black_swaption(
        init_swap_rate=init_swap_rate,
        swap_annuity=swap_annuity,
        option_maturity=swap_fixing_date,
        vol=vol_swap_rate)
    annuity_mapping_params = {
        "alpha0": alpha0,
        "alpha1": alpha1
    }
    forward_fx_diffusion_params = {
        "time": swap_fixing_date,
        "vol": vol_forward_fx,
        "corr": corr_forward_fx,
        "swap_rate_cdf": swap_rate_cdf,
        "swap_rate_pdf": swap_rate_pdf,
        "swap_rate_pdf_fprime": swap_rate_pdf_fprime,
        "is_inverse": True
    }
    payer_pricer_params = {
        "init_swap_rate": init_swap_rate,
        "swap_annuity": swap_annuity,
        "option_maturity": swap_fixing_date,
        "vol": vol_swap_rate,
    }
    bs_pricer = analytic_formula.BlackSwaptionPricerHelper()
    payer_pricer = bs_pricer.make_payers_swaption_wrt_strike(
        **payer_pricer_params)
    receiver_pricer_params = {
        "init_swap_rate": init_swap_rate,
        "swap_annuity": swap_annuity,
        "option_maturity": swap_fixing_date,
        "vol": vol_swap_rate
    }
    receiver_pricer = bs_pricer.make_receivers_swaption_wrt_strike(
        **receiver_pricer_params)
    discount_factor = 1.0
    price = pricer_quanto_cms.replicate(
        init_swap_rate,
        discount_factor,
        payer_pricer,
        receiver_pricer,
        "bull_spread",
        payoff_params,
        forward_fx_diffusion_params,
        "linear",
        annuity_mapping_params,
        min_put_range=1e-16,
        max_call_range=0.040)
    return price


def gen_params(key, value):
    params = {
        "init_swap_rate": INIT_SWAP_RATE,
        "swap_annuity": SWAP_ANNUITY,
        "vol_swap_rate": VOL_SWAP_RATE,
        "swap_fixing_date": SWAP_FIXING_DATE,
        "vol_forward_fx": VOL_FORWARD_FX,
        "corr_forward_fx": CORR_FORWARD_FX,
        "cms_lower_strike": CMS_LOWER_STRIKE,
        "cms_upper_strike": CMS_UPPER_STRIKE,
        "cms_gearing": CMS_GEARING,
        "alpha0": ALPHA0,
        "alpha1": ALPHA1
    }
    params[key] = value
    return params


def main():
    pricer = linear_annuity_bull_spread_pricer

    print("swap rate volatility")
    prices = []
    for vol_swap_rate in VOL_SWAP_RATE_RANGE:
        params = gen_params("vol_swap_rate", vol_swap_rate)
        price = pricer(**params)
        prices.append(price)
        print("\t{0}\t{1}".format(vol_swap_rate, price))
    plot("vol_swap_rate", "price", VOL_SWAP_RATE_RANGE, prices, 1, "percent")

    print("forward fx correlation")
    prices = []
    for corr_forward_fx in CORR_FORWARD_FX_RANGE:
        params = gen_params("corr_forward_fx", corr_forward_fx)
        price = pricer(**params)
        prices.append(price)
        print("\t{0}\t{1}".format(corr_forward_fx, price))
    plot("corr", "price", CORR_FORWARD_FX_RANGE, prices, 2)

    print("forward fx volatility")
    prices = []
    for vol_forward_fx in VOL_FORWARD_FX_RANGE:
        params = gen_params("vol_forward_fx", vol_forward_fx)
        price = pricer(**params)
        prices.append(price)
        print("\t{0}\t{1}".format(vol_forward_fx, price))
    plot("vol_foward_fx", "price", VOL_FORWARD_FX_RANGE, prices, 3, "percent")

    plt.show()


if __name__ == '__main__':
    main()
