#!/bin/python

from mafipy import pricer_quanto_cms
from mafipy import analytic_formula


def linear_annuity(payoff_type):
    if payoff_type == "call":
        gen_payoff_params = gen_call_params
    elif payoff_type == "bull_spread":
        gen_payoff_params = gen_bull_spread_params

    init_swap_rate = 0.018654
    swap_annuity = 2.004720
    maturity = 2.0
    vol_swap_rate = 0.39
    vol_put = 0.39
    vol_call = 0.39
    vol_forward_fx = 0.03
    swap_rate_cdf = pricer_quanto_cms.make_cdf_black_swaption(
        init_swap_rate=init_swap_rate,
        swap_annuity=swap_annuity,
        option_maturity=maturity,
        vol=vol_swap_rate)
    swap_rate_pdf = pricer_quanto_cms.make_pdf_black_swaption(
        init_swap_rate=init_swap_rate,
        swap_annuity=swap_annuity,
        option_maturity=maturity,
        vol=vol_swap_rate)
    swap_rate_pdf_fprime = pricer_quanto_cms.make_pdf_fprime_black_swaption(
        init_swap_rate=init_swap_rate,
        swap_annuity=swap_annuity,
        option_maturity=maturity,
        vol=vol_swap_rate)
    annuity_mapping_params = {
        "alpha0": (0.97990869 / swap_annuity - 0.5) / swap_annuity - 0.5,
        "alpha1": 1.0 / 2.0
    }
    payoff_params = gen_payoff_params()
    forward_fx_diffusion_params = {
        "time": 2.0,
        "vol": vol_forward_fx,
        "corr": 0.1,
        "swap_rate_cdf": swap_rate_cdf,
        "swap_rate_pdf": swap_rate_pdf,
        "swap_rate_pdf_fprime": swap_rate_pdf_fprime
    }
    payer_pricer_params = {
        "init_swap_rate": init_swap_rate,
        "swap_annuity": swap_annuity,
        "option_maturity": 2.0,
        "vol": vol_call,
    }
    bs_pricer = analytic_formula.BlackSwaptionPricerHelper()
    payer_pricer = bs_pricer.make_payers_swaption_wrt_strike(
        **payer_pricer_params)
    receiver_pricer_params = {
        "init_swap_rate": init_swap_rate,
        "swap_annuity": swap_annuity,
        "option_maturity": 2.0,
        "vol": vol_put
    }
    receiver_pricer = bs_pricer.make_receivers_swaption_wrt_strike(
        **receiver_pricer_params)
    discount_factor = 1.0
    price = pricer_quanto_cms.replicate(
        init_swap_rate,
        discount_factor,
        payer_pricer,
        receiver_pricer,
        payoff_type,
        payoff_params,
        forward_fx_diffusion_params,
        "linear",
        annuity_mapping_params,
        min_put_range=0.0002,
        max_call_range=0.050)
    print("price:", price)


def gen_call_params():
    return {
        "strike": 0.0001,
        "gearing": 1.0,
    }


def gen_bull_spread_params():
    return {
        "lower_strike": 0.00001,
        "upper_strike": 0.09327,
        "gearing": 1.0 / 0.09327,
    }


def main():
    print("call")
    linear_annuity("call")
    print("bull_spread")
    linear_annuity("bull_spread")


if __name__ == '__main__':
    main()
