#!/bin/python

from mafipy import pricer_quanto_cms
from mafipy import analytic_formula


def linear_annuity(payoff_type, payoff_params, quanto_cms_params):

    init_swap_rate = quanto_cms_params["init_swap_rate"]
    swap_annuity = quanto_cms_params["swap_annuity"]
    maturity = quanto_cms_params["maturity"]
    vol_swap_rate = quanto_cms_params["vol_swap_rate"]
    vol_put = quanto_cms_params["vol_put"]
    vol_call = quanto_cms_params["vol_call"]
    vol_forward_fx = quanto_cms_params["vol_forward_fx"]
    corr_forward_fx = quanto_cms_params["corr_forward_fx"]

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
    forward_fx_diffusion_params = {
        "time": 2.0,
        "vol": vol_forward_fx,
        "corr": corr_forward_fx,
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
        min_put_range=1e-6,
        max_call_range=0.040)
    print("  price:", price)


def gen_call_params1(cms_params):
    cms_strike = cms_params["cms_strike"]
    return {
        "strike": 1e-4,
        "gearing": 1.0 / cms_strike,
    }


def gen_call_params2(cms_params):
    cms_strike = cms_params["cms_strike"]
    return {
        "strike": cms_strike,
        "gearing": 1.0 / cms_strike,
    }


def gen_bull_spread_params(cms_params):
    cms_strike = cms_params["cms_strike"]
    return {
        "lower_strike": 0.00001,
        "upper_strike": cms_strike,
        "gearing": 1.0 / cms_strike,
    }


def gen_cms_params_atm():
    return {
        "init_swap_rate": 0.018654,
        "cms_strike": 0.018654,
        "swap_annuity": 2.004720,
        "maturity": 2.0,
        "vol_swap_rate": 0.39,
        "vol_put": 0.39,
        "vol_call": 0.39,
        "vol_forward_fx": 0.03,
        "corr_forward_fx": 0.3,
    }


def gen_cms_params_otm():
    return {
        "init_swap_rate": 0.018654,
        "cms_strike": 0.018654 / 2.0,
        "swap_annuity": 2.004720,
        "maturity": 2.0,
        "vol_swap_rate": 0.39,
        "vol_put": 0.39,
        "vol_call": 0.39,
        "vol_forward_fx": 0.03,
        "corr_forward_fx": 0.3,
    }


def gen_cms_params_itm():
    return {
        "init_swap_rate": 0.018654,
        "cms_strike": 0.018654 * 1.5,
        "swap_annuity": 2.004720,
        "maturity": 2.0,
        "vol_swap_rate": 0.39,
        "vol_put": 0.39,
        "vol_call": 0.39,
        "vol_forward_fx": 0.03,
        "corr_forward_fx": 0.1,
    }


def main():
    print("--call")
    print("----ATM")
    cms_params = gen_cms_params_atm()
    payoff_params = gen_call_params1(cms_params)
    linear_annuity("call", payoff_params, cms_params)
    payoff_params = gen_call_params2(cms_params)
    linear_annuity("call", payoff_params, cms_params)
    print("----OTM")
    cms_params = gen_cms_params_otm()
    payoff_params = gen_call_params1(cms_params)
    linear_annuity("call", payoff_params, cms_params)
    payoff_params = gen_call_params2(cms_params)
    linear_annuity("call", payoff_params, cms_params)
    print("----ITM")
    cms_params = gen_cms_params_itm()
    payoff_params = gen_call_params1(cms_params)
    linear_annuity("call", payoff_params, cms_params)
    payoff_params = gen_call_params2(cms_params)
    linear_annuity("call", payoff_params, cms_params)
    print()
    print("--bull_spread")
    print("----ATM")
    cms_params = gen_cms_params_atm()
    payoff_params = gen_bull_spread_params(cms_params)
    linear_annuity("bull_spread", payoff_params, cms_params)
    print("----OTM")
    cms_params = gen_cms_params_otm()
    payoff_params = gen_bull_spread_params(cms_params)
    linear_annuity("bull_spread", payoff_params, cms_params)


if __name__ == '__main__':
    main()
