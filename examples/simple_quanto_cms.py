#!/bin/python

from mafipy import pricer_quanto_cms
from mafipy import analytic_formula


def main():
    init_swap_rate = 0.04
    rate = 0.01
    maturity = 1.0
    vol_swap_rate = 0.05
    swap_rate_cdf = pricer_quanto_cms.make_cdf_black_scholes(
        underlying=init_swap_rate,
        rate=rate,
        maturity=maturity,
        vol=vol_swap_rate)
    swap_rate_pdf = pricer_quanto_cms.make_pdf_black_scholes(
        underlying=init_swap_rate,
        rate=rate,
        maturity=maturity,
        vol=vol_swap_rate)
    swap_rate_pdf_fprime = pricer_quanto_cms.make_pdf_fprime_black_scholes(
        underlying=init_swap_rate,
        rate=rate,
        maturity=maturity,
        vol=vol_swap_rate)
    print(swap_rate_pdf_fprime(1.0))
    annuity_mapping_params = {
        "alpha0": 1.0,
        "alpha1": 0.9
    }
    payoff_params = {
        "strike": 0.04,
        "gearing": 1.0,
    }
    forward_fx_diffusion_params = {
        "time": 1.0,
        "vol": 0.01,
        "corr": 0.1,
        "swap_rate_cdf": swap_rate_cdf,
        "swap_rate_pdf": swap_rate_pdf,
        "swap_rate_pdf_fprime": swap_rate_pdf_fprime
    }
    call_pricer_params = {
        "underlying": init_swap_rate,
        "rate": rate,
        "maturity": 1.0,
        "vol": 0.01,
        "today": 0.0,
    }
    bs_pricer = analytic_formula.BlackScholesPricerHelper()
    call_pricer = bs_pricer.make_call_wrt_strike(**call_pricer_params)
    put_pricer_params = {
        "underlying": init_swap_rate,
        "rate": rate,
        "maturity": 1.0,
        "vol": 0.01,
        "today": 0.0,
    }
    put_pricer = bs_pricer.make_put_wrt_strike(**put_pricer_params)
    pricer = pricer_quanto_cms.SimpleQuantoCmsPricer(
        "linear", annuity_mapping_params,
        "call", payoff_params,
        forward_fx_diffusion_params,
        call_pricer,
        put_pricer)
    discount_factor = 0.9
    price = pricer.eval(discount_factor, init_swap_rate,
                        min_put_range=0.0 + 0.000000001,
                        max_call_range=0.06)
    print("price:", price)


if __name__ == '__main__':
    main()
