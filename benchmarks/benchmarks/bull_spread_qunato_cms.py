#!/bin/python
# for asv
from mafipy.function import analytic_formula
from mafipy.replication import pricer_quanto_cms


# swap rate
INIT_SWAP_RATE = 0.018654
SWAP_ANNUITY = 1.0
SWAP_FIXING_DATE = (365 * 2 - 14.0) / 365.0
VOL_SWAP_RATE = 0.39
# forward fx
VOL_FORWARD_FX = 0.13
CORR_FORWARD_FX = 0.3
# payoff
CMS_GEARING = 1.0 / 0.018654
CMS_LOWER_STRIKE = 1e-10
CMS_UPPER_STRIKE = 0.018654
# annuity mapping func
ALPHA0 = (0.97990869 / SWAP_ANNUITY - 0.5) / SWAP_ANNUITY - 0.5
ALPHA1 = 0.5


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


def time_vol_swap_rate():
    vol_swap_rate = 0.05
    params = gen_params("vol_swap_rate", vol_swap_rate)
    linear_annuity_bull_spread_pricer(**params)
    vol_swap_rate = 0.5
    params = gen_params("vol_swap_rate", vol_swap_rate)
    linear_annuity_bull_spread_pricer(**params)
    vol_swap_rate = 1.0
    params = gen_params("vol_swap_rate", vol_swap_rate)
    linear_annuity_bull_spread_pricer(**params)


def time_corr_forward_fx():
    corr_forward_fx = -1.0
    params = gen_params("corr_forward_fx", corr_forward_fx)
    linear_annuity_bull_spread_pricer(**params)
    corr_forward_fx = 0.0
    params = gen_params("corr_forward_fx", corr_forward_fx)
    linear_annuity_bull_spread_pricer(**params)
    corr_forward_fx = 1.0
    params = gen_params("corr_forward_fx", corr_forward_fx)
    linear_annuity_bull_spread_pricer(**params)


def time_vol_foward_fx():
    vol_forward_fx = 0.05
    params = gen_params("vol_forward_fx", vol_forward_fx)
    linear_annuity_bull_spread_pricer(**params)
    vol_forward_fx = 0.5
    params = gen_params("vol_forward_fx", vol_forward_fx)
    linear_annuity_bull_spread_pricer(**params)
    vol_forward_fx = 1.0
    params = gen_params("vol_forward_fx", vol_forward_fx)
    linear_annuity_bull_spread_pricer(**params)
