#!/bin/python
# -*- coding: utf-8 -*-

import scipy.optimize
import numpy as np
from . import analytic_formula
from . import date_util
from . import market_data


class Calibrator(object):

    def __init__(self):
        pass

    def calibrate(self):
        raise NotImplementedError()


class SimpleSabrCalibrator(Calibrator):
    """
    Calibrate Alpha, Rho, and Nu Directly
    """

    def __init__(self):
        super().__init__()

    def calibrate(self):
        pass


class WestSabrCalibrator(Calibrator):
    """
    Calibrate Rho and Nu by Implying Alpha from At-The-Money Volatility.
    West, G., "Calibration of the SABR Model in Illiquid Markets," Applied Mathematical Finance, 12(4), pp. 371–385, 2004.
    """

    def __init__(self):
        super().__init__()

    def _calc_poly_alpha(self, *args):
        """
        Given rho and nu, then solve cubic polynominal with respecto alpha.
        """
        rho = args[0]
        nu = args[1]
        underlying = args[2]
        beta = args[3]
        time = args[4]
        atm_vol = args[5]

        oneMinusBeta = 1.0 - beta
        numerator1 = (oneMinusBeta**2) * time
        denominator1 = 24.0 * (underlying ** (2 * oneMinusBeta))
        term1 = numerator1 / denominator1
        numerator2 = rho * beta * nu * time
        denominator2 = 4.0 * (underlying**oneMinusBeta)
        term2 = numerator2 / denominator2
        numerator3 = 2.0 - 3.0 * (rho**2) * (nu**2) * time
        term3 = (1.0 + numerator3 / 24.0)
        term4 = atm_vol * (underlying**oneMinusBeta)
        coeff = [term1, term2, term3, term4]

        return np.roots(coeff)

    def _solve_alpha(self, rho, nu, underlying, beta, time, atm_vol):
        """
        """
        complex_to_real = (
            lambda x: np.real(x) if np.abs(x) > 0.0 else float("inf"))
        roots = self._calc_poly_alpha(rho, nu, underlying, beta, time, atm_vol)
        return min(map(complex_to_real, roots))

    def _make_objective_function(self, vol_list, strike_list, beta, time):
        """
        param:
        vol_list:
        """

        atm_strike = market_data.get_atm_strike(strike_list)
        atm_vol = market_data.get_atm_vol(vol_list)
        underlying = atm_strike

        def objective_function(args):
            rho = args[0]
            nu = args[1]
            alpha = self._solve_alpha(rho, nu, atm_strike, beta, time, atm_vol)
            total = 0.0
            for strike, vol in zip(strike_list, vol_list):
                total += (vol - analytic_formula.calc_sabr_model_implied_vol(
                    underlying, strike, time, alpha, beta, rho, nu)) ** 2
            return total
        return objective_function

    def calibrate(self, strike_list, vol_list, beta, rho, nu, time):
        """
        """
        atm_strike = market_data.get_atm_strike(strike_list)
        atm_vol = market_data.get_atm_vol(vol_list)
        objective_function = self._make_objective_function(
            vol_list, strike_list, beta, time)
        # determine rho and nu
        result = scipy.optimize.minimize(
            objective_function, [rho, nu], method="BFGS")
        if not result.success:
            raise ValueError("optimization error!")
        (rho, nu) = result.x
        # determine alpha by rho and nu
        alpha = self._solve_alpha(rho, nu, atm_strike, beta, time, atm_vol)
        return (alpha, rho, nu, beta)


if __name__ == '__main__':
    import market_reader

    path = "./tests/data/market_volatility.json"
    market_date = "2013-06-12"
    maturity_date = "2014-09-12"

    alpha = 1.0
    beta = 0.0
    rho = 0.0
    nu = 0.0
    time = date_util.calc_day_count_fraction(market_date, maturity_date, "")

    reader = market_reader.VolatilityReader()
    reader.read_from_file(path)
    vol_list, strike_list = reader.get_volatility_and_strike(
        market_date, maturity_date)

    calibrator = WestSabrCalibrator()
    alpha = calibrator.calibrate(strike_list, vol_list, beta, rho, nu, time)
    print(alpha)
