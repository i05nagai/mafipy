#!/bin/python
# -*- coding: utf-8 -*-

import json


class Reader(object):

    def __init__(self):
        pass


class VolatilityReader(Reader):
    """VolatilityReader
    """

    scale_dict = {
        "percent": 0.01
    }

    def __init__(self):
        """
        """
        super().__init__()

    def read_from_file(self, path):
        """
        """
        f = open(path, "r")
        self.data = json.load(f)

    def _extract_volatility(self, elem):
        scale = self.scale_dict[elem["volatility_unit"]]
        return [vol * scale for vol in elem["volatility"]]

    def _extract_strike(self, elem):
        scale = self.scale_dict[elem["strike_unit"]]
        return [strike * scale for strike in elem["strike"]]

    def get_volatility_matrix(self, market_date):
        """
        :param market_date:
        :return
        """
        data_at_date = self.data[market_date]["data"]
        return data_at_date

    def get_volatility_and_strike(self, market_date, maturity_date):
        data_at_date = self.data[market_date]["data"]
        for elem in data_at_date:
            if elem["exercise_date"] == maturity_date:
                volatility = self._extract_volatility(elem)
                strike = self._extract_strike(elem)
                return (volatility, strike)
        return None


if __name__ == '__main__':

    path = "./tests/data/market_volatility.json"
    market_date = "12-Jun-2013"
    maturity_date = "12-Sep-2014"

    reader = VolatilityReader()
    reader.read_from_file(path)
    data = reader.get_volatility_and_strike(market_date, maturity_date)
