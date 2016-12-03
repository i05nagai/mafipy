#!/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime


def to_datetime(date):
    return datetime.strptime(date, "%Y-%m-%d")


def calc_day_count_fraction(from_date, to_date, day_count_convention):
    """
    """
    from_d = to_datetime(from_date)
    to_d = to_datetime(to_date)
    return (to_d - from_d).days / 365.0
