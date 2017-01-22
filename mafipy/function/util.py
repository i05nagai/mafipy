#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from . import error


def get_var_name(var, symboltable, error=None):
    """getVarName
    Return a var's name as a string.
    This funciton require a
    symboltable(returned value of globals() or locals())
    in the name space where you search the var's name.
    If you set error='exception',
    this raise a ValueError when the searching failed.

    :param var:
    :param symboltable:
    :param error:
    """
    for k, v in symboltable.iteritems():
        if id(v) == id(var):
            return k
    else:
        if error == "exception":
            raise ValueError("Undefined function is mixed in subspace?")
        else:
            return error


def check_keys(keys, dictionary, symboltable):
    for key in keys:
        if key in dictionary:
            return dictionary
    dictionary_name = get_var_name(dictionary, symboltable)
    error.raise_key_error(keys, dictionary, dictionary_name)
