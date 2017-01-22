#!/bin/python
# -*- coding: utf-8 -*-


def raise_key_error(keys, dictionary, dictionary_name):
    """raise_key_error

    :param keys:
    :param dictionary:
    :param symboltable:
    """
    missing_keys = []
    for key in keys:
        if key not in dictionary:
            missing_keys.append(key)
    msg = "{0} does not contain follwoing keys:\n".format(dictionary_name)
    for key in keys:
        msg += "    {0} is missing.\n".format(key)
    raise KeyError(msg)
