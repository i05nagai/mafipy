#!/bin/bash -l

set -e

#
#
#
eval "$(pyenv init -)"
pyenv virtualenv-delete mafipy_install
pyenv virtualenv 3.5.2 mafipy_install
pyenv activate mafipy_install
pip install --upgrade pip
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://testpypi.python.org/pypi \
  --extra-index-url https://pypi.org/simple/ \
  mafipy==0.1.dev12
pip install \
  matplotlib==2.2.2

#
# run examples
#
python examples/plot_black_scholes_european_vanilla_option.py
python examples/plot_quanto_cms_bull_spread.py
python examples/plot_smile_curve_sabr.py
