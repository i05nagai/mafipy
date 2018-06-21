#!/bin/bash -l

set -e

export ENVIRONMENT_RELEASE="dev"
export MAFIPY_USERNAME_DEV=""
export MAFIPY_PASSWORD_DEV=""
export MAFIPY_USERNAME_PROD=""
export MAFIPY_PASSWORD_PROD=""
export MAFIPY_GITHUB_API_TOKEN=""

do_release()
{
  export MAFIPY_USERNAME=$1
  export MAFIPY_PASSWORD=$2
  shift
  shift
  local args=$@
  bash ${PATH_TO_REPOSITORY}/scripts/release.sh \
    ${args}
}

release_dev()
{
  args="--upload"
  do_release $MAFIPY_USERNAME_DEV $MAFIPY_PASSWORD_DEV ${args}
}

release_prod()
{
  args="--no-test --upload"
  do_release $MAFIPY_USERNAME_PROD $MAFIPY_PASSWORD_PROD ${args}
}


check_released_pkg()
{
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
    mafipy==0.1.dev2
  pip install \
    matplotlib==2.2.2

  #
  # run examples
  #
  python examples/plot_black_scholes_european_vanilla_option.py
  python examples/plot_quanto_cms_bull_spread.py
  python examples/plot_smile_curve_sabr.py
}

