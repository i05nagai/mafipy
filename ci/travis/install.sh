#!/bin/bash

set -e
set -x

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

if [ "${SKIP_TESTS}" == "true" ]; then
    echo "No need to build mafipy when not running the tests"
else

  # Set up our own virtualenv environment to avoid travis' numpy.
  # This venv points to the python interpreter of the travis build
  # matrix.
  pip install --upgrade pip==9.0.3 setuptools
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
  # Build mafipy in the install.sh script to collapse the verbose
  # build output in the travis output when it succeeds.
  python --version
  python -c "import numpy; print('numpy %s' % numpy.__version__)"
  python -c "import scipy; print('scipy %s' % scipy.__version__)"
  python -c "import pytest; print('pytest %s' % pytest.__version__)"

  if [ "${COVERAGE}" = "true" ]; then
    pip install coverage
  fi

  # executing only in python 3.5
  if [ "${CODECLIMATE_COVERAGE_REPORT}" = "true" ]; then
    #
    # install cc-test-reporter
    #
    mkdir -p $HOME/.cache/codeclimate
    curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-latest-linux-amd64 > $HOME/.cache/codeclimate/cc-test-reporter
    chmod +x $HOME/.cache/codeclimate/cc-test-reporter
    ln -s $HOME/.cache/codeclimate/cc-test-reporter /usr/bin/cc-test-reporter
    # run
    /usr/bin/cc-test-reporter before-build
  fi

fi

