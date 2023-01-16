#!/bin/bash

set -e
set -x

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip
ls -la

git submodule init
git submodule update
# Set up our own virtualenv environment to avoid travis' numpy.
# This venv points to the python interpreter of the travis build
# matrix.
pip install --upgrade pip==22.3.1 setuptools
pip install -r requirements.txt
# Build mafipy in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

if [ "${BENCHMARK_TEST}" = "true" ]; then
  pip install -r requirements-dev.txt
fi
