#!/bin/bash

set -e
set -x

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip
ls -la

# Set up our own virtualenv environment to avoid travis' numpy.
# This venv points to the python interpreter of the travis build
# matrix.
pip install --upgrade pip setuptools
pip install numpy
pip install scipy
# Build mafipy in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

if [ "${BENCHMARK_TEST}" = "true" ]; then
  pip install asv
  pip install virtualenv
  cp ~/mafipy/ci/circle_ci/.asv-machine.json ~/.asv-machine.json
fi
