#!/bin/bash -x

set -e

if [[ "${SKIP_TESTS}" == "true" ]]; then
    echo "No need to build mafipy when not running the tests"
else

	# Set up our own virtualenv environment to avoid travis' numpy.
	# This venv points to the python interpreter of the travis build
	# matrix.
	deactivate
	virtualenv --python=python ~/testvenv
	source ~/testvenv/bin/activate
	pip install --upgrade pip setuptools
	pip install numpy
	pip install scipy
    # Build mafipy in the install.sh script to collapse the verbose
    # build output in the travis output when it succeeds.
    python --version
    python -c "import numpy; print('numpy %s' % numpy.__version__)"
    python -c "import scipy; print('scipy %s' % scipy.__version__)"


	if [ "${COVERAGE}" = "true" ]; then
		pip install coverage
	fi

	if [ "${CODECLIMATE_COVERAGE_REPORT}" = "true" ]; then
		pip install codeclimate-test-reporter
	fi

fi

