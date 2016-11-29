#!/bin/bash

set -e

if [ "$COVERAGE" = "true" ]; then
	pip install coverage
fi

if [ "$SKIP_TESTS" == "true" ]; then
    echo "No need to build mafipy when not running the tests"
else
    # Build mafipy in the install.sh script to collapse the verbose
    # build output in the travis output when it succeeds.
    python --version
    python -c "import numpy; print('numpy %s' % numpy.__version__)"
    python -c "import scipy; print('scipy %s' % scipy.__version__)"
    python setup.py develop
fi
