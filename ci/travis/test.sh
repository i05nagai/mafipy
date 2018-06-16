#!/bin/bash

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip
ls $HOME/.cache

if [ "${SKIP_TESTS}" == "true" ]; then
    echo "No need to build mafipy when not running the tests"
else
  # executing only in python 3.5
  if [ "${CODECLIMATE_COVERAGE_REPORT}" = "true" ]; then
    python setup.py test --cov-report=xml,term
  else
    python setup.py test
  fi
fi
