#!/bin/bash

set -e

if [ "${CODECLIMATE_COVERAGE_REPORT}" == "true" ]; then
  echo "Executing /usr/bin/cc-test-reporter after-build"
  $HOME/.cache/codeclimate/cc-test-reporter after-build \
      --coverage-input-type coverage.py \
      --id ${CC_TEST_REPORTER_ID}
fi
