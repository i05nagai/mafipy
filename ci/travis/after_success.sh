#!/bin/bash

set -e
set -x

if [ "${CODECLIMATE_COVERAGE_REPORT}" == "true" ]; then
	set +x
	codeclimate-test-reporter --token ${CODECLIMATE_COVERAGE_TOKEN}
fi
