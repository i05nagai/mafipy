#!/bin/bash -x

set -eu

if [ "$CODECLIMATE_COVERAGE_REPORT" == "true" ]; then
	codeclimate-test-reporter --token $CODECLIMATE_COVERAGE_TOKEN
fi