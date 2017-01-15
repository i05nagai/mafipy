#!/bin/bash

set -e
set -x

if [ "${BENCHMARK_TEST}" = "true" ]; then
	# run benchmarks test
	cd benchmarks
	python run.py run NEW
	python run.py publish
	cd ..
fi
