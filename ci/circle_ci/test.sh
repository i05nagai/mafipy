#!/bin/bash

set -e
set -x

if [ "${BENCHMARK_TEST}" = "true" ]; then
	# run benchmarks test
	cd benchmarks
	python run.py run --skip-existing-commits ALL
	python run.py publish
	cd ..
fi
