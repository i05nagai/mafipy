#!/bin/bash

set -e
set -x

if [ "${BENCHMARK_TEST}" = "true" ]; then
	git submodule update --init

	# run benchmarks test
	cd benchmarks
	python run.py run
	python run.py publish
	cd ..
fi
