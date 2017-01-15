#!/bin/bash

set -e
set -x

if [ "${BENCHMARK_TEST}" = "true" ]; then
	# for detached HEAD
	pushd benchmarks/asv_files
	git checkout master
	popd

	# run benchmarks test
	pushd benchmarks
	python run.py run NEW
	python run.py publish
	popd ..
fi
