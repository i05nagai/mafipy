#!/bin/bash

set -e
set -x

if [ "${BENCHMARK_TEST}" = "true" ]; then
	pushd benchmarks/asv_files
	# for detached HEAD
	git checkout master
	git pull origin master
	popd

	# run benchmarks test
	pushd benchmarks
	python run.py run NEW
	python run.py publish
	popd
fi
