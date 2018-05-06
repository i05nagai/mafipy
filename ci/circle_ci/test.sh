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
  python setup.py benchmark --NEW
  python setup.py benchmark_publish
fi
