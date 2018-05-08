#!/bin/bash

set -e
set -x

if [ "${BENCHMARK_TEST}" = "true" ]; then
  cp ci/circle_ci/.asv-machine.json ~/.asv-machine.json

  # move to git submodule
  pushd benchmarks/asv_files
  # for detached HEAD
  git checkout master
  git pull origin master
  popd

  # run benchmarks test
  python setup.py benchmark --run-type=NEW
  python setup.py benchmark_publish
fi
