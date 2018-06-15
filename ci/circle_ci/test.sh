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

  #
  # run benchmarks test
  # You need to delete all results files named `commithash-....json` and push to GitHub
  # if KeyError occurs when benchmark_publish is executed.
  python setup.py benchmark --run-type=NEW
  python setup.py benchmark_publish
fi
