#!/bin/bash

set -e
set -x

usage() {
  cat <<EOF
deployment.sh is a tool for ...

Usage:
    deployment.sh <path-to-repository>

Environment variables
  BENCHMARK_TEST    true/false
EOF
}

#
# validate arugments
#
PATH_TO_REPOSITORY=$1
if [ -z ${PATH_TO_REPOSITORY+x} ]; then
  usage
  exit 1
fi

# mafipy/master
TEST_BRANCH="master"
# mafipy_benchmarks/master
TARGET_BRANCH="master"
# origin/master of mafipy_benchmarks
TARGET_REPOSITORY="origin"

if [ "${BENCHMARK_TEST}" = "true" ]; then

  # Pull requests and commits to other branches shouldn't try to deploy, just build to verify
  if [ "$CI_PULL_REQUEST" != "" -o "$CIRCLE_BRANCH" != "$TEST_BRANCH" ]; then
    echo "Skipping deploy; just doing a build."
    exit 0
  fi

  # Save some useful information from original repository
  BENCHMARKED_SHA1="$CIRCLE_SHA1"

  # move to submodule
  cd ${PATH_TO_REPOSITORY}/benchmarks/asv_files
  git config user.name "Circle CI"
  git config user.email "circle_ci@i05nagai.me"

  # If there are no changes (e.g. this is a README update) then just bail.
  if [ -z `git diff --exit-code` ]; then
    echo "No changes to the spec on this push; exiting."
    exit 0
  fi

  # push benchmark results to mafipy_benchmarks
  git remote set-url origin git@github.com:i05nagai/mafipy_benchmarks.git
  git add .
  git commit -m "Deploy to GitHub Pages: ${BENCHMARKED_SHA1}"
  GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git push $TARGET_REPOSITORY $TARGET_BRANCH
fi
