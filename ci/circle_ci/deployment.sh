#!/bin/bash

set -e
set -x

# mafipy/master
TEST_BRANCH="master"
# mafipy_benchmarks/master
TARGET_BRANCH="master"

if [ "${BENCHMARK_TEST}" = "true" ]; then

	# Pull requests and commits to other branches shouldn't try to deploy, just build to verify
	if [ "$TRAVIS_PULL_REQUEST" != "false" -o "$CIRCLE_BRANCH" != "$TEST_BRANCH" ]; then
		echo "Skipping deploy; just doing a build."
		curl https://api.csswg.org/bikeshed/ -f -F file=@index.bs > index.html;
		exit 0
	fi

	# repository url
	REPO_URL=`git config remote.origin.url`

	# Save some useful information from original repository
	BENCHMARKED_SHA1="$CIRCLE_SHA1"

	# move to submodule
	cd benchmarks/html
	git config user.name "Circle CI"
	git config user.email "circle_ci@i05nagai.me"


	# If there are no changes (e.g. this is a README update) then just bail.
	if [ -z `git diff --exit-code` ]; then
		echo "No changes to the spec on this push; exiting."
		exit 0
	fi

	git add .
	git commit -m "Deploy to GitHub Pages: ${BENCHMARKED_SHA1}"
	git push $REPO_URL $TARGET_BRANCH
fi