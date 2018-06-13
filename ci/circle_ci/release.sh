#!/bin/bash

set -e
set -x

usage() {
  cat <<EOF
release.sh is a tool for executing release scripts in Circle CI.

Usage:
    deployment.sh <path-to-repository>

Environment variables
              true/false
  ENVIRONMENT_RELEASE     dev/prod
  MAFIPY_USERNAME_DEV    true/false
  MAFIPY_PASSWORD_DEV    true/false
  MAFIPY_USERNAME_PROD    true/false
  MAFIPY_PASSWORD_PROD    true/false
  MAFIPY_GITHUB_API_TOKEN    GitHub API token.
EOF
}

#
# validate arugments
#
readonly PATH_TO_REPOSITORY=$1
if [ -z ${PATH_TO_REPOSITORY+x} ]; then
  usage
  exit 1
fi

#
# validate arguments
#
if [ "${ENVIRONMENT_RELEASE}" = "dev" ]; then
  readonly MAFIPY_USERNAME=$MAFIPY_USERNAME_DEV
  readonly MAFIPY_PASSWORD=$MAFIPY_PASSWORD_DEV
elif [ "${ENVIRONMENT_RELEASE}" = "prod" ]; then
  readonly MAFIPY_USERNAME=$MAFIPY_USERNAME_PROD
  readonly MAFIPY_PASSWORD=$MAFIPY_PASSWORD_PROD
  ARGS="${ARGS} --no-test "
else
  usage
  exit 1
fi

cd ${PATH_TO_REPOSITORY}
# required MAFIPY_USERNAME and MAFIPY_PASSWORD, MAFIPY_GITHUB_API_TOKEN
export MAFIPY_USERNAME
export MAFIPY_PASSWORD
bash ${PATH_TO_REPOSITORY}/scripts/packaging.sh \
  ${ARGS} \
  --upload
