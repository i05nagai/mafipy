#!/bin/bash

set -e

usage() {
  cat <<EOF
release.sh is a tool for executing release scripts in Circle CI.

Usage:
    release.sh <path-to-repository>

Environment variables
  ENVIRONMENT_RELEASE    Valid values are dev/prod
  MAFIPY_USERNAME_DEV    Username for test pypi.org
  MAFIPY_PASSWORD_DEV    Password for test pypi.org
  MAFIPY_USERNAME_PROD    Username for pypi.org
  MAFIPY_PASSWORD_PROD    Password for pypi.org
  MAFIPY_GITHUB_API_TOKEN    GitHub API token.
EOF
}

#
# validate arugments
#
readonly PATH_TO_REPOSITORY=$1
if [ -z ${PATH_TO_REPOSITORY+x} ]; then
  echo "Arguments are required."
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
  echo "Invalid value of ENVIRONMENT_RELEASE=${ENVIRONMENT_RELEASE}."
  usage
  exit 1
fi

cd ${PATH_TO_REPOSITORY}
# required MAFIPY_USERNAME and MAFIPY_PASSWORD, MAFIPY_GITHUB_API_TOKEN
export MAFIPY_USERNAME
export MAFIPY_PASSWORD
echo "${PATH_TO_REPOSITORY}/scripts/release.sh ${ARGS} --upload"
bash ${PATH_TO_REPOSITORY}/scripts/release.sh \
  ${ARGS} \
  --upload
