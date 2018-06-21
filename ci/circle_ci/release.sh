#!/bin/bash

set -e

usage() {
  cat <<EOF
release.sh is a tool for executing release scripts in Circle CI.
Create wheel and tarball.

Usage:
    release.sh <path-to-repository>

Environment variables
  ENVIRONMENT_RELEASE    Valid values are dev/prod
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
  # currently do nothing
elif [ "${ENVIRONMENT_RELEASE}" = "prod" ]; then
  # currently do nothing
else
  echo "Invalid value of ENVIRONMENT_RELEASE=${ENVIRONMENT_RELEASE}."
  usage
  exit 1
fi

cd ${PATH_TO_REPOSITORY}
echo "${PATH_TO_REPOSITORY}/scripts/release.sh"
bash ${PATH_TO_REPOSITORY}/scripts/release.sh
