#!/bin/bash

#
# check and test ci/circle_ci/release.sh scripts.
#


#
# fill these values
#
ENVIRONMENT_RELEASE="dev"
MAFIPY_USERNAME_DEV=""
MAFIPY_PASSWORD_DEV=""
MAFIPY_USERNAME_PROD=""
MAFIPY_PASSWORD_PROD=""
MAFIPY_GITHUB_API_TOKEN=""

PATH_TO_REPOSITORY=$(cd $(dirname ${0});cd ../..;pwd)
cd ${PATH_TO_REPOSITORY}
bash ci/circle_ci/release.sh ${PATH_TO_REPOSITORY}
