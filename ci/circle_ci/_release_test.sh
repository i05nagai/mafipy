#!/bin/bash

#
# check and test ci/circle_ci/release.sh scripts.
#


#
# fill these values
#
export ENVIRONMENT_RELEASE="dev"

PATH_TO_REPOSITORY=$(cd $(dirname ${0});cd ../..;pwd)
cd ${PATH_TO_REPOSITORY}
bash ci/circle_ci/release.sh ${PATH_TO_REPOSITORY}
