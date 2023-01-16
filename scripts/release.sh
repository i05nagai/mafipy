#!/bin/bash

set -e

usage() {
  cat <<EOF
release.sh is a tool for release mafipy and uploading the package to PYPI.
Version is retrieved from VERSION constant in setup.py.

Usage:
  release.sh <options>
  # upload and release
  release.sh --no-test --upload

Options:
  --no-test     upload to PyPI.org if the flag is set. Otherwise, test.PyPI.org
  --upload      upload to PyPI

Environment variables:
  MAFIPY_USERNAME   username of pypi.org
  MAFIPY_PASSWORD   password of pypi.org
  MAFIPY_GITHUB_API_TOKEN GITHUB token. required if --upload is set.
EOF
}

while [ $# -gt 0 ];
do
  case ${1} in
    --debug|-d)
      set -x
    ;;

    --help|-h)
      usage
      exit 0
    ;;

    --no-test)
      is_no_test=true
    ;;

    --upload)
      upload=true
    ;;

    *)
      echo "[ERROR] Invalid option '${1}'"
      usage
      exit 1
    ;;
  esac
  shift
done

#
# validate environment variables
#
if [ -z ${MAFIPY_PASSWORD+x} ]; then
  echo "You need to export environment variable MAFIPY_PASSWORD"
  echo ""
  usage
  exit 1
fi

readonly PATH_TO_REPOSITORY=$(cd $(dirname ${0});cd ..;pwd)
cd ${PATH_TO_REPOSITORY}

#
# install dependency
#
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r scripts/release/requirements.txt

#
# create packages
#
cd ${PATH_TO_REPOSITORY}
python setup.py sdist
python setup.py bdist_wheel

#
# upload and release if needed
#
if [ $upload ]; then
  if [ ! $is_no_test ]; then
    args="${args} --repository-url https://test.pypi.org/legacy/"
  fi
  twine upload  \
    ${args} \
    --username ${MAFIPY_USERNAME} \
    --password ${MAFIPY_PASSWORD} \
    dist/*
  if [ ! $is_no_test ]; then
    # validate enviornment variables
    if [ -z ${MAFIPY_GITHUB_API_TOKEN+x} ]; then
      echo "You need to export environment variable MAFIPY_GITHUB_API_TOKEN"
      echo ""
      usage
      exit 1
    fi
    VERSION=`python -c 'import setup; print(setup.VERSION)' | tr -d '\n'`
    python scripts/reelase/release.py release \
      --commitish master \
      --path dist \
      --tag ${VERSION} \
      --repository mafipy
  fi
fi
