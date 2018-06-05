#!/bin/bash

set -e

usage() {
  cat <<EOF
packaging.sh is a tool for packaging mafipy and uploading the package to PYPI.

Usage:
    packaging.sh <options>

Options:
  --no-test     upload to real PyPI.


Environment variables:
  MAFIPY_USERNAME   username of pypi.org
  MAFIPY_PASSWORD   password of pypi.org
  MAFIPY_GITHUB_API   password of pypi.org
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
if [ -z ${MAFIPY_USERNAME+x} ]; then
  echo "You need to export environment variable MAFIPY_USERNAME"
  echo ""
  usage
  exit 1
fi
if [ -z ${MAFIPY_PASSWORD+x} ]; then
  echo "You need to export environment variable MAFIPY_PASSWORD"
  echo ""
  usage
  exit 1
fi

readonly PATH_TO_REPOSITORY=$(cd $(dirname ${0});cd ..;pwd)

#
# flag
#
if [ ! $is_no_test ]; then
  args="${args} --repository-url https://test.pypi.org/legacy/"
fi

#
# do pakaging
#
cd ${PATH_TO_REPOSITORY}
python setup.py sdist
python setup.py bdist_wheel
twine upload  \
  ${args} \
  --username ${MAFIPY_USERNAME} \
  --password ${MAFIPY_PASSWORD} \
  dist/*
