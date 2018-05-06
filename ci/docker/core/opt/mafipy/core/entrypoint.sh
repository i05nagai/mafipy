#!/bin/bash

if [ ! -z ${MAFIPY_DEBUG+x} ]; then
  set -x
fi

exec "$@"
