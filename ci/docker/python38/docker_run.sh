#!/bin/bash

set -x
PATH_TO_REPOSITORY=$(cd $(dirname ${0})/../../..;pwd)

docker run --rm -it \
  --volume ${PATH_TO_REPOSITORY}:/root/project \
  --workdir /root/project \
  i05nagai/mafipy-python38:latest \
  /bin/bash
