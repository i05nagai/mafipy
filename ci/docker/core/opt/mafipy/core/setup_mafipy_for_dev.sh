#!/bin/bash

pyenv activate ${MAFIPY_VIRTUALENV}
git clone https://github.com/i05nagai/mafipy.git
cd mafipy
git submodule init
git submodule update

pip install --disable-pip-version-check -r requirements.txt
# pip install --disable-pip-version-check -r requirements-dev.txt
