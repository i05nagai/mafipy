ARG DOCKER_PREFIX
FROM ${DOCKER_PREFIX}mafipy-core:latest

ENV PYTHON_VERSION=3.6.5
RUN \
    apt-get update \
    && apt-get install -y \
        build-essential \
        checkinstall \
        libreadline-gplv2-dev \
        libncursesw5-dev \
        libssl-dev \
        libsqlite3-dev \
        tk-dev \
        libgdbm-dev \
        libc6-dev \
        libbz2-dev \
        curl \
    && cd /usr/src \
    && curl -L -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make install \
    # pip
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py pip==22.3.1 \
    # cleanup
    && /opt/mafipy/core/build_cleanup.sh

RUN \
    apt-get update \
    && apt-get install -y \
        git \
    && ln -s /usr/local/bin/python3 /usr/local/bin/python \
    # cleanup
    && /opt/mafipy/core/build_cleanup.sh
