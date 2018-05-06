#!/bin/sh

apt-get autoclean \
apt-get clean \
apt-get autoremove -y \
# Remove extraneous files
rm -rf /var/lib/apt/lists/* \
rm -rf /usr/share/man/* \
rm -rf /usr/share/info/* \
rm -rf /var/cache/man/* \
rm -rf /var/cache/apt/archives/*
# Clean up tmp directory
rm -rf /tmp/*
