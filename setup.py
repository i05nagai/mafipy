#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools.command.test import test as TestCommand


NAME = "mafipy"
MAINTAINER = "i05nagai"
MAINTAINER_EMAIL = ""
DESCRIPTION = """ """
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
LICENSE = ""
URL = ""
VERSION = "0.0.1"
DOWNLOAD_URL = ""
CLASSIFIERS = """ \
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Programming Language :: Python :: 3.5
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: MacOS
"""


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ["mafipy", "-v", "--cov=mafipy"]
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)


def main():
    cmdclass = {'tests': PyTest}
    metadata = dict(
        name=NAME,
        packages=[NAME],
        version=VERSION,
        description=DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        long_description=LONG_DESCRIPTION,
        tests_require=['pytest'],
        cmdclass=cmdclass
    )

    setup(**metadata)


if __name__ == '__main__':
    main()
