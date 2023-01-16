#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import Command
from setuptools.command.test import test as TestCommand
import distutils.core as core
import os
import setuptools
import subprocess
import sys
import textwrap


import mafipy

VERSION = mafipy.__version__
NAME = "mafipy"
MAINTAINER = "i05nagai"
MAINTAINER_EMAIL = ""
DESCRIPTION = """ """
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
LICENSE = ""
URL = ""
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
    """
    """
    cov_report_desc = """type of report to generate:
    term, term-missing, annotate, html, xml.
    Comma separated multiple types is allowed (e.g. --cov-report=xml,html).
    term, term missing may be followed by ':skip-covered'.
    annotate, html and xml may be followed by ':DEST'
    where DEST specifies the output location.
    """
    user_options = [
        ('cov=', '-', "coverage target."),
        ('cov-report=', '-', textwrap.dedent(cov_report_desc)),
        ('pdb', '-', "start the interactive Python debugger on errors."),
        ('pudb', '-', "start the PuDB debugger on errors."),
        ('quiet', 'q', "decrease verbosity."),
        ('verbose', 'v', "increase verbosity."),
        # collection:
        ('doctest-modules', '-', "run doctests in all .py modules"),
    ]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.cov = ''
        self.cov_report = ''
        self.pdb = ''
        self.pudb = ''
        self.quiet = ''
        self.verbose = ''
        # collection:
        self.doctest_modules = ''
        # for benchmarks tests
        self.bench = ''
        self.test_args = ["mafipy"]
        self.test_suite = True

    def finalize_options(self):
        TestCommand.finalize_options(self)
        # if cov option is specified, option is replaced.
        if self.cov:
            self.test_args += ["--cov={0}".format(self.cov)]
        else:
            self.test_args += ["--cov=mafipy"]
        # user_options doesn't support multiple valued options
        if self.cov_report is not None:
            options = self.cov_report.split(',')
            template = '--cov-report={0}'
            self.test_args += [template.format(op) for op in options]
        if self.pdb:
            self.test_args += ["--pdb"]
        if self.pudb:
            self.test_args += ["--pudb"]
        if self.quiet:
            self.test_args += ["--quiet"]
        if self.verbose:
            self.test_args += ["--verbose"]
        if self.doctest_modules:
            self.test_args += ["--doctest-modules"]

    def run_tests(self):
        import pytest

        print("executing 'pytest {0}'".format(" ".join(self.test_args)))
        errno = pytest.main(self.test_args)
        sys.exit(errno)


class Benchmark(Command):
    """
    asv run command
    """

    user_options = [
        ('run-type=', '-', "execcute asv run <run-type>. One of NEW, ALL."),
        #
        ('bench=', 'b', "Regular expression(s) for benchmark to run. When not\
                        provided, all benchmarks are run."),
        ('config=', '-', "Benchmark configuration file"),
        ('dry-run', 'n', "Do not save any results to disk."),
        ('skip-existing', 'k', "Skip running benchmarks that have previous\
                                successful or failed results"),
        ('steps=', 's', "Maximum number of steps to benchmark. This is used to\
                        subsample the commits determined by range to a\
                        reasonable number."),
        ('verbose', 'v', "Increase verbosity"),
    ]

    def initialize_options(self):
        self.run_type = None
        self.bench = None
        self.config = None
        self.dry_run = None
        self.skip_existing = None
        self.steps = None
        self.verbose = None
        self.args = []

    def finalize_options(self):
        pass

    def run(self):
        if self.run_type is not None:
            self.args += ["{0}".format(self.run_type)]
        if self.bench is not None:
            self.args += ["--bench={0}".format(self.bench)]
        if self.config is not None:
            self.args += ["--config={0}".format(self.config)]
        else:
            self.args += ["--config=benchmarks/asv.conf.json"]
        if self.dry_run is not None:
            self.args += ["--dry-run"]
        if self.skip_existing is not None:
            self.args += ["--skip-existing"]
        if self.steps is not None:
            self.args += ["--steps={0}".format(self.steps)]
        if self.verbose is not None:
            self.args += ["--verbose"]

        sys.exit(self.run_asv(self.args))

    def run_asv(self, args):
        # current working directory
        cwd = os.path.abspath(os.path.dirname(__file__))
        cmd = ["asv", "run"] + list(args)
        env = dict(os.environ)
        # essential to execute benchmarks withou installing this packages
        env["PYTHONPATH"] = cwd

        try:
            print("executing '{0}'".format(" ".join(cmd)))
            return subprocess.call(cmd, env=env, cwd=cwd)
        except OSError as err:
            if err.errno == 2:
                print("Error when running '%s': %s\n" % (" ".join(cmd), str(err),))
                print("You need to install Airspeed Velocity \
                      https://spacetelescope.github.io/asv/")
                print("to run Scipy benchmarks")
                return 1
            raise err


class BenchmarkPublish(Command):
    """
    asv publish
    """

    user_options = [
        # for benchmark tests
    ]

    def initialize_options(self):
        self.args = []

    def finalize_options(self):
        pass

    def run(self):
        sys.exit(self.publish_asv(self.args))

    def publish_asv(self, args):
        # current working directory
        root_dir = os.path.dirname(__file__)
        # output directory of html is asv_files/html from current directory.
        # This configuration is written in asv.conf.json
        cwd = os.path.abspath(os.path.join(root_dir, "benchmarks"))
        cmd = ["asv", "publish"] + list(args)
        env = dict(os.environ)

        try:
            print("executing '{0}'".format(" ".join(cmd)))
            return subprocess.call(cmd, env=env, cwd=cwd)
        except OSError as err:
            if err.errno == 2:
                print("Error when running '%s': %s\n" % (" ".join(cmd), str(err),))
                print("You need to install Airspeed Velocity \
                      https://spacetelescope.github.io/asv/")
                print("to run Scipy benchmarks")
                return 1
            raise err


class BenchmarkPreview(Command):
    """
    asv preview
    """

    user_options = [
        # for benchmark tests
    ]

    def initialize_options(self):
        self.args = []

    def finalize_options(self):
        pass

    def run(self):
        sys.exit(self.preview_asv(self.args))

    def preview_asv(self, args):
        # current working directory
        root_dir = os.path.dirname(__file__)
        # output directory of html is asv_files/html from current directory.
        # This configuration is written in asv.conf.json
        cwd = os.path.abspath(os.path.join(root_dir, "benchmarks"))
        cmd = ["asv", "preview"] + list(args)
        env = dict(os.environ)

        try:
            print("executing '{0}'".format(" ".join(cmd)))
            return subprocess.call(cmd, env=env, cwd=cwd)
        except OSError as err:
            if err.errno == 2:
                print("Error when running '%s': %s\n" % (" ".join(cmd), str(err),))
                print("You need to install Airspeed Velocity \
                      https://spacetelescope.github.io/asv/")
                print("to run Scipy benchmarks")
                return 1
            raise err


def parse_setuppy_commands(args):
    """Check the commands and respond appropriately.  Disable broken commands.
    Return a boolean value for whether or not to run the build or not (avoid
    parsing Cython and template files if False).
    """
    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg', 'build_sphinx',
                     'test')

    for command in good_commands:
        if command in args:
            return True

    bad_commands = dict(
        upload="""
            `setup.py upload` is not supported, because it's insecure.
            Instead, build what you want to upload and upload those files
            with `twine upload -s <filenames>` instead.
            """,
        clean="""
            `setup.py clean` is not supported, use one of the following instead:
              - `git clean -xdf` (cleans all files)
              - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                  files that aren't checked into the git repo)
            """,
    )
    not_supported_commands = [
        'upload_docs',
        'easy_install',
        'bdist',
        'bdist_dumb',
        'register',
        'check',
        'install_data',
        'install_headers',
        'install_lib',
        'install_scripts',
        'flake8',
    ]
    for command in not_supported_commands:
        bad_commands[command] = "`setup.py %s` is not supported" % command
    for command in bad_commands.keys():
        if command in args:
            print(textwrap.dedent(bad_commands[command]) +
                  "\nAdd `--force` to your command to use it anyway if you "
                  "must (unsupported).\n")
            sys.exit(1)


def get_ext_modules():
    import Cython.Build
    ext_sobol = core.Extension(
        'mafipy.math.qmc._sobol',
        ['mafipy/math/qmc/*.pyx'],
    )
    extensions = [
        ext_sobol,
    ]
    ext_modules = Cython.Build.cythonize(extensions)
    return ext_modules


def get_packages():
    packages = setuptools.find_packages()
    return packages


def get_install_requires():
    return [
        'numpy==1.21.0',
        'scipy==1.10.0',
    ]


def main():
    cmdclass = {
        'test': PyTest,
        'benchmark': Benchmark,
        'benchmark_publish': BenchmarkPublish,
        'benchmark_preview': BenchmarkPreview,
    }

    metadata = dict(
        name=NAME,
        packages=get_packages(),
        version=VERSION,
        description=DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        install_requires=get_install_requires(),
        license=LICENSE,
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        long_description=LONG_DESCRIPTION,
        tests_require=['pytest', 'pytest-cov'],
        cmdclass=cmdclass,
    )
    args = sys.argv[1:]
    build_required = parse_setuppy_commands(args)

    if build_required:
        ext_modules = get_ext_modules()
        metadata['ext_modules'] = ext_modules

    core.setup(**metadata)


if __name__ == '__main__':
    main()
