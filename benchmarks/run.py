#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="This script is ...")
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help='debug mode if this flag is set (default: False)')
    parser.add_argument("asv_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.exit(run_asv(args.asv_command))


def run_asv(args):
    cwd = os.path.abspath(os.path.dirname(__file__))
    cmd = ["asv"] + list(args)
    env = dict(os.environ)

    # Run
    try:
        return subprocess.call(cmd, env=env, cwd=cwd)
    except OSError as err:
        if err.errno == 2:
            print("Error when running '%s': %s\n" % (" ".join(cmd), str(err),))
            print("You need to install Airspeed Velocity https://spacetelescope.github.io/asv/")
            print("to run Scipy benchmarks")
            return 1
        raise


if __name__ == '__main__':
    main()
