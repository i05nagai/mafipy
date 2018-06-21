"""
Support only python 3.5+.

Usage
=====

$ pip install -r scripts/release/requirements.txt
$ export GITHUB_API_KEY=''
$ python scripts/reelase/release.py release --commitish master --path /path/to/asset --tag v0.1 --repository mafipy

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import github
import mimetypes
import os
import sys
try:
    from urllib.request import pathname2url
except ImportError:
    from urllib import pathname2url


def guess_type(path):
    url = pathname2url(path)
    content_type, encoding = mimetypes.guess_type(url)
    if content_type is None:
        content_type = 'application/octet-stream'
    return content_type


def get_filename(path):
    return os.path.basename(path)


def _is_dir(path):
    return os.path.isdir(path)


def _to_abspath(path):
    cwd = os.getcwd()
    # relative path
    if not os.path.isabs(path):
            path = os.path.join(cwd, path)
    return path


def _get_path_from_dir(path):
    for root, subdirs, files in os.walk(path):
        for f in files:
            path_to_file = os.path.join(root, f)
            yield path_to_file


def dir_to_path(path):
    path = _to_abspath(path)
    # if directory return files in the directory
    # otherwise return path
    if _is_dir(path):
        for p in _get_path_from_dir(path):
            yield p
    else:
        yield path


class Github(object):

    TIMEOUT = 60

    def __init__(self, api_key, repository_name='', repository_type='public'):
        self.github = github.Github(api_key, timeout=self.TIMEOUT)
        self.repository = self.get_repository(repository_name, repository_type)

    def get_repository(self, repository_name, repository_type='public'):
        repos = self.github.get_user().get_repos(type=repository_type)
        for _repo in repos:
            if _repo.name == repository_name:
                repo = _repo
                break

        if repo is None:
            msg = 'repository {0} does not exist'.format(repository_name)
            raise ValueError(msg)
        return repo

    def create_release(self, tag_name='v0.1', target_commit='master', prerelease=False):
        """create_release
        https://developer.github.com/v3/repos/releases/#create-a-release
        http://pygithub.readthedocs.io/en/latest/github_objects/Repository.html#github.Repository.Repository.create_git_release

        :param tag_name:
        :param target_commit:
        """
        # name of relase
        name = tag_name
        # description of release
        body = ''
        # unblished release
        draft = False
        github_release = self.repository.create_git_release(
            tag_name,
            name,
            body,
            draft=draft,
            prerelease=prerelease,
            target_commitish=target_commit)
        return github_release

    def upload_asset(self, release, paths):
        git_assets = []
        for _path in paths:
            for path in dir_to_path(_path):
                print('Uploading: {0}'.format(path))
                content_type = guess_type(path)
                label = get_filename(path)
                git_asset = release.upload_asset(path, label, content_type)
                git_assets.append(git_asset)
        return git_assets


def release(args):
    if 'GITHUB_API_KEY' in os.environ:
        token = os.environ['GITHUB_API_KEY']
    else:
        token = args.token

    if token is None or token == '':
        msg = '--token <token> or GITHUB_API_KEY environment variable required'
        raise ValueError(msg)

    github = Github(token, repository_name=args.repository)
    release = github.create_release(args.tag)
    git_assets = github.upload_asset(release, args.path)
    print(git_assets)


def add_subparser_release(subparsers):
    # create the subcommand parser
    subparser = subparsers.add_parser(
        'release',
        help='Make a release in GitHub with assets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.set_defaults(func=release)
    # required
    subparser.add_argument(
        '--repository',
        required=True,
        type=str,
        help='Repository name.')
    subparser.add_argument(
        '--commitish',
        required=True,
        default='master',
        type=str,
        help='Commit SHA1 or branch name which the tag is added to.')
    subparser.add_argument(
        '--tag',
        required=True,
        type=str,
        help='Name of tag and release. e.g. v0.1')
    # not requiredd
    subparser.add_argument(
        '--path',
        action='append',
        type=str,
        help='Paths to asset. This can specify multiple times.')
    subparser.add_argument(
        '--prerelease',
        action='store_true',
        default=False,
        help='Set if prerelease.')
    subparser.add_argument(
        '--draft',
        action='store_true',
        default=False,
        help='Set if draft.')
    subparser.add_argument(
        '--token',
        type=str,
        help='Github API token. Required to set thid option or set GITHUB_API_KEY environ variables.')


def make_parser():
    parser = argparse.ArgumentParser(
        description="Release helper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.set_defaults(help)
    # sub parsers
    subparsers = parser.add_subparsers(
        help='subcommands')
    add_subparser_release(subparsers)
    return parser


def main():
    parser = make_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
