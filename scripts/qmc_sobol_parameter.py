"""
This scripts download direction number and format it for cython program.

Usage
=====
python scripts/qmc_sobol_parameter.py --initial_number=joe_kuo_d6
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    import urllib.request as request
except Exception:
    import urllib2 as request
import argparse
import re


def add_indent(array, num_indent=1):
    indents = '    ' * num_indent
    return ['{0}{1}'.format(indents, a) for a in array]


def array_to_str(array, num_indent=1):
    array = add_indent(array, num_indent)
    return '\n'.join(array)


def retrieve_file(url):
    response = request.urlopen(url)
    return response.read().decode('utf-8')


def download_file(url, filename):
    request.urlretrieve(url, filename)


def convert_polynomial(degree, polynomial_num):
    """convert_irreducible_polynomial

    Example
    =======
    >>> degree = 8
    >>> polynomial_num = 14
    # coeffs = (1(001110)1)
    >>> 0b10011101 = convert_polynomial(degree, polynomial_num)
    """
    return ((1 << degree)
            +
            (polynomial_num << 1)
            +
            1)


def format_initial_number_joe_kuo_d6_21201(lines):
    """format_joe_kuo_21201

    :param lines:
    """
    lines = lines.rstrip()

    variables = []
    definitions = []
    template_variable = 'joe_kuo_d6_initial_number{0:05d}'
    template_initial_number = 'static int {0}[] = {1};'
    for idx, line in enumerate(lines.split('\n')[1:]):
        # dim, degree, coeff, initial_direction_number
        cols = re.split(r'\s\s+', line)
        # initlal number
        initial_number = ', '.join(cols[3].split())
        initial_number = '{' + initial_number + ', 0}'
        # variable
        variable = template_variable.format(idx + 1)
        variables.append(variable + ',')
        initial_number = template_initial_number.format(variable, initial_number)
        definitions.append(initial_number)

    variables = array_to_str(variables)
    variable_definition = array_to_str(definitions, 0)
    return """
// m[k][j]: k-th initial number, j-dim
// m[1][1] = 1
// m[1][2] = 1, m_[2,2]
// up to 21201 dim

{variable_definition}
static int* joe_kuo_d6_initial_number[] = {left_bracket}
{variables}
{right_bracket};
""".format(variable_definition=variable_definition,
           variables=variables,
           left_bracket='{',
           right_bracket='}')


def format_irreducible_polynomial_joe_kuo_d6_21201(lines):
    """format_joe_kuo_21201

    :param lines:
    """
    lines = lines.rstrip()
    irreducibles = []
    for idx, line in enumerate(lines.split('\n')[1:]):
        # dim, degree, coeff, initial_direction_number
        cols = re.split(r'\s\s+', line)
        body = convert_polynomial(int(cols[1]), int(cols[2]))
        irreducibles.append(str(body) + ',')

    comment = """
// p_{j}: j-th irreducible poly, a_{j, k}: j-dim, k-th coeff
// 3 -> p_{1} = a_{1, 1}x + a_{1, 0} = 1x + 1
// 7 -> p_{2} = a_{2, 2}x^{2} + a_{2, 1} x + a_{2, 2} = 1x^{2} + 1x + 1
// up to 21201 dim
"""
    irreducibles = array_to_str(irreducibles, 1)
    return """
{comment}
static int joe_kuo_d6_irreducibles[] = {left_bracket}
{irreducibles}
{right_bracket};
    """.format(left_bracket='{',
               right_bracket='}',
               comment=comment,
               irreducibles=irreducibles)


def process_file(item):
    print('downloading ...')
    data = retrieve_file(item['url'])
    print('  Done!')
    print('Writing to {0}'.format(item['initial_number_filename']))
    print('Writing to {0}'.format(item['irreducible_filename']))

    initial_number = item['formatter'][0](data)
    irreducible = item['formatter'][1](data)
    try:
        with open(item['initial_number_filename'], 'w') as f:
            f.write(initial_number)
        with open(item['irreducible_filename'], 'w') as f:
            f.write(irreducible)
    except IOError as e:
        print(e)
    print('  Done!')


def main():
    JOE_KUO_D6 = {
        'url': 'http://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201',
        'initial_number_filename': 'joe_kuo_d6_21201_initial_number.h',
        'irreducible_filename': 'joe_kuo_d6_21201_irreducible.h',
        'formatter': [
            format_initial_number_joe_kuo_d6_21201,
            format_irreducible_polynomial_joe_kuo_d6_21201,
        ],
    }
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--initial_number',
        choices=['joe_kuo_d6'],
        required=True,
        help='type of initial number')
    args = parser.parse_args()

    print(args.initial_number)
    if args.initial_number == 'joe_kuo_d6':
        process_file(JOE_KUO_D6)


if __name__ == '__main__':
    main()
