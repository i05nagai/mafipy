.. image:: https://travis-ci.org/i05nagai/mafipy.svg?branch=master
   :target: https://travis-ci.org/i05nagai/mafipy

.. image:: https://codeclimate.com/github/i05nagai/mafipy/badges/gpa.svg
   :target: https://codeclimate.com/github/i05nagai/mafipy
   :alt: Code Climate

.. image:: https://codeclimate.com/github/i05nagai/mafipy/badges/coverage.svg
   :target: https://codeclimate.com/github/i05nagai/mafipy/coverage
   :alt: Test Coverage

.. image:: https://codeclimate.com/github/i05nagai/mafipy/badges/issue_count.svg
   :target: https://codeclimate.com/github/i05nagai/mafipy
   :alt: Issue Count

.. image:: https://badges.gitter.im/mafipy/Lobby.svg
   :alt: Join the chat at https://gitter.im/mafipy/Lobby
   :target: https://gitter.im/mafipy/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge


mafipy
======

Description
============

mathmatical finance in python.
Supported methods are belows:

* replication method

  * QuantoCMS

* Analytic formula

  * Black Scholes

  * Black 

  * SABR

Installation
============

.. code:: shell

   git clone https://github.com/i05nagai/mafipy.git
   cd mafipy
   pip install -r requirements.txt


Test
======

To run tests, you additionally need to install `pytest`.

.. code:: shell

   pip install pytest
   python setup.py test


Benchmarks
==========

`asv` is required to execute benchmarks.
You can install `asv` by `pip`.

.. code:: shell

   pip install asv

Then 

.. code:: shell

   # execute benchmarks
   python setup.py benchmark
   # generate html file from the results
   python setup.py benchmark_publish
   # preview the generated html through local server
   python setup.py benchmark_preview


Related Projects
================
* `GitHub - lballabio/QuantLib: The QuantLib C++ library <https://github.com/lballabio/QuantLib>`_

  * One of the best library for mathmatical finance.
    The library is written in C++. 
    There are many wrapper projects of the QuantLib.
* `GitHub - finmath/finmath-lib: Mathematical Finance Library: Algorithms and methodologies related to mathematical finance. <https://github.com/finmath/finmath-lib>`_

  * Mathematical Finance Library: Algorithms and methodologies related to mathematical finance.
    The library is written in Java.


Documentation
=============
* `API document`_ 

  .. _API document: https://i05nagai.github.io/mafipy_docs/html/

* `Benchmarks`_

  .. _`Benchmarks`: https://i05nagai.github.io/mafipy_benchmarks/html/
