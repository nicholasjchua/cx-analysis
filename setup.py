#!/usr/bin/env python

from setuptools import setup, find_packages
import os
from os import path
import sys


here = path.abspath(path.dirname(__file__))

with open('README.md', 'r') as rmf:
    readme = rmf.read()

############
# This stanza asks for caiman datafiles (demos, movies, ...) to be stashed in "share/caiman", either
# in the system directory if this was installed with a system python, or inside the virtualenv/conda
# environment dir if this was installed with a venv/conda python. This ensures:
# 1) That they're present somewhere on the system if Caiman is installed this way, and
# 2) We can programmatically get at them to manage the user's conda data directory.
#
# We can access these by using sys.prefix as the base of the directory and constructing from there.
# Note that if python's packaging standards ever change the install base of data_files to be under the
# package that made them, we can switch to using the pkg_resources API.

binaries = []
extra_dirs = []
data_files = []

############

setup(
    name='cx_analysis',
    version='0.0.1',
    author='Nicholas Chua',
    author_email='nchua@flatironinstitute.org',
    url='https://github.com/nicholasjchua/cx-analysis',
    description='Analysis scripts for connectomics research.',
    long_description=readme,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    keywords='neuroscience analysis connectomics catmaid',
    packages=find_packages(),
    install_requires=['']
)
