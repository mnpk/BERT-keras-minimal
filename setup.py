#!/usr/bin/env python

from distutils.core import setup

from setuptools import find_packages

setup(
    name='bertkeras',
    version='1.0',
    description='Python Distribution Utilities',
    author='Max Jeblick',
    author_email='',
    url='https://github.com/bindung/BERT-keras-minimal.git',
    pakcages=find_packages(exclude=["tests", "tests.*", "examples"]),
)
