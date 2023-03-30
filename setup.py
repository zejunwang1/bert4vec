#!usr/bin/env python
# -*- coding:utf-8 -*-

import io
from setuptools import setup, find_packages

def _get_readme():
    """
    Use pandoc to generate rst from md.
    pandoc --from=markdown --to=rst --output=README.rst README.md
    """
    with io.open("README.rst", encoding='utf-8') as fid:
        return fid.read()

setup(
    name='bert4vec',
    version='0.3.0',
    description='Chinese Sentence Embeddings using SimBERT / RoFormer-Sim / Paraphrase-Multilingual-MiniLM',
    long_description=_get_readme(),
    license='Apache License 2.0',
    url='https://github.com/zejunwang1/bert4vec',
    author='wangzejun',
    author_email='wangzejunscut@126.com',
    install_requires=[
        'transformers>=4.6.0,<5.0.0',
        'torch>=1.6.0',
        'numpy',
        'huggingface-hub'
    ],
    packages=find_packages()
)
