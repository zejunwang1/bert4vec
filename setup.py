#!usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='bert4vec',
    version='1.0.0',
    description='Chinese Sentence Embeddings using SimBERT / RoFormer-Sim / Paraphrase-Multilingual-MiniLM',
    long_description='bert4vec: https://github.com/zejunwang1/bert4vec',
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
