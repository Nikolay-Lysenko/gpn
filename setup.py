"""
Just a regular `setup.py` file.

Author: Nikolay Lysenko
"""


import os
from setuptools import setup, find_packages


current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gpn',
    version='0.0.1',
    description='Generative Predictive Networks (GPN)',
    long_description=long_description,
    url='https://github.com/Nikolay-Lysenko/gpn',
    author='Nikolay Lysenko',
    author_email='nikolay-lysenco@yandex.ru',
    license='MIT',
    keywords='generative_models neural_networks gan',
    packages=find_packages(exclude=['tests', 'docs']),
    python_requires='>=3.6',
    install_requires=['tensorflow', 'numpy', 'PyYAML']
)
