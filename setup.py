from setuptools import find_packages, setup

setup(
    name='exclusiveAI',
    packages=find_packages(include=['numpy']),
    version='0.0.1',
    description='A simple neural network library',
    author='Francesco Paolo Liuzzi & Paul Maximilian Magos',
    test_suite='eclusiveAITest',
)