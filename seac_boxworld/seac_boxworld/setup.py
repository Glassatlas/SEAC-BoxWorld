import setuptools
from setuptools import setup

setup(
    name='seac',
    version='0.0.1',
    install_requires=['stable_baselines3', 'tensorboard'],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
