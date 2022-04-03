import io
import os
import re

from setuptools import find_packages
from setuptools import setup

exec(open("maxent/version.py").read())


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="maxent-infer",
    version=__version__,
    url="https://github.com/ur-whitelab/maxent",
    license="GPL v2",
    author="Mehrad Ansari <Mehrad.ansari@rochester.edu>, Rainier Barrett <rainier.barrett@gmail.com>, Andrew White <andrew.white@rochester.edu>",
    author_email="andrew.white@rochester.edu",
    description="Maximum entropy inference Keras implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    install_requires=["numpy", "tensorflow", "tensorflow_probability"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
