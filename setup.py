import io
import os
import re

from setuptools import find_packages
from setuptools import setup

exec(open("maxent/version.py").read())


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="maxent",
    version=__version__,
    url="https://github.com/ur-whitelab/maxent",
    license="GPL v2",
    author="Mehrad Ansari <Mehrad.ansari@rochester.edu>, Rainier Barrett <rainier.barrett@gmail.com>, Andrew White <andrew.white@rochester.edu>",
    author_email="andrew.white@rochester.edu",
    description="Maximum entropy inference Keras implementation",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=["numpy", "tensorflow", "tensorflow_probability"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GPL v2 License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
