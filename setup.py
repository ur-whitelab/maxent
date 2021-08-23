import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="maxent",
    version="1.0",
    url="https://github.com/ur-whitelab/maxent",
    license="GPL v2",
    author="Rainier Barret, Mehrad Ansari, Andrew D White",
    author_email="andrew.white@rochester.edu",
    description="Maximum entropy inference Keras implementation",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=["numpy", "tensorflow < 2.6", "tensorflow_probability"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GPL v2 License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
