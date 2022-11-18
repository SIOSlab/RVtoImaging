"""
Sets up RVtools, a package to make it easier to create precursor
radial velocity measurements for direct imaging purposes
"""

from setuptools import find_packages, setup

setup(
    name="RVtools",
    version="0.1",
    packages=find_packages(include=["RVtools", "RVtools.*"]),
    install_requires=[
        "astropy",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
    ],
)
