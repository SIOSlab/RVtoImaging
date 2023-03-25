"""
Sets up RVtoImaging, a package to make it easier to create precursor
radial velocity measurements for direct imaging purposes
"""

from setuptools import find_packages, setup

setup(
    name="RVtoImaging",
    version="0.1",
    packages=find_packages(include=["RVtoImaging", "RVtoImaging.*"]),
    install_requires=[
        "astropy",
        "keplertools",
        "matplotlib",
        "numpy",
        "pandas",
        "rebound",
        "scipy",
        "tqdm",
        "xarray",
    ],
)
