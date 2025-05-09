from setuptools import setup, find_packages

setup(
    name="solar",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "sunpy",
        "astropy",
    ],
)