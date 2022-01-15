from setuptools import setup, find_packages
import pathlib

import sbfit

setup(
    name='sbfit',
    version=sbfit.__version__,
    packages=find_packages(),
    url='https://gitlab.sron.nl/asg/x-ray-extended/sbfit',
    license='MIT License',
    install_requires=["numpy<=1.20",
                      "scipy",
                      "numba==0.54",
                      "emcee>=3.0",
                      "astropy>=4.2",
                      "matplotlib>=3.1",
                      "corner",
                      "pyregion>=2.0"],
    python_requires=">=3.7",
    author='Xiaoyuan Zhang',
    author_email='x.zhang@sron.nl',
    description='Astronomical X-ray sources surface brightness profile fitting package',
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown"
)
