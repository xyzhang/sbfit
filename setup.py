from setuptools import setup, find_packages
import sbfit

setup(
    name='sbfit',
    version=sbfit.__version__,
    packages=find_packages(),
    url='https://gitlab.sron.nl/asg/x-ray-extended/sbfit',
    license='MIT License',
    install_requires=["numpy", "scipy", "numba", "emcee", "astropy", "matplotlib", "corner", "pyregion"],
    python_requires=">=3.7",
    author='Xiaoyuan Zhang',
    author_email='x.zhang@sron.nl',
    description='X-ray extended source surface brightness fitting toolkit.',
)
