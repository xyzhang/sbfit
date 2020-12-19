from setuptools import setup, find_packages

setup(
    name='SBFit',
    version='0.1.0',
    packages=find_packages(),
    url='https://gitlab.sron.nl/asg/x-ray-extended/sbfit',
    license='MIT License',
    install_requires=["numpy", "scipy", "numba", "emcee", "astropy", "matplotlib", "corner"],
    python_requires=">=3.7",
    author='Xiaoyuan Zhang',
    author_email='x.zhang@sron.nl',
    description='X-ray extended source surface brightness fitting toolkit.',
)
