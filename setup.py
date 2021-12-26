import numpy
from setuptools import setup, find_packages

setup(
    name='lintinv',
    version='0.0.1',
    description='3D Dark Map',
    author='Xiangchong Li et al.',
    author_email='mr.superonion@hotmail.com',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
    ],
    include_dirs=numpy.get_include(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
