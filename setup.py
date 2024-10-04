from setuptools import setup, find_packages

setup(
    name='CellAlignDTW',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scanpy',
        'tslearn',
        'matplotlib',
        'scipy'
    ],
)
