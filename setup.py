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
        'scipy',
        'pygam',
        'tqdm'
    ],
    tests_require=[
        'pytest',
        'pytest-cov'
    ],
    python_requires='>=3.7',
)
