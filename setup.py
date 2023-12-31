from setuptools import setup, find_packages

setup(
    name='splinecloud-scipy',
    version='0.0.2',
    author = "Vadym Pasko",
    author_email = "vadym@splinecloud.com",
    description = "A SplineCloud client based on SciPy.",
    license = "MIT",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'requests',
    ],
)
