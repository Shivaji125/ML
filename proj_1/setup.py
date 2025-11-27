from setuptools import setup, find_packages

setup(
    name='my_ml_project',
    version='0.1.0',
    packages=find_packages(), # Automatically finds 'src' and sub-packages
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'PyYAML',
    ],
    description='A modular template for machine learning projects.',
    author='Your Name',
)