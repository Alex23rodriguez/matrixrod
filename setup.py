from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='matrixrod',
    url='https://github.com/Alex23rodriguez/matrixrod.git',
    author='Alex Rodriguez',
    author_email='alex.rodriguez.oro@gmail.com',
    # Needed to actually package something
    packages=['matrixrod'],
    # Needed for dependencies
    install_requires=[],
    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='MIT',
    description='Python package with functionality for matrix and vector manipulation',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
