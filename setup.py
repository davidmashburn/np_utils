from distutils.core import setup

setup(
    name='np_utils',
    version='0.2.1',
    author='David N. Mashburn',
    author_email='david.n.mashburn@gmail.com',
    packages=['np_utils'],
    scripts=[],
    url='http://pypi.python.org/pypi/np_utils/',
    license='LICENSE.txt',
    description='collection of utilities for array and list manipulation',
    long_description=open('README.rst').read(),
    install_requires=[
                      'numpy>=1.0',
                      'scipy>=0.8',
                     ],
)
