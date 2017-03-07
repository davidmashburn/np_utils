from distutils.core import setup

# Read the version number
with open("np_utils/_version.py") as f:
    exec(f.read())

setup(
    name='np_utils',
    version=__version__, # use the same version that's in _version.py
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
                      'future>=0.16',
                     ],
)
