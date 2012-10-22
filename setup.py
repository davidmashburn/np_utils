from distutils.core import setup

setup(
    name='np_utils',
    version='0.2.0',
    author='David N. Mashburn',
    author_email='david.n.mashburn@gmail.com',
    packages=['np_utils'],
    scripts=[],
    url='http://pypi.python.org/pypi/np_utils/',
    license='LICENSE.txt',
    description='',
    long_description=open('README.rst').read(),
    install_requires=[
                      'numpy>=1.0',
                      'scipy>=0.8',
                     ],
)
