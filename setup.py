from setuptools import setup, find_packages


with open('jelli/_version.py') as f:
    exec(f.read())

setup(
  name='jelli',
  version=__version__,
  author='Aleks Smolkovic <aleks.smolkovic@unibas.ch>, Peter Stangl <peter.stangl@cern.ch>',
  license='MIT',
  packages=find_packages(),
  install_requires=[
      'jax',
  ]
)