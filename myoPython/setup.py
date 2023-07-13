# This file was auto-generated by Shut. DO NOT EDIT
# For more information about Shut, check out https://pypi.org/project/shut/

from __future__ import print_function
import io
import os
import setuptools
import sys

readme_file = 'README.md'
if os.path.isfile(readme_file):
  with io.open(readme_file, encoding='utf8') as fp:
    long_description = fp.read()
else:
  print("warning: file \"{}\" does not exist.".format(readme_file), file=sys.stderr)
  long_description = None

requirements = [
  'cffi >=1.11.5,<2.0.0',
  'six >=1.11.0,<2.0.0',
]

setuptools.setup(
  name = 'myo-python',
  version = '1.0.5',
  author = 'Niklas Rosenstein',
  author_email = 'rosensteinniklas@gmail.com',
  description = 'Python bindings for the Thalmic Labs Myo SDK.',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/NiklasRosenstein/myo-python',
  license = 'MIT',
  packages = setuptools.find_packages('.', ['test', 'test.*', 'tests', 'tests.*', 'docs', 'docs.*']),
  package_dir = {'': '.'},
  include_package_data = True,
  install_requires = requirements,
  extras_require = {},
  tests_require = [],
  python_requires = '>=3.5.0,<4.0.0',
  data_files = [],
  entry_points = {},
  cmdclass = {},
  keywords = [],
  classifiers = [],
  zip_safe = True,
)
