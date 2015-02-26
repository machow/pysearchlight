from setuptools import setup

setup(name='pysearchlight',
      version='0.1.0',
      description='Simple fMRI searchlight tool',
      url='http://github.com/machow/pysearchlight',
      author='Michael Chow',
      author_email='machow@princeton.edu',
      packages=['pysearchlight'],
      entry_points = {
          'console_scripts': ['pysearch=pysearchlight.search:main']
          },
      zip_safe=False)
