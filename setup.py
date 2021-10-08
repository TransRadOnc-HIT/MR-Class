# -*- coding: utf-8 -*-


from setuptools import setup, find_packages



pkgs = ['nibabel',
	'pydicom',
	'opencv-python',
	'torchvision==0.8.2',
	'matplotlib',
	'scipy',
	'PyYAML',
	'pandas',
	'scikit-image',
	'torch==1.7.1',
	'medpy']

setup(name='MR_CLASS',
      versiom='0.1.0',
      description='MR_contrast classifier',
      url='https://github.com/pgsalome/mrclass',
      python_requires='>=3.5',
      author='Patrick Salome',
      author_email='p.salome@dkfz.de',
      license='Apache 2.0',
      zip_safe=False,
      install_requires=pkgs,
      packages=find_packages(exclude=['docs', 'tests*']),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )
