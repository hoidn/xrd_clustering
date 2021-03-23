import sys
import os
import pip

from setuptools import setup, find_packages

#pip.main(['install',  'pyzmq', '--install-option=--zmq=bundled'])

setup(name='xrdc',
    version='1.0',
    packages = find_packages('.'),
    package_dir={'xrdc': 'xrdc'},
    package_data={'xrdc': ['inputs/*']},
#    scripts = [
#        'scripts/mecana.py', 'scripts/logbooksync.py'
#    ],
#    install_requires = ['recordclass', 'google-api-python-client', 'httplib2', 'atomicfile', 'urllib3', 'gspread', 'requests>=2.9.1', 'multiprocess', 'dill', 'pox', 'ppft', 'ipdb', 'joblib', 'mpi4py', 'h5py'],
    zip_safe = False,
    )

#pip.main(['install', 'git+https://github.com/uqfoundation/pathos.git'])
#pip.main(['install', 'pymongo'])
#pip.main(['install', 'pytest'])

#print  "Packages are: ", find_packages('.')
