"""
Helper to import pyro_webppl.py without dealing with paths.

"""
import os
import sys

EXAMPLES_PATH = os.path.realpath(__file__)
PROBMODS_PATH = '/'.join(EXAMPLES_PATH.split('/')[:-2])

sys.path.append(PROBMODS_PATH + '/src')
from webppl import *
