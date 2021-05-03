"""Import this file to ensure the scripts in this folder can be run"""
import sys
import os

my_path = '..{}'.format(os.sep, os.sep)
sys.path.append(my_path)