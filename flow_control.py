import os
import sys
from settings import *

def log(msg): # For printing debugging messages
    if (DEBUG_MODE == True):
        print(msg)

# Disable prints
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore prints
def enablePrint():
    sys.stdout = sys.__stdout__
    

