# -*- coding: utf-8 -*-


import emcee
import warnings
from packaging import version




if version.parse(emcee.__version__) >= version.parse('3.0.0'):
    
    EMCEESUFFICIENTVERSION = True
    
else:
    
    EMCEESUFFICIENTVERSION = False
    
    warnings.warn("update emcee to at least v3.0.0 or some features may not work")
