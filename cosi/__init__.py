# -*- coding: utf-8 -*-

import emcee
import warnings
from packaging import version

from .core import Probability, CosI




__all__ = ["Probability", "CosI"]




if version.parse(emcee.__version__) < version.parse('3.0.0'):
    warnings.warn("update emcee to at least v3.0.0 or some features may not work")
