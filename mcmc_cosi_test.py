#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:41:07 2021

@author: mjfields
"""


import numpy as np
import pandas as pd
import emcee

import matplotlib.pyplot as plt



class Cosi(object):
    
    
    def __init__(self, vsini, e_vsini, rstar, e_rstar, prot, e_prot):
        
        self.vsini = vsini # km/s
        self.e_vsini = e_vsini
        
        self.rstar = rstar # R_Sun
        self.e_rstar = e_rstar
        
        self.prot = prot # days
        self.e_prot = e_prot
        
        
    def log_likelihood(self, theta):
        
        cosi, rstar, prot = theta
        
        
        sini = np.sqrt(1 - cosi**2)
        
        # line-of-sight velocity; km/s
        cv = 2 * np.pi * self.rstar * 6.957e5 / (self.prot * 24 * 3600)
        
        cvsini = cv * sini
        
        if cvsini > self.vsini:
            return -np.inf
        
        
        chi2 = (self.vsini - cvsini)**2 / self.e_vsini**2 + (rstar - self.rstar)**2 / self.e_rstar**2 + (prot - self.prot)**2 / self.e_prot**2
        
        return -0.5 * chi2
    
    
    def log_prior(self, theta):
        
        cosi, rstar, prot = theta
        
        if cosi < 0 or cosi > 1:
            return -np.inf
        
        if rstar < 0 or prot < 0:
            return -np.inf
        
        
        return 0.0
    
    
    def log_probability(self, theta):
        
        lp = self.log_prior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta)
    
    
    def run_mcmc(self, nwalkers, nsteps, progress=True):
        
        ndim = 3
        
        p = [0.5, self.rstar, self.prot]
        
        pos = p + 0.02 * np.random.randn(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        
        sampler.run_mcmc(pos, nsteps, progress=progress)
        
        return sampler
    
    
    
    
    def get_max_probability(self, chain):
        
        results = list(map(self.log_probability, [coord for coord in chain]))
        
        log_prob_array = np.array([float(r) for r in results])
        
        if np.any(np.isnan(log_prob_array)):
            raise ValueError("Probability function returned NaN")
            
        max_log_prob_index = np.argmax(log_prob_array)
        
        return chain[max_log_prob_index]
    
    
    
    
    
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'demi'

plt.rcParams['mathtext.bf'] = 'serif:demi'
plt.rcParams['mathtext.rm'] = 'serif:demi'

plt.rcParams['lines.linewidth'] = 3

plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.labelweight'] = 'demi'
plt.rcParams['axes.labelpad'] = 20.0
plt.rcParams['axes.labelsize'] = 13 
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.formatter.limits'] = [-4, 4]
plt.rcParams['axes.edgecolor'] = 'black'
 
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 2.5

plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.major.width'] = 2.5

plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.framealpha'] = 0.5
plt.rcParams['legend.loc'] = 'best'

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.titleweight'] = 'demi'

plt.rcParams['savefig.bbox'] = 'tight'
        
        
        







