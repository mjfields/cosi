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
        
        
        


# c = Cosi(7.3, 0.3, 0.912, 0.034, 6.45, 0.05)

# sampler = c.run_mcmc(100, 1500)

# chain = sampler.get_chain(discard=500, flat=True)

# cosi = chain[:, 0]

# i = np.arccos(cosi) * 180 / np.pi # degrees

# i95 = np.nanpercentile(i, 95)

# i_maxprob = np.arccos(c.get_max_probability(chain)[0]) * 180 / np.pi

# fig = plt.figure()

# plt.hist(i, bins='scott', linewidth=2.5, histtype='step', color='black')

# plt.xlabel("Stellar Inclination ($^{\\circ}$)")
# plt.ylabel("N")

# fig.savefig('/Users/mjfields/Documents/Research/Disk_Alignment/Analysis/StellarInclination/python_inclination.pdf')

# plt.close()




""" autocorr stuff """

# def next_pow_two(n):
#     i = 1
#     while i < n:
#         i = i << 1
#     return i


# def autocorr_func_1d(x, norm=True):
#     x = np.atleast_1d(x)
#     if len(x.shape) != 1:
#         raise ValueError("invalid dimensions for 1D autocorrelation function")
#     n = next_pow_two(len(x))

#     # Compute the FFT and then (from that) the auto-correlation function
#     f = np.fft.fft(x - np.mean(x), n=2 * n)
#     acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
#     acf /= 4 * n

#     # Optionally normalize
#     if norm:
#         acf /= acf[0]

#     return acf


# def auto_window(taus, c):
#     m = np.arange(len(taus)) < c * taus
#     if np.any(m):
#         return np.argmin(m)
#     return len(taus) - 1


# def autocorr_new(y, c=5.0):
#     f = np.zeros(y.shape[1])
#     for yy in y:
#         f += autocorr_func_1d(yy)
#     f /= len(y)
#     taus = 2.0 * np.cumsum(f) - 1.0
#     window = auto_window(taus, c)
#     return taus[window]

