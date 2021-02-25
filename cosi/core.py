# -*- coding: utf-8 -*-


import numpy as np
import emcee




class Probability(object):
    
    def __init__(self, vsini, e_vsini, rstar, e_rstar, prot, e_prot):
        
        self.vsini = vsini # km/s
        self.e_vsini = e_vsini
        
        self.rstar = rstar # R_Sun
        self.e_rstar = e_rstar
        
        self.prot = prot # days
        self.e_prot = e_prot
        
        
        
        
    def log_likelihood(self, theta):
        
        cosi, r, p = theta
        
        
        sini = np.sqrt(1 - cosi**2)
        
        # line-of-sight velocity; km/s
        cv = 2 * np.pi * self.rstar * 6.957e5 / (self.prot * 24 * 3600)
        
        cvsini = cv * sini
        
        if cvsini > self.vsini:
            return -np.inf
        
        
        chi2 = (self.vsini - cvsini)**2 / self.e_vsini**2 + (r - self.rstar)**2 / self.e_rstar**2 + (p - self.prot)**2 / self.e_prot**2
        
        return -0.5 * chi2
    
    
    
    
    def log_prior(self, theta):
        
        cosi, r, p = theta
        
        if cosi < 0 or cosi > 1:
            return -np.inf
        
        if r < 0 or p < 0:
            return -np.inf
        
        
        return 0.0
    
    
    
    
    def log_probability(self, theta):
        
        lp = self.log_prior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta)
    
    
    
    
class CosI(Probability):
    
    def __init__(self, vsini, e_vsini, rstar, e_rstar, prot, e_prot, nwalkers=100):
        
        super().__init__(vsini, e_vsini, rstar, e_rstar, prot, e_prot)
    
    
    
    
    def run_mcmc(self, nwalkers, nsteps, position=None, progress=True):
        
        ndim = 3
        
        p = position
        
        if p is None:
            p = [0.5, self.rstar, self.prot]
        
        pos = p + 0.02 * np.random.randn(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        
        sampler.run_mcmc(pos, nsteps, progress=progress)
        
        return sampler
    
    
    
    
    def get_posterior(self, sampler, burnin=500, thin=1):
        
        if len(self.sampler.get_chain()) < 1500:
            burnin = len(self.sampler.get_chain()) // 3
        
        chain = self.sampler.get_chain(flat=True, discard=burnin, thin=thin)
        
        return chain[:, 0]
    
    
    
    
    
