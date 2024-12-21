# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 21:37:43 2021

@author: wangbinb
"""

class water():
    def __init__(self, temperature = 22):
        self.temperature = temperature
        self.rho = self.density(temperature)
        self.nu = self.viscosity(temperature)[0]
        self.mu = self.viscosity(temperature)[1]
        self.u = 0.
        self.v = 0.
        self.w = 0.
        self.tke = 0.
        self.dtke = 0.
        
    def density(self, temperature):
        rho = 999.842594 + \
        6.793952e-2 * self.temperature - \
        9.905290e-3 * self.temperature ** 2 + \
        1.001685e-4 * self.temperature ** 3 - \
        1.120083e-6 * self.temperature ** 4 + \
        6.536332e-9 * self.temperature ** 5
        return rho

    def viscosity(self, temperature):
        mu = 2.414e-5 * 10 ** (247.8 / (self.temperature+273.15-140))
        nu = mu/ self.rho
        return nu, mu
    
    
class artificial_water():   
    def __init__(self, temperature = 22, density = 1000.):
        self.temperature = temperature
        self.name = 'defined_water'
        self.rho = density
        self.nu = self.viscosity(temperature)[0]
        self.mu = self.viscosity(temperature)[1]
        
    def viscosity(self, temperature):
        mu = 2.414e-5 * 10 ** (247.8 / (self.temperature+273.15-140))
        nu = mu/ self.rho
        return nu, mu