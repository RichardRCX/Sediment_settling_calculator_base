# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:02:11 2021

@author: wangbinb
"""
import numpy as np
from scipy.optimize import fsolve
from scipy import interpolate
import ref_water
water = ref_water.water()

g = -9.81

class tracer():
    def __init__(self):
        # dummy numbers
        self.dia = 1e-2
        self.temperature = 20
        self.ws = 0


class sand():
    def __init__(self, water, dia, density = 2650.):
        self.name = 'sand'
        self.density = density
        self.dia = dia
        self.ws = self.settling_velocity(water)[0]
        self.Re = self.settling_velocity(water)[1]
        self.tau = self.settling_velocity(water)[2]


    def settling_velocity(self,water):
        def settling_equation(ws):
            Re = np.abs(ws*self.dia/water.nu)
            # Cd = 24/Re * (1+0.152*Re**0.5+0.0151*Re)
            Cd = ((32/Re)**(1/1.5) + 1)**1.5

            return np.abs(4. * g * self.dia * (water.rho - self.density)) - (3. * Cd * water.rho) * ws ** 2


        ws = fsolve(settling_equation,0.001)
        ws = ws * np.sign(water.rho - self.density)
        Re = np.abs(ws*self.dia/water.nu)
        # Cd = 24/Re * (1+0.152*Re**0.5+0.0151*Re)
        Cd = ((32/Re)**(1/1.5) + 1)**1.5


        # particle relaxation time
        tau = np.abs(4*self.density*self.dia/(3*water.rho*Cd*ws))
        # ws = np.ndarray.tolist(ws)
        # Re = np.ndarray.tolist(Re)
        # tau = np.ndarray.tolist(tau)
        return ws, Re, tau
