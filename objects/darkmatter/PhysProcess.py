#!/usr/bin/env python3
import numpy as np
import datetime
from array import array
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy import constants as c
from astropy import units as u


class PhysProcess:
    def __init__(self, name):
        self.name = name

class VirializationProcess(PhysProcess):
    def __init__(self, name, radiation_functions={}):
        '''Provide a python function of d^2N/dE/dt in a dictionary radiation_function for each particle, 
        where the 1st argument "e" and the 2nd "o" are the photon energy and the radiator object, respectivly.'''
        super().__init__(name=name)
        self.radiation_functions = radiation_functions

    def count_dEdt(self, particle, energy, astro_object):
        '''Returns energy-differential flux of the given particle. energy is the particle energy, which is multiplied by an energy unit defined by astropy.units. '''
        if particle in self.radiation_functions:
            return self.radiation_functions[particle](energy, astro_object).to(u.GeV**(-1)*u.s**(-1))
        else:
            return 0
        
    def energy_dEdt(self, particle, energy, astro_object):
        '''Returns energy-differential flux of the given particle. energy is the particle energy, which is multiplied by an energy unit defined by astropy.units. '''
        if particle in self.radiation_functions:
            return energy * self.radiation_functions[particle](energy, astro_object)
        else:
            return 0
        
    def count_dt(self, particle, ebins, astro_object):      
        '''Returns energy-integral flux of the given particle. ebins is the energy bins, which is given as a numpy array multiplied by an energy unit defined by astropy.units.'''  
        return ( self.count_dEdt(particle=particle, energy=ebins[:-1], astro_object=astro_object) + self.count_dEdt(particle=particle, energy=ebins[1:], astro_object=astro_object) ) / 2. * (ebins[1:]-ebins[:-1])
    
    def energy_dt(self, particle, ebins, astro_object):      
        '''Returns energy-integral flux of the given particle. ebins is the energy bins, which is given as a numpy array multiplied by an energy unit defined by astropy.units.'''  
        return ( self.energy_dEdt(particle=particle, energy=ebins[:-1], astro_object=astro_object) + self.energy_dEdt(particle=particle, energy=ebins[1:], astro_object=astro_object) ) / 2. * (ebins[1:]-ebins[:-1])
