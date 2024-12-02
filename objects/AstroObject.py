#!/usr/bin/env python3
from importlib.metadata import distribution
import numpy as np
from datetime import datetime, timedelta
from array import array
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from logging import getLogger, StreamHandler

# Logger
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)

from astropy import constants as c
from astropy import units as u
from astropy.coordinates import SkyCoord
from .Particle import Particle, ParticleDistribution

class AstroObject:
    '''Most general class of astronomical objects.
    This class holds a time-profile of each physical parameter.'''
    
    def __init__(self, proper_time=0*u.s, mass=None, radius=None, position={'RA': 0*u.deg, 'DEC': 0*u.deg, 'distance': 1*u.parsec}, velocity=(0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s), internal_processes=[]):
        '''Constructor for the AstroObject class.
        Args:
        - proper_time (float): The initial proper time of the AstroObject. Use astropy.units.
        - mass: The mass of the AstroObject. Use astropy.units.
        - radius: The radius of the AstroObject. Use astropy.units.
        - position (tuple): The initial position of the AstroObject (RA, DEC, luminosity distance). Use astropy.units.
        - velocity (tuple): The initial velocity of the AstroObject (vx, vy, vz). Use astropy.units.
        - internal_processes (list): List of internal processes of the AstroObject.
        '''
        self.proper_time = proper_time
        self.mass = mass
        self.radius = radius
        self.position = position
        self.velocity = velocity
        self.internal_processes = internal_processes

    def sky_coord(self):
        return SkyCoord(self.position["RA"], self.position["DEC"], frame="icrs")

    def step(self, delta_time, ebins):
        '''Returns a dictionary of PaticleDistribution for the energy range and the AstroObject.
        The AstroObject is not changed by this function.'''
        radiated_distributions = {} # Dictionary of Particle: ParticleDistribution
        for process in self.internal_processes:
            for particle in process.radiation_functions.keys():
                distribution = ParticleDistribution(
                        particle=particle, 
                        spectrum=np.histogram(
                            ebins[:-1],
                            bins=ebins,
                            density=False,
                            weights=process.count_dt(particle=particle, ebins=ebins, astro_object=self)*delta_time,
                        ),
                        position=self.position
                    )
                if not particle in radiated_distributions:
                    radiated_distributions[particle] = distribution
                else:
                    radiated_distributions[particle].spectrum[0] += distribution.spectrum[0]
        self.proper_time += delta_time
        return radiated_distributions            


class Blackhole(AstroObject):
    '''Class representing a Black Hole, derived from AstroObject.'''
    
    def __init__(self, proper_time=0*u.s, mass=None, position={'RA': 0*u.deg, 'DEC': 0*u.deg, 'distance': 1*u.parsec}, velocity=(0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s), internal_processes=[]):
        '''Constructor for the BlackHole class.
        Args:
        - proper_time (float): The initial proper time of the Blackhole. Use astropy.units.
        - mass: The mass of the BlackHole. Use astropy.units.
        - position (tuple): The initial position of the Blackhole (RA, DEC, luminosity distance). Use astropy.units.
        - velocity (tuple): The initial velocity of the Blackhole (vx, vy, vz). Use astropy.units.
        - internal_processes (list): List of internal processes of the Blackhole.
        '''
        AstroObject.__init__(self, proper_time=proper_time, mass=mass, position=position, velocity=velocity, internal_processes=internal_processes)

    def temperature(self):
        '''This method calculates the temperature of the Blackhole.'''
        # Assuming c from scipy.constants for speed of light
        temp = ( c.hbar * c.c**3 / (8 * np.pi * c.G * self.mass * c.k_B) ).to(u.K)
        logger.debug(f'Temperature of the Blackhole: {temp:1.1E}')
        return temp

    def lifetime(self):
        '''Ukwatta et al. 2016'''
        lifetime = (4.8 * 10**2 *pow((c.k_B*self.temperature()/u.TeV).to(u.Unit(1)), -3)) * u.s
        logger.debug(f'Lifetime of the Blackhole: {lifetime:1.1E}')
        return lifetime

    def step(self, delta_time, ebins):
        '''Returns a dictionary of PaticleDistribution for the energy range. 
        The mass of the Blackhole is decreased by the total energy radiated.'''
        radiated_distributions = super().step(delta_time=delta_time, ebins=ebins)
        logger.debug(radiated_distributions)
        energy_transfer = 0
        for distribution in radiated_distributions.values():
            energy_transfer += distribution.total_energy()
        delta_mass = (energy_transfer / c.c**2).to(u.kg)
        logger.debug(f'Mass of the Blackhole decrfeases by: {delta_mass:1.1E}')
        self.mass -= delta_mass
        return radiated_distributions    
    
    def velocity(self):
        pass
