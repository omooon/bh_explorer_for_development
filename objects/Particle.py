#!/usr/bin/env python3
import numpy as np
from astropy import units as u

class Particle:
    def __init__(self, name, decay_functions, rest_mass, charge, flavour, spin, ndof):
        self.name = name
        self.rest_mass = rest_mass
        self.charge = charge
        self.flavour = flavour
        self.spin = spin
        self.ndof = ndof
        self.decay_functions = decay_functions

    def decay(self, product, energy, delta_time=0):
        '''Returns a tuple of (numpy.histogram, float).
        The histogram represents a PDF of the decay product energy
        for the given energy and the delta_time.
        The float stands for a probability the original particle decays.'''
        return self.decay_functions[product](energy, delta_time)

        
class ParticleDistribution:
    def __init__(self, particle, position, spectrum=None, name="", spec_hist=True):
        '''spectrum should be given by a numpy.histogram.'''
        self.particle = particle
        self.position = position
        self.spectrum = spectrum  # Should be a numpy.histogram
        self.name = name
        self.spec_hist = spec_hist  # True if spectrum is numpy.histogram; False if spectrum is a tuple of two numpy.array objects
        
        ParticleDistributionProfile.__init__(
            self=self, spectra=spectra_list, energy_bins=energy_bins
            )

    def total_count(self):
        return self.spectrum[0].sum()

    def total_energy(self):
        return (self.spectrum[0] * np.sqrt(self.spectrum[1][:-1]*self.spectrum[1][1:])).sum()
    
    def intrinsic_distribution(self, delta_time=0):
        '''Processes the decay process of the particle.
        The lifetime is limited to zero for simplification (to be improved).
        All counts of this object are set zero after this process.'''
        product_distributions = {}
        for product in self.particle.decay_functions.keys():
            product_distributions[product] = \
                ParticleDistribution(
                    particle=product,
                    spectrum=np.histogram([], bins=self.spectrum[1]),
                    name=' '.join([product.name, str(delta_time)])
                )
            for ibin, (count, e_lo, e_hi) in enumerate(zip(
                    self.spectrum[0], self.spectrum[1][:-1], self.spectrum[1][1:]
                    )):
                prod_spec, decay_prob = self.particle.decay(
                    product=product,
                    energy=np.sqrt(e_lo*e_hi),
                    delta_time=delta_time
                    )
                product_distributions[product].spectrum[0]\
                    += prod_spec[0] * count
                self.spectrum[0][ibin] -= count * decay_prob
                # Decrement the particle counts
        return product_distributions
    
    def observed_distribution(self):
        pass


class ParticleDistributionProfile:
    def __init__(self, spectra, energy_bins):
        self.spectra = spectra
        self.spectra_bins = energy_bins
