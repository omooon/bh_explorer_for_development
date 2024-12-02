#!/usr/bin/env python3
import numpy as np
from astropy import units as u


class ParticleDistribution:
    def __init__(self, particle, position, spectrum=None, name="", spec_hist=True):
        '''spectrum should be given by a numpy.histogram.'''
        self.particle = particle
        self.position = position
        self.spectrum = spectrum  # Should be a numpy.histogram
        self.name = name
        self.spec_hist = spec_hist  # True if spectrum is numpy.histogram; False if spectrum is a tuple of two numpy.array objects

    def total_count(self):
        return self.spectrum[0].sum()

    def total_energy(self):
        return (self.spectrum[0] * np.sqrt(self.spectrum[1][:-1]*self.spectrum[1][1:])).sum()

    def step(self, delta_time=0):
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


class ParticleDistributionProfile:
    def __init__(self, spectra, energy_bins):
        self.spectra = spectra
        self.spectra_bins = energy_bins


class ParticleDistributionProfileBlackHawk:
    def __init__(self, spec_table, blackhole, particle, name):
        energy_bins = np.array([
            u.Quantity(colname).to(u.GeV).value
            for colname in spec_table.colnames[1:]
            ]) * u.GeV
        spectra_list = []
        for colname in spec_table.colnames[1:]:
            spectra_list.append(
                (spec_table[colname] * u.cm**3).to(u.GeV**(-1)).value
                )  # Convert from 1/(GeV cm3) to count number/GeV
        spectra_array = np.array(spectra_list).T
        spectra_list = []
        for irow in reversed(range(len(spec_table))):
            spectrum_values = spectra_array[irow] * u.GeV**(-1)
            particle_graph = (spectrum_values, energy_bins)
            spectra_list.append(
                ParticleDistribution(
                    particle=particle,
                    position=blackhole.position,
                    spectrum=particle_graph,  # particle_hist,
                    name=name,
                    spec_hist=False,
                )
            )
            # profiles['{0}_spectrum'.format(particle)].append(particle_graph)
        ParticleDistributionProfile.__init__(
            self=self, spectra=spectra_list, energy_bins=energy_bins
            )
