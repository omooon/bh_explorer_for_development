#!/usr/bin/env python3
from importlib.metadata import distribution
import numpy as np
from datetime import datetime, timedelta
from array import array
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from pbh_explorer.objects.blackhole.BlackHawk import BlackHawk

from astropy import constants as c
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time, TimeDelta
from astropy.table import QTable

from gammapy.modeling.models import (
    PointSpatialModel,
    TemplateSpectralModel,
    CompoundSpectralModel,
    SkyModel
)

from logging import getLogger, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
loglevel="INFO"
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)

class AstroObjectTracker:
    def __init__(
        self,
        epoch="2000-01-01 00:00:00",
        position={"ra": 0 * u.deg, "dec": 0 * u.deg, "distance": 1 * u.pc},
        velocity={"pm_ra_cosdec": 0 * u.mas / u.yr, "pm_dec": 0 * u.mas / u.yr, "radial_velocity": 0 * u.km / u.s},
        loglevel=None
    ):
        """
        Most general class of astronomical objects. This class holds a time-profile of each physical parameter.
    
        Although a galactic coordinate system can be given, all functions are calculated on the basis of an equatorial coordinate system.

        Parameters:
        - position (dict): Initial position (RA/Dec/Distance).
        - velocity (dict): Initial velocity (Proper Motion/Radial Velocity).
        - initial_epoch
        - final_epoch
        """
        if loglevel:
            logger.setLevel(loglevel)
            for handler in logger.handlers:
                handler.setLevel(loglevel)
            
        if "ra" in position and "dec" in position:
            icrs_sky_coord = SkyCoord(
                frame="icrs",
                obstime=Time(epoch),
                ra=position.get("ra"),
                dec=position.get("dec"),
                distance=position.get("distance"),
                pm_ra_cosdec=velocity.get("pm_ra_cosdec"),
                pm_dec=velocity.get("pm_dec"),
                radial_velocity=velocity.get("radial_velocity")
            )
            galactic_sky_coord = icrs_sky_coord.galactic
        elif "l" in position and "b" in position:
            galactic_sky_coord = SkyCoord(
                frame="galactic",
                obstime=Time(epoch),
                ra=position.get("l"),
                dec=position.get("b"),
                distance=position.get("distance"),
                pm_ra_cosdec=velocity.get("pm_l_cosb"),
                pm_dec=velocity.get("pm_b"),
                radial_velocity=velocity.get("radial_velocity")
            )
            icrs_sky_coord = galactic_sky_coord.icrs
        else:
            raise ValueError("Position dictionary must contain either ra/dec or l/b.")

        self.epoch_state = {
            "date": icrs_sky_coord.obstime.iso,
            "ra": icrs_sky_coord.ra,
            "dec": icrs_sky_coord.dec,
            "l": icrs_sky_coord.galactic.l,
            "b": icrs_sky_coord.galactic.b,
            "distance": icrs_sky_coord.distance,
            "pm_ra_cosdec": icrs_sky_coord.pm_ra_cosdec,
            "pm_dec": icrs_sky_coord.pm_dec,
            "pm_l_cosb": icrs_sky_coord.galactic.pm_l_cosb,
            "pm_b": icrs_sky_coord.galactic.pm_b,
            "radial_velocity": icrs_sky_coord.radial_velocity,
        }
        self.skycoord = icrs_sky_coord
        
    @classmethod
    def arrangement_of_random_epoch_distributions(
        cls,
        epoch_range=("2000-01-01", "2000-01-01"),
        position_range={"ra": (0, 360)*u.deg, "dec": (-90, 90)*u.deg, "distance": (0.01, 1)*u.parsec},
        velocity_range={"pm_ra_cosdec": (0, 0)*u.mas/u.yr, "pm_dec": (0, 0)*u.mas/u.yr, "radial_velocity": (0, 0)*u.km/u.s},
        loglevel=None
    ):
        '''
        Generate an AstroObject instance with random parameters, including epoch.
        
        Parameters:
        - position_range (dict): Range for position parameters.
        - velocity_range (dict): Range for velocity parameters.
        -initial_epoch_range
        -final_epoch_range
    
        Returns:
        - AstroObject: A randomly generated AstroObject instance.
        '''
        epoch_range = Time(epoch_range)
        random_epoch = epoch_range[0] + np.random.uniform(0, (epoch_range[1] - epoch_range[0]).jd) * u.day
        
        position = {
            "ra": np.random.uniform(position_range["ra"][0].value, position_range["ra"][1].value) * position_range["ra"].unit,
            "dec": np.random.uniform(position_range["dec"][0].value, position_range["dec"][1].value) * position_range["dec"].unit,
            "distance": np.random.uniform(position_range["distance"][0].value, position_range["distance"][1].value) * position_range["distance"].unit,
        }
        
        velocity = {
            "pm_ra_cosdec": np.random.uniform(velocity_range["pm_ra_cosdec"][0].value, velocity_range["pm_ra_cosdec"][1].value) * velocity_range["pm_ra_cosdec"].unit,
            "pm_dec": np.random.uniform(velocity_range["pm_dec"][0].value, velocity_range["pm_dec"][1].value) * velocity_range["pm_dec"].unit,
            "radial_velocity": np.random.uniform(velocity_range["radial_velocity"][0].value, velocity_range["radial_velocity"][1].value) * velocity_range["radial_velocity"].unit,
        }
        return cls(
            epoch=random_epoch,
            position=position,
            velocity=velocity,
            loglevel=loglevel
        )
    
    def track_states(self, observation_dates):
        """
        Tracks the state of the object at each specified observation time.

        Parameters:
        - observation_dates (list of str): List of observation times in ISO format.

        Returns:
        - List of dictionaries, each containing the state at a specific time.
        """
        state = {
            "time": [],
            "ra": [],
            "dec": [],
            "distance": [],
            "pm_ra_cosdec": [],
            "pm_dec": [],
            "radial_velocity": [],
            "l": [],
            "b": [],
            "pm_l_cosb": [],
            "pm_b": [],
        }
        for obs_time in observation_dates:
            time = Time(obs_time)
            # Propagate the position to the observation time
            new_coord = self.skycoord.apply_space_motion(new_obstime=time)

            # Store the state at this time
            state["time"].append(time.iso)
            state["ra"].append(new_coord.ra)
            state["dec"].append(new_coord.dec)
            state["distance"].append(new_coord.distance)
            state["pm_ra_cosdec"].append(new_coord.pm_ra_cosdec)
            state["pm_dec"].append(new_coord.pm_dec)
            state["radial_velocity"].append(new_coord.radial_velocity)
            state["l"].append(new_coord.galactic.l)
            state["b"].append(new_coord.galactic.b)
            state["pm_l_cosb"].append(new_coord.galactic.pm_l_cosb)
            state["pm_b"].append(new_coord.galactic.pm_b)
        return state

class BlackHole(AstroObjectTracker):
    def __init__(
        self,
        blackhole={"mass": 1e+15*u.kg, "spin": 0*u.dimensionless_unscaled, "charge": 0*u.dimensionless_unscaled},
        **kwargs
    ):
        '''Class representing a Black Hole, derived from AstroObject.'''
        super().__init__(**kwargs)
        
        self.epoch_state["mass"] = blackhole.get("mass")
        self.epoch_state["spin"] = blackhole.get("spin")
        self.epoch_state["charge"] = blackhole.get("charge")
        self.epoch_state["radius"] = self.calculate_radius(
            blackhole.get("mass"),
            blackhole.get("spin"),
            blackhole.get("charge")
        )

    def calculate_radius(self, mass, spin, charge):
        """Calculate various radii of the black hole based on its type."""
        # ブラックホールの初期パラメータによってbh_typeを決める
        if spin == 0 * u.dimensionless_unscaled \
                and charge == 0 * u.dimensionless_unscaled:
            bh_type = "schwarzschild"
        else:
            logger.error('now not using')
        
        if bh_type == "schwarzschild":
            radius = (2 * c.G * mass / c.c**2).to(u.m)
        elif bh_type == "ergosphere":
            spin = self.state_history["spin"][-1]
            radius = (2 * c.G * mass / c.c**2 * (1 + spin)).to(u.m)
        elif bh_type == "photon_sphere":
            radius = (3 * c.G * mass / c.c**2).to(u.m)
        else:
            raise ValueError(f"Unknown black hole type: {bh_type}")
        return radius
            
    def step(self, delta_time, ebins):
        '''Returns a dictionary of ParticleDistribution for the energy range.
        The mass of the BlackHole is decreased by the total energy radiated.'''
        radiated_distributions = super().step(delta_time=delta_time, ebins=ebins)
        logger.debug(radiated_distributions)
        energy_transfer = 0
        for distribution in radiated_distributions.values():
            energy_transfer += distribution.total_energy()
        delta_mass = (energy_transfer / c.c**2).to(u.kg)
        logger.debug(f'Mass of the Blackhole decrfeases by: {delta_mass:1.1E}')
        self.mass -= delta_mass
        return radiated_distributions
    #internal_processes = {"AccretionProcess":(rate=1*u.M_sun/u.year), "JetFormation":(power=1e38*u.W)}

class DarkMatter(AstroObjectTracker):
    def __init__(
        self,
        darkmater={"velocity_dispersion": 270*u.km/u.s},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.epoch_state["velocity_dispersion"] = darkmater.get("velocity_dispersion")

class PrimordialBlackHole:
    hawking_calculated_flag = False
    _hawking_radiation_instance = None
    #hawking_calculated_params = {"mass": 1e+18*u.kg, "spin": 0, "charge": 0}

    def __init__(
        self,
        final_epoch="2000-01-01 00:00:00",
        internal_processes={"HawkingRadiation": BlackHawk},
        loglevel=None,
        **kwargs
    ):
        """Primordial Black Hole, representing an astronomical object, black hole, and dark matter entity."""
        self.darkmatter = DarkMatter(epoch=final_epoch, loglevel=loglevel, **kwargs)
        
        self.blackhole = BlackHole(
            blackhole={
                "mass": 0*u.kg,
                "spin": 0*u.dimensionless_unscaled,
                "charge": 0*u.dimensionless_unscaled
            },
            epoch=final_epoch,
            loglevel=loglevel,
            **kwargs
        )

        # parameter fo primordial black hole
        self.epoch_state = {
            "date": Time(final_epoch).iso,
            "ra": self.darkmatter.epoch_state["ra"],
            "dec": self.darkmatter.epoch_state["dec"],
            "l": self.darkmatter.epoch_state["l"],
            "b": self.darkmatter.epoch_state["b"],
            "distance": self.darkmatter.epoch_state["distance"],
            "pm_ra_cosdec": self.darkmatter.epoch_state["pm_ra_cosdec"],
            "pm_dec": self.darkmatter.epoch_state["pm_dec"],
            "pm_l_cosb": self.darkmatter.epoch_state["pm_l_cosb"],
            "pm_b": self.darkmatter.epoch_state["pm_b"],
            "radial_velocity": self.darkmatter.epoch_state["radial_velocity"],
            "velocity_dispersion": self.darkmatter.epoch_state["velocity_dispersion"],
            "mass": self.blackhole.epoch_state["mass"],
            "spin": self.blackhole.epoch_state["spin"],
            "charge": self.blackhole.epoch_state["charge"]
        }
        self.epoch_state["temperature"] = self.calculate_temperature(self.epoch_state["mass"])
        self.epoch_state["lifetime"] = self.calculate_lifetime(self.epoch_state["temperature"])

        self.hawking = None
        self.run_hawking_radiation_process(internal_processes["HawkingRadiation"])

    def calculate_temperature(self, mass):
        '''This method calculates the temperature of the Blackhole.'''
        # Assuming c from scipy.constants for speed of light
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide", category=RuntimeWarning
            )
            temperature_Kelvin = ( (c.hbar * c.c**3) / (8 * np.pi * c.G * mass.to(u.kg) * c.k_B) ).to(u.K)
        #temperature_GeV = temperature_Kelvin * c.k_B
        #temperature_GeV = temperature_GeV.to(u.GeV)  # conversion from Kelvin to GeV
        logger.debug(f'Temperature of the Blackhole: {temperature_Kelvin:1.1E}')
        return temperature_Kelvin
        
    def calculate_lifetime(self, temperature):
        '''Ukwatta et al. 2016'''
        lifetime = (4.8 * 10**2 *pow((c.k_B * temperature.to(u.K)/u.TeV).to(u.Unit(1)), -3)) * u.s
        logger.debug(f'Lifetime of the Blackhole: {lifetime:1.1E}')
        return lifetime

    def run_hawking_radiation_process(self, process_method):
        """Calculate Hawking radiation if not already done."""
        if not PrimordialBlackHole.hawking_calculated_flag:
            if issubclass(process_method, BlackHawk):
                bhawk = process_method(
                    Path('/Users/omooon/pbh_explorer/blackhawk_v2.3'),
                    Path('/Users/omooon/pbh_explorer/pbh_explorer/objects/blackhole/blackhawk_params/parameters_PYTHIA_present_epoch.txt')
                )
                #bhawk.launch_tot()
                bhawk.read_results(
                    particles = ["photon_primary", "photon_secondary"]
                )
        
                # インスタンスを渡す
                PrimordialBlackHole._hawking_radiation_instance = bhawk
                self.hawking = bhawk
            else:
                #TODO: 他の方法があればここに書く
                pass

            # 計算後、フラッグをTrueにする
            PrimordialBlackHole.hawking_calculated_flag = True

        elif self.hawking_calculated_flag:
            logger.debug("Hawking radiation already calculated, skipping.")
            self.hawking = PrimordialBlackHole._hawking_radiation_instance

    @classmethod
    def generate_random_profile_pbh(
        cls,
        final_epoch_range=("2000-01-01", "2010-01-01"),
        internal_processes={"HawkingRadiation": BlackHawk},
        loglevel=None,
        **kwargs
    ):
        '''
        # Example usage
        # Generate a list of 5 random PrimordialBlackHole instances
        pb_holes = [PrimordialBlackHole.generate_random_instance() for _ in range(5)]

        # Create AstroObjects with the generated Black Holes as the source
        astro_objects = [AstroObject(source=pb_hole, coordinate_system="spherical", position={"RA": 10 * u.deg, "Dec": 10 * u.deg}) for pb_hole in pb_holes]
        '''
        random_tracker = AstroObjectTracker.arrangement_of_random_epoch_distributions(
            epoch_range=final_epoch_range,
            loglevel=loglevel,
            **kwargs
        )

        position = {
            "ra": random_tracker.epoch_state["ra"],
            "dec":  random_tracker.epoch_state["dec"],
            "distance": random_tracker.epoch_state["distance"],
        }
        
        velocity = {
            "pm_ra_cosdec": random_tracker.epoch_state["pm_ra_cosdec"],
            "pm_dec": random_tracker.epoch_state["pm_dec"],
            "radial_velocity": random_tracker.epoch_state["radial_velocity"]
        }
            
        return cls(
            final_epoch=random_tracker.epoch_state["date"],
            internal_processes=internal_processes,
            position=position,
            velocity=velocity,
            loglevel=loglevel
        )

    def track_states(self, observation_dates):
        TIME_OF_EVAPORATION = Time(self.epoch_state["date"])
        time_to_evaporation = [max((TIME_OF_EVAPORATION - Time(obs_time)).sec, 0) for obs_time in observation_dates] * u.s
        
        mass_to_evaporation = np.interp(
            time_to_evaporation,
            np.flip(self.hawking.time_to_evaporation),
            np.flip(self.hawking.mass_to_evaporation),
            left=0
        )
        spin_to_evaporation = np.interp(
            time_to_evaporation,
            np.flip(self.hawking.time_to_evaporation),
            np.flip(self.hawking.spin_to_evaporation),
            left=0
        )
        charge_to_evaporation = np.interp(
            time_to_evaporation,
            np.flip(self.hawking.time_to_evaporation),
            np.flip(self.hawking.charge_to_evaporation),
            left=0
        )
        
        temp_to_evaporation = u.Quantity([self.calculate_temperature(i) for i in mass_to_evaporation])
        
        dm_states = self.darkmatter.track_states(observation_dates)
        
        # parameter fo primordial black hole
        obs_states = {
            "date": Time(observation_dates).iso,
            "ra": u.Quantity(dm_states["ra"]),
            "dec": u.Quantity(dm_states["dec"]),
            "l": u.Quantity(dm_states["l"]),
            "b": u.Quantity(dm_states["b"]),
            "distance": u.Quantity(dm_states["distance"]),
            "pm_ra_cosdec": u.Quantity(dm_states["pm_ra_cosdec"]),
            "pm_dec": u.Quantity(dm_states["pm_dec"]),
            "pm_l_cosb": u.Quantity(dm_states["pm_l_cosb"]),
            "pm_b": u.Quantity(dm_states["pm_b"]),
            "radial_velocity": u.Quantity(dm_states["radial_velocity"]),
            #"velocity_dispersion": u.Quantity(dm_states["velocity_dispersion"]),
            "mass": mass_to_evaporation,
            "spin": spin_to_evaporation,
            "charge": charge_to_evaporation,
            "temperature": temp_to_evaporation,
            "lifetime": time_to_evaporation
        }
        # Logging the observation states with a detailed and formatted output
        logger.debug("Observation States:\n")
        for key, value in obs_states.items():
            logger.debug(f"{key.capitalize()}: {value}")
        return obs_states

    def get_observation_averaged_spectral_models(
        self,
        observation_dates,
        distance=1*u.pc,
        particle="photon",
        frame="galactic"
    ):
        obs_states = self.track_states(observation_dates)

        total_spectral_models = []
        for idx in range(len(obs_states["date"]) - 1):
            diff_primary_spectra_table = self.hawking.diff_spectra[f"{particle}_primary"]
            diff_secondary_spectra_table = self.hawking.diff_spectra[f"{particle}_secondary"]
            
            primary_energy_colnames = diff_primary_spectra_table.colnames[1:]
            secondary_energy_colnames = diff_secondary_spectra_table.colnames[1:]
            
            # エネルギーアレイの取得
            primary_energy_array = u.Quantity([u.Quantity(e) for e in primary_energy_colnames])
            secondary_energy_array = u.Quantity([u.Quantity(e) for e in secondary_energy_colnames])
            
            # 微分スペクトルアレイの取得
            diff_primary_spectra_array = u.Quantity([diff_primary_spectra_table[colname] for colname in primary_energy_colnames])
            diff_secondary_spectra_array = u.Quantity([diff_secondary_spectra_table[colname] for colname in secondary_energy_colnames])
            
            # 積分スペクトルアレイの取得、全てを昇順(ascending order)にしてから解析
            flip_diff_primary_spectra_array = diff_primary_spectra_array[:, ::-1]
            flip_diff_secondary_spectra_array = diff_secondary_spectra_array[:, ::-1]
            flip_time_to_evaporation = np.flip(self.hawking.time_to_evaporation)
            flip_delta_time = np.flip(self.hawking.delta_time)
            
            # Extract time data
            lifetime_at_obs_start = obs_states["lifetime"][idx]
            lifetime_at_obs_stop = obs_states["lifetime"][idx+1]
            flip_idx_min = np.searchsorted(flip_time_to_evaporation, lifetime_at_obs_stop, side="left")
            flip_idx_max = np.searchsorted(flip_time_to_evaporation, lifetime_at_obs_start, side="left")

            # Integrate the differential spectra over time
            if flip_idx_min < flip_idx_max:
                integ_time_array = u.Quantity([
                    (flip_time_to_evaporation[flip_idx_min] - lifetime_at_obs_stop).to(u.s),
                    *(flip_delta_time[flip_idx_min+1:flip_idx_max]).to(u.s),
                    (lifetime_at_obs_start - flip_time_to_evaporation[flip_idx_max-1]).to(u.s)
                ])
                integrated_primary_spectrum_array = np.sum(
                    flip_diff_primary_spectra_array[:, flip_idx_min:flip_idx_max+1] * integ_time_array[np.newaxis, :],
                    axis=1
                )
                integrated_secondary_spectrum_array = np.sum(
                    flip_diff_secondary_spectra_array[:, flip_idx_min:flip_idx_max+1] * integ_time_array[np.newaxis, :],
                    axis=1
                )
            elif flip_idx_min == flip_idx_max:
                integ_time = lifetime_at_obs_start - lifetime_at_obs_stop
                integrated_primary_spectrum_array = flip_diff_primary_spectra_array[:,flip_idx_min] * integ_time
                integrated_secondary_spectrum_array = flip_diff_secondary_spectra_array[:,flip_idx_min] * integ_time
            
            TIME_OF_OBS_START = Time(obs_states["date"][idx])
            TIME_OF_OBS_STOP = Time(obs_states["date"][idx+1])
            obs_duration = (TIME_OF_OBS_STOP - TIME_OF_OBS_START).sec * u.s
            distance_factor = 4 * np.pi * distance.to(u.cm)**2
        
            primary_spectral_model = TemplateSpectralModel(
                energy = primary_energy_array,
                values = integrated_primary_spectrum_array / distance_factor / obs_duration * u.cm**3
            )
            secondary_spectral_model = TemplateSpectralModel(
                energy = secondary_energy_array,
                values = integrated_secondary_spectrum_array / distance_factor / obs_duration * u.cm**3
            )
            total_spectral_model = CompoundSpectralModel(
                primary_spectral_model,
                secondary_spectral_model,
                operator=lambda x, y: x + y
            )
            total_spectral_models.append(total_spectral_model)
        return total_spectral_models, obs_states

    def get_observation_averaged_skymodels(
        self,
        observation_dates,
        distance=1*u.pc,
        particle="photon",
        frame="galactic"
    ):
        # Create spectral models
        obs_averaged_spectral_models, obs_states = self.get_observation_averaged_spectral_models(
            observation_dates,
            distance=distance,
            particle=particle,
            frame=frame
        )
        
        # Create skymodels
        obs_averaged_skymodels = []
        for spectral_model in obs_averaged_spectral_models:
            # Create spatial models
            if frame == "galactic":
                spatial_model = PointSpatialModel(
                    lon_0=self.epoch_state["l"],
                    lat_0=self.epoch_state["b"],
                    frame="galactic"
                )
            elif frame == "icrs":
                spatial_model = PointSpatialModel(
                    lon_0=self.epoch_state["ra"],
                    lat_0=self.epoch_state["dec"],
                    frame="icrs"
                )
            
            obs_averaged_skymodels.append(
                SkyModel(
                    spectral_model=spectral_model,
                    spatial_model=spatial_model
                )
            )
        return obs_averaged_skymodels, obs_states

    def get_evolution_skymodels(
        self,
        distance=1*u.pc,
        particle="photon",
        frame="galactic",
    ):
        """
        Generate SkyModel objects from PBH data, considering observation start time.
        
        Parameters:
        - observation_start_time (str): ISO format string representing the start of observation.
        """
        # primary成分とsecondary成分のエネルギーとフラックスを抽出
        component_energy_array = {}
        component_spectra_array = {}
        for component in ["primary","secondary"]:
            spectra_table = self.hawking.integ_spectra[f"{particle}_{component}"]
            
            energy_bins = np.array([
                u.Quantity(colname).to(u.GeV).value for colname in spectra_table.colnames[1:]
            ]) * u.GeV
            component_energy_array[component] = energy_bins
            
            spectra_list = []
            for colname in spectra_table.colnames[1:]:
                # Convert from 1/(GeV cm3) to count number/GeV
                #TODO:合ってるか確認
                spectra_list.append(
                    (spectra_table[colname] * u.cm**3).to(u.GeV**(-1)).value
                )
            component_spectra_array[component] = np.array(spectra_list).T * u.GeV**(-1)
        
        # creationg short-term signal skymodel for PBH
        evol_skymodels = []
        for idx, (primary_spectrum, secondary_spectrum) in \
            enumerate(zip(component_spectra_array["primary"], component_spectra_array["secondary"])):
            
            distance_factor = 4 * np.pi * distance.to(u.cm)**-2
            primary_spectral_model = TemplateSpectralModel(
                energy = component_energy_array["primary"].to(u.GeV),
                values = primary_spectrum.to(1/u.GeV) * distance_factor * self.hawking.delta_time[idx]**-1
            )
            secondary_spectral_model = TemplateSpectralModel(
                energy = component_energy_array["secondary"].to(u.GeV),
                values = secondary_spectrum.to(1/u.GeV) * distance_factor * self.hawking.delta_time[idx]**-1
            )
            total_spectral_model = CompoundSpectralModel(
                primary_spectral_model,
                secondary_spectral_model,
                operator=lambda x, y: x + y
            )
            
            if frame == "galactic":
                spatial_model = PointSpatialModel(
                    lon_0=self.epoch_state["l"],
                    lat_0=self.epoch_state["b"],
                    frame="galactic"
                )
            elif frame == "icrs":
                spatial_model = PointSpatialModel(
                    lon_0=self.epoch_state["ra"],
                    lat_0=self.epoch_state["dec"],
                    frame="icrs"
                )
            
            evol_skymodels.append(
                SkyModel(
                    spectral_model=total_spectral_model,
                    spatial_model=spatial_model,
                    name=f"lifetime{self.hawking.time_to_evaporation[idx].to(u.s).value}s_distance{distance.to(u.pc).value}pc"
                )
            )
        return evol_skymodels






'''
for idx in range(len(obs_states["date"]) - 1):
# Extract time and spectrum data
lifetime_at_obs_start = obs_states["lifetime"][idx]
lifetime_at_obs_stop = obs_states["lifetime"][idx+1]

# Create spectral models
component_spectral_models = []
for component in ["primary", "secondary"]:
    diff_spectra_table = self.hawking.diff_spectra[f"{particle}_{component}"]
    energy_colnames = diff_spectra_table.colnames[1:]
    
    energy_array = u.Quantity(
        [u.Quantity(e) for e in energy_colnames]
    )
    diff_spectra_array = np.array(
        [diff_spectra_table[colname].value for colname in energy_colnames]
    )
    
    # 全てを昇順(ascending order)にしてから解析
    flip_diff_spectra_array = np.array(
        [diff_spectra_table[::-1][colname].value for colname in energy_colnames]
    )
    flip_time_to_evaporation = np.flip(self.hawking.time_to_evaporation)
    flip_delta_time = np.flip(self.hawking.delta_time)

    flip_idx_min = np.searchsorted(flip_time_to_evaporation, lifetime_at_obs_stop, side="left")
    flip_idx_max = np.searchsorted(flip_time_to_evaporation, lifetime_at_obs_start, side="left")
    
    # Integrate the differential spectra over time
    accumulated_time = 0 * u.s
    #if flip_idx_min < flip_idx_max:
    #   integ_time = np.diff(np.concatenate(([lifetime_at_obs_stop], flip_time_to_evaporation[flip_idx_min:flip_idx_max], [lifetime_at_obs_start])))
    #    integrated_spectrum_array = np.sum(flip_diff_spectra_array[:, flip_idx_min:flip_idx_max] * integ_time, axis=1)
    if flip_idx_min < flip_idx_max:
        for flip_idx in range(flip_idx_min, flip_idx_max+1):
            if flip_idx == flip_idx_min:
                integ_time = (flip_time_to_evaporation[flip_idx] - lifetime_at_obs_stop).to(u.s)
                integrated_spectrum_array = flip_diff_spectra_array[:,flip_idx] * integ_time
                
            elif flip_idx_min < flip_idx < flip_idx_max:
                integ_time = (flip_delta_time[flip_idx]).to(u.s)
                integrated_spectrum_array += flip_diff_spectra_array[:,flip_idx] * integ_time
                
            elif flip_idx == flip_idx_max:
                integ_time = (lifetime_at_obs_start - flip_time_to_evaporation[flip_idx-1]).to(u.s)
                integrated_spectrum_array += flip_diff_spectra_array[:,flip_idx] * integ_time
            
            accumulated_time += integ_time
    elif flip_idx_min == flip_idx_max:
        integ_time = lifetime_at_obs_start - lifetime_at_obs_stop
        integrated_spectrum_array = flip_diff_spectra_array[:,flip_idx_min] * integ_time
        accumulated_time += integ_time
    integrated_spectrum = integrated_spectrum_array * diff_spectra_table[energy_colnames[0]].unit * u.cm**3
    
    TIME_OF_OBS_START = Time(obs_states["date"][idx])
    TIME_OF_OBS_STOP = Time(obs_states["date"][idx+1])
    obs_duration = (TIME_OF_OBS_STOP - TIME_OF_OBS_START).sec * u.s
    logger.debug(f"Accumulated Time: {accumulated_time}, Observation Duration: {obs_duration}")

    distance_factor = 4 * np.pi * distance.to(u.cm)**2
    
    spectral_model = TemplateSpectralModel(
        energy = energy_array,
        values = integrated_spectrum / distance_factor / obs_duration
    )
    component_spectral_models.append(spectral_model)
    
from functools import reduce
total_spectral_model = reduce(
    lambda x, y: CompoundSpectralModel(x, y, operator=lambda a, b: a + b),
    component_spectral_models
)
'''
