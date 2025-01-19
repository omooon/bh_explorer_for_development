#!/usr/bin/env python3
from importlib.metadata import distribution
import numpy as np
from datetime import datetime, timedelta
from array import array
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from logging import getLogger, StreamHandler

from astropy import constants as const
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time, TimeDelta

from pathlib import Path
from pbh_explorer.objects import Particle
from pbh_explorer.objects.blackhole.BlackHawk import BlackHawk
from astropy.time import Time

from gammapy.modeling.models import (
    Models,
    PointSpatialModel,
    SkyModel,
    TemplateSpectralModel,
    CompoundSpectralModel
)

logger = getLogger(__name__)
handler = StreamHandler()
loglevel="INFO"
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)

class AstroObject:
    def __init__(
        self,
        initial_epoch="2000-01-01 00:00:00",
        final_epoch=None,
        position={"ra":0*u.deg, "dec":0*u.deg, "distance":1*u.parsec},
        velocity={"pm_ra_cosdec":0*u.mas/u.yr, "pm_dec":0*u.mas/u.yr, "radial_velocity":0*u.km/u.s},
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
        
        if initial_epoch and final_epoch \
                or not initial_epoch and not final_epoch:
            raise ValueError('you need to specify either epoch parameter')
            
        if "ra" in position and "dec" in position:
            icrs_sky_coord = SkyCoord(
                frame="icrs",
                obstime=Time(initial_epoch) if not final_epoch else Time(final_epoch),
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
                obstime=Time(initial_epoch) if not final_epoch else Time(final_epoch),
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
        
        # 与えられたepoch時点での天体の状態を記録
        if initial_epoch and not final_epoch:
            self.initial_state_sky_coords = {
                "icrs": icrs_sky_coord,
                "galactic": galactic_sky_coord
            }
            self.initial_state = {
                "time": icrs_sky_coord.obstime.iso,
                "ra": icrs_sky_coord.ra,
                "dec": icrs_sky_coord.dec,
                "l": icrs_sky_coord.galactic.l,
                "b": icrs_sky_coord.galactic.b,
                "distance": icrs_sky_coord.distance,
                "pm_ra_cosdec": icrs_sky_coord.pm_ra_cosdec,
                "pm_dec": icrs_sky_coord.pm_dec,
                "pm_l_cosb": icrs_sky_coord.galactic.pm_l_cosb,
                "pm_b": icrs_sky_coord.galactic.pm_b,
                "radial_velocity": icrs_sky_coord.radial_velocity
            }
            self.final_state_sky_coords = None
            self.final_state = None
            
        elif not initial_epoch and final_epoch:
            self.final_state_sky_coords = {
                "icrs": icrs_sky_coord,
                "galactic": galactic_sky_coord
            }
            self.final_state = {
                "time": icrs_sky_coord.obstime.iso,
                "ra": icrs_sky_coord.ra,
                "dec": icrs_sky_coord.dec,
                "l": icrs_sky_coord.galactic.l,
                "b": icrs_sky_coord.galactic.b,
                "distance": icrs_sky_coord.distance,
                "pm_ra_cosdec": icrs_sky_coord.pm_ra_cosdec,
                "pm_dec": icrs_sky_coord.pm_dec,
                "pm_l_cosb": icrs_sky_coord.galactic.pm_l_cosb,
                "pm_b": icrs_sky_coord.galactic.pm_b,
                "radial_velocity": icrs_sky_coord.radial_velocity
            }
            self.initial_state_sky_coords = None
            self.initial_state = None
            
        # 観測記録と同期させるための履歴用辞書
        self.obs_history = None

    def to_table(self):
        if self.initial_state is not None:
            print(f"Initial State Parameters:")
            print(
                QTable({key: [value] for key, value in self.initial_state.items()})
            )
            print("")
        
        if self.final_state is not None:
            print(f"Final State Parameters:")
            print(
                QTable({key: [value] for key, value in self.final_state.items()})
            )
            print("")
        
        print(f"Observation State History:")
        print(
            QTable(self.obs_history)
        )
        for key, value in self.obs_history.items():
            logger.debug(f"{key}: Length = {len(value)}")

    @classmethod
    def arrangement_of_random_epoch_distributions(
        cls,
        initial_epoch_range=("2000-01-01", "2010-01-01"),
        final_epoch_range=None,
        position_range={"ra": (0, 360)*u.deg, "dec": (-90, 90)*u.deg, "distance": (0.01, 1)*u.parsec},
        velocity_range={"pm_ra_cosdec": (0, 0)*u.mas/u.yr, "pm_dec": (0, 0)*u.mas/u.yr, "radial_velocity": (0, 0)*u.km/u.s},
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
        if initial_epoch_range and final_epoch_range or not initial_epoch_range and not final_epoch_range:
            raise ValueError('you need to specify either epoch parameter')
        elif initial_epoch_range and not final_epoch_range:
            epoch_range = Time(initial_epoch_range)
        elif not initial_epoch_range and final_epoch_range:
            epoch_range = Time(final_epoch_range)
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
            initial_epoch=random_epoch if initial_epoch_range else None,
            final_epoch=random_epoch if final_epoch_range else None,
            position=position,
            velocity=velocity
        )

    def update_obs_history(
        self,
        time_step_since_initial_state=None,
        time_step_since_final_state=None,
        **kwargs
    ):
        """
        時間の経過に従って天体の状態を更新し、state_historyに追加する関数。
        
        Parameters:
        - delta_time (float): 時間の増分（年単位）。
        - kwargs (dict): 更新したい追加のパラメータ（例: temperature, density など）。
        """
        # TODO: 新しく追加するやつの時間順序がおかしかったらエラーをはく
        if self.obs_history is None:
            self.obs_history = {key: [] for key in self.final_state.keys()}
            self.obs_history["interp_bounds"] = []
        
        
        # 新しい位置と速度を計算
        if time_step_since_initial_state is not None:
            time_step = time_step_since_initial_state
            evolved_icrs_coord = self.initial_state_sky_coords["icrs"].apply_space_motion(dt=time_step).copy()
        elif time_step_since_final_state is not None:
            time_step = time_step_since_final_state
            evolved_icrs_coord = self.final_state_sky_coords["icrs"].apply_space_motion(dt=time_step).copy()

    
        if time_step.value >= 0:
            # final_stateを入れる？
            # 時間ごとのプロフィールを更新
            self.obs_history["time"].append(evolved_icrs_coord.obstime.iso)
            self.obs_history["ra"].append(evolved_icrs_coord.ra)
            self.obs_history["dec"].append(evolved_icrs_coord.dec)
            self.obs_history["l"].append(evolved_icrs_coord.galactic.l)
            self.obs_history["b"].append(evolved_icrs_coord.galactic.b)
            self.obs_history["distance"].append(evolved_icrs_coord.distance)
            self.obs_history["pm_ra_cosdec"].append(evolved_icrs_coord.pm_ra_cosdec)
            self.obs_history["pm_dec"].append(evolved_icrs_coord.pm_dec)
            self.obs_history["pm_l_cosb"].append(evolved_icrs_coord.galactic.pm_l_cosb)
            self.obs_history["pm_b"].append(evolved_icrs_coord.galactic.pm_b)
            self.obs_history["radial_velocity"].append(evolved_icrs_coord.radial_velocity)

            # kwargsに基づいてstate_historyに新しいパラメータを追加
            for key, value in kwargs.items():
                self.obs_history[key].append(value)
                
        elif time_step.value < 0:
            # initial_stateを入れる？
            # 時間ごとのプロフィールを更新
            self.obs_history["time"].insert(0, evolved_icrs_coord.obstime.iso)
            self.obs_history["ra"].insert(0, evolved_icrs_coord.ra)
            self.obs_history["dec"].insert(0, evolved_icrs_coord.dec)
            self.obs_history["l"].insert(0, evolved_icrs_coord.galactic.l)
            self.obs_history["b"].insert(0, evolved_icrs_coord.galactic.b)
            self.obs_history["distance"].insert(0, evolved_icrs_coord.distance)
            self.obs_history["pm_ra_cosdec"].insert(0, evolved_icrs_coord.pm_ra_cosdec)
            self.obs_history["pm_dec"].insert(0, evolved_icrs_coord.pm_dec)
            self.obs_history["pm_l_cosb"].insert(0, evolved_icrs_coord.galactic.pm_l_cosb)
            self.obs_history["pm_b"].insert(0, evolved_icrs_coord.galactic.pm_b)
            self.obs_history["radial_velocity"].insert(0, evolved_icrs_coord.radial_velocity)
            
            # kwargsに基づいてstate_historyに新しいパラメータを追加
            for key, value in kwargs.items():
                self.obs_history[key].insert(0, value)



class BlackHole(AstroObject):
    #internal_processes = {"AccretionProcess":(rate=1*u.M_sun/u.year), "JetFormation":(power=1e38*u.W)}
    def __init__(
        self,
        blackhole={"mass": 1e+15*u.kg, "spin": 0*u.dimensionless_unscaled, "charge": 0*u.dimensionless_unscaled},
        internal_processes={},
        **kwargs
    ):
        '''Class representing a Black Hole, derived from AstroObject.'''
        super().__init__(**kwargs)
        
        # 初期状態もしくは最終状態にブラックホールの物理パラメータを追加
        if self.initial_state is not None:
            self.initial_state["mass"] = blackhole.get("mass")
            self.initial_state["spin"] = blackhole.get("spin")
            self.initial_state["charge"] = blackhole.get("charge")
            self.initial_state["radius"] = self.calculate_radius(
                                                      blackhole.get("mass"),
                                                      blackhole.get("spin"),
                                                      blackhole.get("charge")
                                                  )
            
        elif self.final_state is not None:
            self.final_state["mass"] = blackhole.get("mass")
            self.final_state["spin"] = blackhole.get("spin")
            self.final_state["charge"] = blackhole.get("charge")
            self.final_state["radius"] = self.calculate_radius(
                                                    blackhole.get("mass"),
                                                    blackhole.get("spin"),
                                                    blackhole.get("charge")
                                                )

        self.internal_processes = internal_processes

    def calculate_radius(self, mass, spin, charge):
        """Calculate various radii of the black hole based on its type."""
        
        # ブラックホールの初期パラメータによってbh_typeを決める
        if spin == 0 * u.dimensionless_unscaled \
                and charge == 0 * u.dimensionless_unscaled:
            bh_type = "schwarzschild"
        else:
            logger.error('now not using')
        
        if bh_type == "schwarzschild":
            radius = (2 * const.G * mass / const.c**2).to(u.m)
        elif bh_type == "ergosphere":
            spin = self.state_history["spin"][-1]
            radius = (2 * const.G * mass / const.c**2 * (1 + spin)).to(u.m)
        elif bh_type == "photon_sphere":
            radius = (3 * const.G * mass / const.c**2).to(u.m)
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
    def _run_accuration_process(self):
        for process in self.internal_processes:
            if isinstance(process, AccretionProcess):
                accreted_mass = process.evolve(delta_time)
                self._initial_params['mass'] += accreted_mass  # 質量を増加
            elif isinstance(process, JetFormation):
                radiated_energy = process.radiate_energy(delta_time)
                self._initial_params['mass'] -= radiated_energy / (c.c**2)  # 質量エネルギー保存則に基づいて減少
            elif isinstance(process, HawkingRadiation):
                emission_spectrum = process.calculate_emission(self._initial_params['mass'])
                # 放出されたエネルギーや質量を適用



class DarkMatter(AstroObject):
    def __init__(
        self,
        darkmater={"velocity_dispersion": 270*u.km/u.s},
        internal_processes={},
        **kwargs
    ):
        # `AstroObject` の親クラスを呼び出して基本的な初期化を行う
        super().__init__(**kwargs)
        
        if self.initial_state is not None:
            self.initial_state["velocity_dispersion"] = darkmater.get("velocity_dispersion")
        elif self.final_state is not None:
            self.final_state["velocity_dispersion"] = darkmater.get("velocity_dispersion")

        self.internal_processes = internal_processes



class PrimordialBlackHole(AstroObject):
    hawking_calculated_flag = False
    _hawking_radiation_instance = None

    def __init__(self, final_epoch="2000-01-01 00:00:00", internal_processes=None, **kwargs):
        """Primordial Black Hole, representing an astronomical object, black hole, and dark matter entity."""
        internal_processes = internal_processes or {"HawkingRadiation": BlackHawk}
        pbh_final_state = {
            "mass": 0*u.kg,
            "spin": 0*u.dimensionless_unscaled,
            "charge": 0*u.dimensionless_unscaled,
            "velocity_dispersion": 270*u.km/u.s
        }
        
        bh_instance = BlackHole(
            initial_epoch=None,
            final_epoch=final_epoch,
            blackhole={"mass": pbh_final_state["mass"], "spin": pbh_final_state["spin"], "charge": pbh_final_state["charge"]},
            internal_processes=internal_processes,
            **kwargs
        )
        dm_instance = DarkMatter(
            initial_epoch=None,
            final_epoch=final_epoch,
            darkmater={"velocity_dispersion": pbh_final_state["velocity_dispersion"]},
            internal_processes=internal_processes,
            **kwargs
        )
        internal_processes["BlackHole"] = bh_instance
        internal_processes["DarkMatter"] = dm_instance
        
        # `AstroObject` の親クラスを呼び出して基本的な初期化を行う
        logger=None
        super().__init__(
            initial_epoch=None,
            final_epoch=final_epoch,
            **kwargs
        )
        
        if self.initial_state is not None:
            raise ValueError("")
        if self.final_state is not None:
            # 速度の情報はdmを元に書き換える
            self.final_state["pm_ra_cosdec"] = dm_instance.final_state["pm_ra_cosdec"]
            self.final_state["pm_dec"] = dm_instance.final_state["pm_dec"]
            self.final_state["pm_l_cosb"] = dm_instance.final_state["pm_l_cosb"]
            self.final_state["pm_b"] = dm_instance.final_state["pm_b"]
            self.final_state["radial_velocity"] = dm_instance.final_state["radial_velocity"]
            self.final_state["velocity_dispersion"] = dm_instance.final_state["velocity_dispersion"]
            # ブラックホールと蒸発に伴うpbh固有の物理パラメータを追加
            self.final_state["mass"] = bh_instance.final_state["mass"]
            self.final_state["spin"] = bh_instance.final_state["spin"]
            self.final_state["charge"] = bh_instance.final_state["charge"]
            self.final_state["radius"] = bh_instance.final_state["radius"]
            self.final_state["temperature"] = self.calculate_temperature(
                                                        self.final_state["mass"]
                                                    )
            self.final_state["lifetime"] = self.calculate_lifetime(
                                                    self.final_state["temperature"]
                                                )
        
        # 指定したモデルに沿って、ホーキング放射の計算を行う
        self.internal_processes = internal_processes
        self._run_hawking_radiation_process()

    def _run_hawking_radiation_process(self):
        """Calculate Hawking radiation if not already done."""
        hawking_calculated_params = {"mass": 1e+18*u.kg, "spin": 0, "charge": 0}
        if not PrimordialBlackHole.hawking_calculated_flag \
            and "HawkingRadiation" in self.internal_processes.keys():
                
                # hawking radiationの計算プロセス方法のメソッドを取得
                process_method = self.internal_processes["HawkingRadiation"]
    
                if issubclass(process_method, BlackHawk):
                    bhawk = process_method(Path('/Users/omooon/pbh_explorer/blackhawk_v2.3'), Path('/Users/omooon/pbh_explorer/pbh_explorer/objects/blackhole/blackhawk_params/parameters_PYTHIA_present_epoch.txt'))
                    #bhawk.launch_tot()
                    bhawk.read_results(particles = ['photon_primary', 'photon_secondary'])
                    # メソッドだった部分を実行したインスタンスに書き換える
                    PrimordialBlackHole._hawking_radiation_instance = bhawk
                    self.internal_processes["HawkingRadiation"] = bhawk
                    # 計算後、フラッグをTrueにする
                    PrimordialBlackHole.hawking_calculated_flag = True
                else:
                    pass

        elif self.hawking_calculated_flag:
            logger.info("Hawking radiation already calculated, skipping.")
            self.internal_processes["HawkingRadiation"] = PrimordialBlackHole._hawking_radiation_instance

    @classmethod
    def generate_random_pbh(
        cls,
        final_epoch_range=("2000-01-01", "2010-01-01"),
        position_range={"ra": (0, 360)*u.deg, "dec": (-90, 90)*u.deg, "distance": (0.01, 1)*u.parsec},
        velocity_range={"pm_ra_cosdec": (0, 0)*u.mas/u.yr, "pm_dec": (0, 0)*u.mas/u.yr, "radial_velocity": (0, 0)*u.km/u.s},
        internal_processes={"HawkingRadiation": BlackHawk},
        **kwargs
    ):
        '''
        # Example usage
        # Generate a list of 5 random PrimordialBlackHole instances
        pb_holes = [PrimordialBlackHole.generate_random_instance() for _ in range(5)]

        # Create AstroObjects with the generated Black Holes as the source
        astro_objects = [AstroObject(source=pb_hole, coordinate_system="spherical", position={"RA": 10 * u.deg, "Dec": 10 * u.deg}) for pb_hole in pb_holes]
        '''
        astro_obj = AstroObject.arrangement_of_random_epoch_distributions(
            initial_epoch_range=None,
            final_epoch_range=final_epoch_range,
            position_range=position_range,
            velocity_range=velocity_range
        )
        
        final_epoch=astro_obj.final_state_params["time"]
        position={
            "ra": astro_obj.final_state_params["ra"],
            "dec": astro_obj.final_state_params["dec"],
            "distance": astro_obj.final_state_params["distance"],
        }
        velocity={
            "pm_ra_cosdec":astro_obj.final_state_params["pm_ra_cosdec"],
            "pm_dec":astro_obj.final_state_params["pm_dec"],
            "radial_velocity": astro_obj.final_state_params["radial_velocity"]
        }
        
        return cls(
                    final_epoch=final_epoch,
                    internal_processes=internal_processes,
                    position=position,
                    velocity=velocity,
                    **kwargs
                )

    def calculate_temperature(self, mass):
        '''This method calculates the temperature of the Blackhole.'''
        # Assuming c from scipy.constants for speed of light
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide", category=RuntimeWarning
            )
            temperature_Kelvin = ( (const.hbar * const.c**3) / (8 * np.pi * const.G * mass.to(u.kg) * const.k_B) ).to(u.K)
        #temperature_GeV = temperature_Kelvin * const.k_B
        #temperature_GeV = temperature_GeV.to(u.GeV)  # conversion from Kelvin to GeV
        logger.debug(f'Temperature of the Blackhole: {temperature_Kelvin:1.1E}')
        return temperature_Kelvin
        
    def calculate_lifetime(self, temperature):
        '''Ukwatta et al. 2016'''
        lifetime = (4.8 * 10**2 *pow((const.k_B * temperature.to(u.K)/u.TeV).to(u.Unit(1)), -3)) * u.s
        logger.debug(f'Lifetime of the Blackhole: {lifetime:1.1E}')
        return lifetime
        
        
    def applying_observation(
        self,
        obs_range=("1999-01-01 00:00:00", "2001-01-01 00:00:00"),
        time_step=1*u.day
    ):
        """
        This function processes observational data (add self.obs_hisory) based on a specified observation range
        (denoted by |--|) and compares it with event epoch (denoted by *).

        Three scenarios are considered:
        
        1. Event (*, the epoch time) occurs 'within' the observation range:
           Observation Start (| left bar) <= Event (*) <= Observation Stop (right bar |)
           
             |---*--------------------|
                                                              
        
        2. Event occurs 'exactly at the edge' of the observation range:
           Event (*) coincides with the start or stop of the observation range.
    
             *-------------------|
             

        3. Event occurs 'outside' the observation range but within the extended timeline:
           Event (*) occurs before or after the observation range (|--|).
        
             |------------------------|   *

        Parameters:
        - obs_range (tuple): A tuple defining the observation start (S) and end (E) times.
        - time_interval (Quantity): The time interval for sampling observational data (default is 1 day).

        Returns:
            self.obs_history
        """
        # 適用する観測時間
        TIME_OF_OBS_START = Time(obs_range[0])
        TIME_OF_OBS_STOP = Time(obs_range[1])
    
        # pbhが蒸発した時刻
        TIME_OF_EVAPORATION = Time(self.final_state["time"])
        
        # 観測時間に対応するlifetimeの記録用リスト
        obs_elapsed_time_history = []
        pbh_lifetime_history = []
        
        TIME_OF_OBS_CURRENT = TIME_OF_OBS_START
        while TIME_OF_OBS_CURRENT < TIME_OF_OBS_STOP:
            # 観測経過時間
            elapsed_time = (TIME_OF_OBS_CURRENT - TIME_OF_OBS_START).sec
            obs_elapsed_time_history.append(elapsed_time)
        
            # lifetimeの計算（観測時刻と蒸発時刻との差）
            lifetime = (TIME_OF_EVAPORATION - TIME_OF_OBS_CURRENT).sec
            lifetime = max(lifetime, 0) # lifetimeが負なら0にする
            pbh_lifetime_history.append(lifetime)
            
            # 次のtime_stepへ進める
            TIME_OF_OBS_CURRENT += TimeDelta(time_step)

        # TIME_OF_OBS_STOP時の経過時間も追加
        elapsed_time = (TIME_OF_OBS_STOP - TIME_OF_OBS_START).sec
        obs_elapsed_time_history.append(elapsed_time)
        obs_elapsed_time_history = obs_elapsed_time_history * u.s
        
        # TIME_OF_OBS_STOP時のlifetimeも追加
        lifetime = (TIME_OF_EVAPORATION - TIME_OF_OBS_STOP).sec
        lifetime = max(lifetime, 0) # lifetimeが負なら0にする
        pbh_lifetime_history.append(lifetime)
        pbh_lifetime_history = pbh_lifetime_history * u.s
            
        logger.debug(obs_elapsed_time_history)
        logger.debug(pbh_lifetime_history)
        
        # internal_processでの計算結果をもとに、線形保管をしつつ、計算された時間ステップでのパラメータをself.obs_historyに反映 0の値を後ろに置くべきでは？
        ref_evol_params = self.internal_processes["HawkingRadiation"].evolution_params
        for elapsed_time, lifetime in zip(obs_elapsed_time_history, pbh_lifetime_history):
            dark_matter = self.internal_processes["DarkMatter"]
            black_hole = self.internal_processes["BlackHole"]
            
            kwargs = {"velocity_dispersion": dark_matter.final_state_params["velocity_dispersion"]}
            
            time_to_evaporation = ref_evol_params["time_to_evaporation"]
            mass_to_evaporation = ref_evol_params["mass_to_evaporation"]
            spin_to_evaporation = ref_evol_params["spin_to_evaporation"]
            charge_to_evaporation = ref_evol_params["charge_to_evaporation"]
            if lifetime.value > time_to_evaporation[-1].value:
                interp_idx_high = np.absolute(time_to_evaporation.value >= lifetime.value).argmin() - 1
                interp_idx_low = np.absolute(time_to_evaporation.value <= lifetime.value).argmax() - 1
                kwargs["mass"] = (mass_to_evaporation[interp_idx_high] + mass_to_evaporation[interp_idx_low]) / 2
                kwargs["spin"] = (spin_to_evaporation[interp_idx_high] + spin_to_evaporation[interp_idx_low]) / 2
                kwargs["charge"] = (charge_to_evaporation[interp_idx_high] + charge_to_evaporation[interp_idx_low]) / 2
                kwargs["radius"] = black_hole.calculate_radius(
                                        kwargs.get("mass"),
                                        kwargs.get("spin"),
                                        kwargs.get("charge")
                                    )
                kwargs["interp_bounds"] = (interp_idx_low, interp_idx_high)
            elif time_to_evaporation[-1].value >= lifetime.value > 0:
                interp_idx_high = time_to_evaporation.size
                interp_idx_low = time_to_evaporation.size
                kwargs["mass"] = (mass_to_evaporation[interp_idx_high] + mass_to_evaporation[interp_idx_low]) / 2
                kwargs["spin"] = (spin_to_evaporation[interp_idx_high] + spin_to_evaporation[interp_idx_low]) / 2
                kwargs["charge"] = (charge_to_evaporation[interp_idx_high] + charge_to_evaporation[interp_idx_low]) / 2
                kwargs["radius"] = black_hole.calculate_radius(
                                        kwargs.get("mass"),
                                        kwargs.get("spin"),
                                        kwargs.get("charge")
                                    )
                kwargs["interp_bounds"] = (interp_idx_low, interp_idx_high)
            elif lifetime.value == 0:
                kwargs["mass"] = 0 * mass_to_evaporation.unit
                kwargs["spin"] = 0 * spin_to_evaporation.unit
                kwargs["charge"] = 0 * charge_to_evaporation.unit
                kwargs["radius"] = black_hole.calculate_radius(
                                        kwargs.get("mass"),
                                        kwargs.get("spin"),
                                        kwargs.get("charge")
                                    )
                kwargs["interp_bounds"] = None

            kwargs["temperature"] = self.calculate_temperature(kwargs.get("mass"))
            kwargs["lifetime"] = lifetime

            # 1. 最終蒸発が観測範囲より前にある場合
            #if TIME_OF_EVAPORATION < TIME_OF_OBS_START:
            # 2. 最終蒸発が観測範囲内にある場合
            #elif TIME_OF_OBS_START <= TIME_OF_EVAPORATION <= TIME_OF_OBS_STOP:
            # 3. 最終蒸発が観測範囲より後にある場合
            #elif TIME_OF_OBS_STOP < TIME_OF_EVAPORATION:
            '''
            ref_evol_params = self.internal_processes["HawkingRadiation"].evolution_params
            time_to_evap = ref_evol_params["time_to_evaporation"].value

            for elapsed_time, lifetime in zip(obs_elapsed_time_history, pbh_lifetime_history):
                kwargs = {"velocity_dispersion": self._dm.final_state_params["velocity_dispersion"]}

                # lifetimeが0の場合
                if lifetime.value == 0:
                    kwargs.update({
                        "mass": 0 * ref_evol_params["mass_to_evaporation"].unit,
                        "spin": 0 * ref_evol_params["spin_to_evaporation"].unit,
                        "charge": 0 * ref_evol_params["charge_to_evaporation"].unit
                    })
                else:
                    # 範囲外の補間処理（大きい場合は最大値で補間）
                    if lifetime.value > time_to_evap[-1]:
                        interp_time = time_to_evap[-1]
                    else:
                        interp_time = lifetime.value

                    # 線形補間を使って mass, spin, charge を求める
                    kwargs["mass"] = np.interp(interp_time, time_to_evap, ref_evol_params["mass_to_evaporation"].value) * ref_evol_params["mass_to_evaporation"].unit
                    kwargs["spin"] = np.interp(interp_time, time_to_evap, ref_evol_params["spin_to_evaporation"].value) * ref_evol_params["spin_to_evaporation"].unit
                    kwargs["charge"] = np.interp(interp_time, time_to_evap, ref_evol_params["charge_to_evaporation"].value) * ref_evol_params["charge_to_evaporation"].unit

                # radius を計算
                kwargs["radius"] = self._bh.calculate_radius(
                    kwargs.get("mass"),
                    kwargs.get("spin"),
                    kwargs.get("charge")
                )
            '''
            self.update_obs_history(
                time_step_since_initial_state=None,
                time_step_since_final_state=elapsed_time + (TIME_OF_OBS_START - TIME_OF_EVAPORATION).sec * u.s,
                **kwargs
            )
        
        # 時間順にソートする
        from datetime import datetime
        # 1. 時間順にソートするためのインデックスを取得
        time_list = self.obs_history["time"]
        sorted_indices = sorted(range(len(time_list)), key=lambda i: datetime.fromisoformat(time_list[i]))
        # 2. time, ra, dec, l などの各リストを sorted_indices に基づいて並べ替え
        sorted_obs_history = {
            key: [self.obs_history[key][i] for i in sorted_indices] for key in self.obs_history
        }
        self.obs_history = sorted_obs_history

        
    def get_observation_averaged_model(
        self,
        obs_range=("1999-12-30 00:08:00", "2000-01-10 00:00:00"),
        coordsys="galactic",
        particle_list=["photon"],
        decay_functions_list=[{}],
        rest_mass_list=[0],
        charge_list=[0],
        flavour_list=[None],
        spin_list=[1],
        ndof_list=[2]
    ):
        """
        指定された initial_time から initial_time + time_window までの
        データ範囲のスペクトルリストを作成する

        Parameters:
        initial_lifetime : Quantity array
            開始する時間（観測の開始時刻、単位付き）
        time_window : Quantity
            観測する時間間隔（単位付き）

        Returns:
        filtered_spectra : Quantity array
            指定された範囲のデータ
        """
        TIME_OF_OBS_START = Time(obs_range[0])
        TIME_OF_OBS_STOP = Time(obs_range[1])
        obs_duration = (TIME_OF_OBS_STOP - TIME_OF_OBS_START).sec * u.s
        
        #観測開始と終了時点でのパラメータのみがを取得
        self.applying_observation(obs_range=obs_range, time_step=obs_duration)
        
        # ref
        ref_evol_params = self.internal_processes["HawkingRadiation"].evolution_params
        
        # pbh_initial_lifetimeの参照インデックスとその補正係数の設定
        pbh_lifetime_at_obs_start = self.obs_history["lifetime"][0]
        if pbh_lifetime_at_obs_start.value > ref_evol_params["time_to_evaporation"][-1].value:
            start_idx = np.absolute(ref_evol_params["time_to_evaporation"].value >= pbh_lifetime_at_obs_start.value).argmin() - 1
            # 最初のスペクトルデータにかけるファクター
            excess_time = ref_evol_params["time_to_evaporation"][start_idx].value - pbh_lifetime_at_obs_start.value
            time_duration = ref_evol_params["delta_time"][start_idx].value
            start_idx_time_factor = (time_duration - excess_time) / time_duration
        elif ref_evol_params["time_to_evaporation"][-1].value >= pbh_lifetime_at_obs_start.value > 0:
            start_idx = ref_evol_params["time_to_evaporation"].size
            # 最初のスペクトルデータにかけるファクター
            excess_time = ref_evol_params["time_to_evaporation"][start_idx].value - pbh_lifetime_at_obs_start.value
            time_duration = ref_evol_params["delta_time"][start_idx].value
            start_idx_time_factor = (time_duration - excess_time) / time_duration
        elif pbh_lifetime_at_obs_start.value == 0:
            # 最後の行のインデックスを指定するけど、0かけて無かったことにする
            start_idx = ref_evol_params["time_to_evaporation"].size
            start_idx_time_factor = 0

        # pbh_final_lifetimeの参照インデックス
        pbh_lifetime_at_obs_stop = self.obs_history["lifetime"][-1]
        if pbh_lifetime_at_obs_stop.value > ref_evol_params["time_to_evaporation"][-1].value:
            # pbh_lifetime_at_obs_stop.valueを超えた大きさのインデックス
            stop_idx = np.absolute(ref_evol_params["time_to_evaporation"].value > pbh_lifetime_at_obs_stop.value).argmin() - 1
            # 最初のスペクトルデータにかけるファクター
            excess_time = pbh_lifetime_at_obs_stop.value - ref_evol_params["time_to_evaporation"][stop_idx].value
            time_duration = ref_evol_params["delta_time"][stop_idx].value
            stop_idx_time_factor = (time_duration - excess_time) / time_duration
        elif ref_evol_params["time_to_evaporation"][-1].value >= pbh_lifetime_at_obs_stop.value > 0:
            stop_idx = ref_evol_params["time_to_evaporation"].size
            # 最初のスペクトルデータにかけるファクター
            excess_time = ref_evol_params["time_to_evaporation"][stop_idx].value - pbh_lifetime_at_obs_stop.value
            time_duration = ref_evol_params["delta_time"][stop_idx].value
            stop_idx_time_factor = (time_duration - excess_time) / time_duration
        elif pbh_lifetime_at_obs_stop.value == 0:
            stop_idx = ref_evol_params["time_to_evaporation"].size
            stop_idx_time_factor = 0
    
        # 指定したparticleごとにスペクトルモデルを作成 # 与えられた観測時間に対する平均のモデルを作成
        # 辞書にする
        signal_skymodels = []
        hawking = self.internal_processes["HawkingRadiation"]
        for i, particle in enumerate(particle_list):
            if "photon" not in particle.lower():
                logger.error("Only photons can be modeled now! %s is not a photon.", particle)
                raise ValueError()
    
            # primary成分とsecondary成分ごとにエネルギーとフラックスを抽出
            component_energy_array = {}
            component_spectra_array = {}
            for component in ["primary", "secondary"]:
                # 欲しいidxの範囲を取得
                spectra_table = hawking.integral_radiation_spectra[f"{particle}_{component}"][start_idx:stop_idx+1]
                
                # エネルギー
                energy_bins = np.array([
                    u.Quantity(colname).to(u.GeV).value for colname in spectra_table.colnames[1:]
                ]) * u.GeV
                component_energy_array[component] = energy_bins
                
                # スペクトル
                spectra_list = []
                for colname in spectra_table.colnames[1:]:
                    #TODO:合ってるか確認
                    # Convert from 1/(GeV cm3) to count number/GeV
                    # 最初の行（str_idx）の値にはtime_factorをかける
                    first_col = (spectra_table[colname][0] * start_idx_time_factor * u.cm**3).to(u.GeV**(-1)).value
                    medium_col = np.sum((spectra_table[colname][1:] * u.cm**3).to(u.GeV**(-1)).value)
                    last_col = np.sum((spectra_table[colname][1:] * stop_idx_time_factor * u.cm**3).to(u.GeV**(-1)).value)
        
                    spectra_list.append( first_col + medium_col + last_col )
                component_spectra_array[component] = np.array(spectra_list).T * u.GeV**(-1)

            # spectral_models
            distance_factor = 4 * np.pi * (self.obs_history["distance"][0].to(u.cm))**2
            primary_spectral_model = TemplateSpectralModel(
                energy = component_energy_array["primary"].to(u.MeV),
                values = component_spectra_array["primary"].to(1/u.MeV) / distance_factor / obs_duration
            )
            secondary_spectral_model = TemplateSpectralModel(
                energy = component_energy_array["secondary"].to(u.MeV),
                values = component_spectra_array["secondary"].to(1/u.MeV) / distance_factor / obs_duration
            )
            total_spectral_model = CompoundSpectralModel(
                primary_spectral_model,
                secondary_spectral_model,
                operator=lambda x, y: x + y
            )
            
            # spatial_models
            if coordsys == "icrs":
                lon_0 = self.obs_history["ra"][0]
                lat_0 = self.obs_history["dec"][0]
            elif coordsys == "galactic":
                lon_0 = self.obs_history["l"][0]
                lat_0 = self.obs_history["b"][0]
            else:
                raise ValueError("")
            spatial_model = PointSpatialModel(
                lon_0=lon_0,
                lat_0=lat_0,
                frame=coordsys
            )
            
            signal_skymodels.append(
                SkyModel(
                    spectral_model=total_spectral_model,
                    spatial_model=spatial_model,
                    name=f"{str(obs_duration)} averaged {particle} model (initial lifetime={str(pbh_lifetime_at_obs_start)})"
                )
            )
        return signal_skymodels

    def get_evolution_skymodels(self, particle="photon", distance=1*u.pc):
        """
        Generate SkyModel objects from PBH data, considering observation start time.
        
        Parameters:
        - observation_start_time (str): ISO format string representing the start of observation.
        """
        # 計算済みホーキング放射インスタンスの取得
        hawking = self.internal_processes["HawkingRadiation"]
        
        #bhawk_spectra = bhawk.time_reduce_spectra(
        #    min_duration=2*u.s,
        #    min_deltamass=0.05
        #)

        # primary成分とsecondary成分のエネルギーとフラックスを抽出
        component_energy_array = {}
        component_spectra_array = {}
        for component in ["primary","secondary"]:
            spectra_table = hawking.integral_radiation_spectra[f"{particle}_{component}"]
            
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
            
            primary_spectral_model = TemplateSpectralModel(
                energy = component_energy_array["primary"].to(u.GeV),
                values = primary_spectrum.to(1/u.GeV) * distance.to(u.cm)**-2 * u.s**-1
            )
            secondary_spectral_model = TemplateSpectralModel(
                energy = component_energy_array["secondary"].to(u.GeV),
                values = secondary_spectrum.to(1/u.GeV) * distance.to(u.cm)**-2 * u.s**-1
            )
            total_spectral_model = CompoundSpectralModel(
                primary_spectral_model,
                secondary_spectral_model,
                operator=lambda x, y: x + y
            )
            
            spatial_model = PointSpatialModel(
                lon_0="0 deg",
                lat_0="0 deg",
                frame="galactic"
            )
            
            evol_skymodels.append(
                SkyModel(
                    spectral_model=total_spectral_model,
                    spatial_model=spatial_model,
                    name=f"lifetime{hawking.evolution_params['time_to_evaporation'][idx].to(u.s).value}s_distance{distance.to(u.pc).value}pc"
                )
            )
        return evol_skymodels
