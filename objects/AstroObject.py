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
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.table import QTable

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

class AstroObject:
    '''Most general class of astronomical objects.
    This class holds a time-profile of each physical parameter.'''
    def __init__(
        self,
        epoch="J2000.0",
        position={"RA": 0*u.deg, "Dec": 0*u.deg, "distance": 1*u.parsec},
        velocity={"proper_motion_ra":0*u.deg/u.yr, "proper_motion_dec":0*u.deg/u.yr, "radial_velocity": 0*u.km/u.s},
    ):
        '''
        Initialize the AstroObject instance.

        Parameters:
        - source : Name or source identifier for the object.
            このパラメータた、objectの物理的な性質（mass,radius,luminosity,temperature）について保持している
        - current_time (str): Reference time for the given position and motion.
        - position (dict): Initial position (RA/Dec/Distance or x/y/z).
        - velocity (dict): Initial velocity (Proper Motion/Radial Velocity or vx/vy/vz).
        '''
        
        self.epoch = Time(epoch)
        self.ra = position.get("RA")
        self.dec = position.get("Dec")
        self.distance = position.get("distance")  # in parsecs or other units
        self.proper_motion_ra = velocity.get("proper_motion_ra")  # arcsec/yr
        self.proper_motion_dec = velocity.get("proper_motion_dec")  # arcsec/yr
        self.radial_velocity = velocity.get("radial_velocity")  # km/s
        
        # 状態履歴を記録する辞書、初期状態を保存しておく
        self.current_time = self.epoch
        self.state_history = QTable(self.get_current_state())

    def __str__(self):
        """
        Display the initial state of the AstroObject.
        """
        return (f"[Epoch: {self.epoch}] - Spherical Coordinates:\n"
                f"  RA: {self.ra}\n"
                f"  Dec: {self.dec}\n"
                f"  Distance: {self.distance} parsecs\n"
                f"  Proper Motion (RA): {self.proper_motion_ra} arcsec/yr\n"
                f"  Proper Motion (Dec): {self.proper_motion_dec} arcsec/yr\n"
                f"  Radial Velocity: {self.radial_velocity} km/s\n")
            
    @classmethod
    def generate_random_distribution(
        cls,
        epoch_range=("2000-01-01", "2010-01-01"),
        position_range={"RA": (0, 360), "Dec": (-90, 90), "distance": (0.01, 1)},
        velocity_range={"proper_motion_ra": (0, 0), "proper_motion_dec": (0, 0), "radial_velocity": (0, 0)},
    ):
        '''
        Generate an AstroObject instance with random parameters, including epoch.
        
        Parameters:
        - coordinate_system (str): Coordinate system to use ('spherical' or 'cartesian').
        - time_range (tuple): Range for epoch as start and end date strings (e.g., "2000-01-01").
        - position_range (dict): Range for position parameters.
        - velocity_range (dict): Range for velocity parameters.
        
        Returns:
        - AstroObject: A randomly generated AstroObject instance.
        '''
        # ランダムな時刻を生成
        start_epoch = Time(epoch_range[0])
        end_epoch = Time(epoch_range[1])
        random_epoch = start_epoch + np.random.uniform(0, (end_epoch - start_epoch).jd) * u.day

        position = {
            "RA": np.random.uniform(position_range["RA"][0], position_range["RA"][1]) * u.deg,
            "Dec": np.random.uniform(position_range["Dec"][0], position_range["Dec"][1]) * u.deg,
            "distance": np.random.uniform(position_range["distance"][0], position_range["distance"][1]) * u.parsec,
        }
        
        velocity = {
            "proper_motion_ra": np.random.uniform(velocity_range["proper_motion_ra"][0], velocity_range["proper_motion_ra"][1]) * u.deg/u.yr,
            "proper_motion_dec": np.random.uniform(velocity_range["proper_motion_dec"][0], velocity_range["proper_motion_dec"][1]) * u.deg/u.yr,
            "radial_velocity": np.random.uniform(velocity_range["radial_velocity"][0], velocity_range["radial_velocity"][1]) * u.km/u.s,
        }

        return cls(
            epoch=random_epoch.iso,  # ISOフォーマットで渡す
            position=position,
            velocity=velocity
        )
        
    def get_current_state(self):
        '''
        Get the current state of the AstroObject.
        Returns a dictionary representing the current position, velocity, and time.
        '''
        state = {
            "time": [self.current_time.iso],
        }
        state.update({
            "RA": [self.ra],
            "Dec": [self.dec],
            "Distance": [self.distance],
            "Proper Motion RA": [self.proper_motion_ra],
            "Proper Motion Dec": [self.proper_motion_dec],
            "Radial Velocity": [self.radial_velocity],
        })
        return state

    def update_state(self, time):
        '''
        Update the state of the object to a specific time.
        The method calculates the position and velocity at the given time.
        
        Parameters:
        - time (str or astropy.time.Time): The target time to update the state to.
        '''
        new_time = Time(time)
        delta_time = (new_time - self.current_time).to(u.yr).value  # 年単位の差分

        # Proper motionによるRA/Decの更新
        new_ra = self.ra + (self.proper_motion_ra * delta_time)
        new_dec = self.dec + (self.proper_motion_dec * delta_time)
        # 距離と視線速度による距離変化
        new_distance = self.distance + (self.radial_velocity * delta_time).to(u.parsec)

        # 更新
        self.ra = new_ra
        self.dec = new_dec
        self.distance = new_distance

        # 現在時刻の更新
        self.current_time = new_time

        # 状態履歴に記録
        self.state_history.append(self.get_current_state())

    def to_skycoord(self, frame="icrs"):
        """
        Create a SkyCoord object from RA, Dec, and distance, with a specified coordinate frame.
        
        Parameters:
        - ra (float): Right Ascension (RA) in degrees.
        - dec (float): Declination (Dec) in degrees.
        - distance (float): Distance to the object in parsecs.
        - frame (str): Coordinate frame ("icrs" or "galactic"). Defaults to "icrs".
        
        Returns:
        - SkyCoord: SkyCoord object in the specified frame.
        """
        # Create the SkyCoord object in the ICRS frame
        coord = SkyCoord(ra=self.ra, dec=self.dec, distance=self.distance, frame="icrs")
        
        # Convert to the desired frame if specified
        if frame == "galactic":
            coord = coord.transform_to("galactic")
        elif frame != "icrs":
            raise ValueError("Unsupported frame. Use 'icrs' or 'galactic'.")
        
        return coord

    def to_qtable(self):
        '''Convert parameters to a QTable.'''
        initial_qtable = QTable()
        for key, value in self._initial_params.items():
            if isinstance(value, dict):  # For position and velocity dictionaries
                for subkey, subvalue in value.items():
                    initial_qtable[f"{subkey}"] = [subvalue]
            else:
                initial_qtable[key] = [value]
        return initial_qtable

class BlackHole(AstroObject):
    '''Class representing a Black Hole, derived from AstroObject.'''
    #internal_processes = {"AccretionProcess":(rate=1*u.M_sun/u.year), "JetFormation":(power=1e38*u.W)}
    
    def __init__(self, mass=1e+15*u.kg, spin=0, charge=0, internal_processes={}, **kwargs):
        # `AstroObject` の親クラスを呼び出して基本的な初期化を行う
        super().__init__(**kwargs)
        
        self.mass = mass
        self.spin = spin
        self.charge = charge
        self.internal_processes = internal_processes

    def __str__(self):
        """
        Returns a string representation of the BlackHole instance,
        including parent class properties and BlackHole-specific properties.
        """
        parent_str = super().__str__()  # Get string from the parent class
        black_hole_info = (
            f"  Mass: {self.mass}\n"
            f"  Spin: {self.spin}\n"
            f"  Charge: {self.charge}\n"
        )
        internal =f"Internal Processes: {', '.join(str(proc) for proc in self.internal_processes) if self.internal_processes else 'None'}"
        return f"{parent_str}\nBlack Hole Specifics:\n{black_hole_info}\n{internal}"

    @classmethod
    def generate_random_bh(
        cls,
        epoch_range=("2000-01-01", "2010-01-01"),
        position_range={"RA": (0, 360), "Dec": (-90, 90), "distance": (0.01, 1)},
        velocity_range={"proper_motion_ra": (0, 0), "proper_motion_dec": (0, 0), "radial_velocity": (0, 0)},
        mass_range=(1e+12, 1e+30),
        spin_range=(0, 0),
        charge_range=(0, 0),
        internal_processes={}
    ):
        '''
        Generate a random BlackHole instance with random physical parameters, including mass, spin, charge, and epoch.
        
        Parameters:
        - coordinate_system (str): Coordinate system to use ('spherical' or 'cartesian').
        - epoch_range (tuple): Range for epoch as start and end date strings (e.g., "2000-01-01").
        - position_range (dict): Range for position parameters.
        - velocity_range (dict): Range for velocity parameters.
        - mass_range (tuple): Range for mass of the black hole.
        - spin_range (tuple): Range for spin parameter.
        - charge_range (tuple): Range for charge parameter.
        
        Returns:
        - BlackHole: A randomly generated BlackHole instance.
        '''
        # 既存のランダム分布生成を呼び出す
        astro_obj = super().generate_random_distribution(
            epoch_range=epoch_range,
            position_range=position_range,
            velocity_range=velocity_range
        )
        epoch=astro_obj.epoch.iso,
        position={"RA": astro_obj.ra, "Dec": astro_obj.dec, "distance": astro_obj.distance}
        velocity={"proper_motion_ra":astro_obj.proper_motion_ra, "proper_motion_dec":astro_obj.proper_motion_dec, "radial_velocity": astro_obj.radial_velocity}
        
        # ブラックホールの特異なパラメータ（質量、スピン、電荷）をランダムに生成
        mass = np.random.uniform(mass_range[0], mass_range[1]) * u.kg
        spin = np.random.uniform(spin_range[0], spin_range[1])
        charge = np.random.uniform(charge_range[0], charge_range[1])
        
        # `BlackHole` のインスタンスを生成して返す
        return cls(
            mass=mass,
            spin=spin,
            charge=charge,
            internal_processes=internal_processes,
            epoch=epoch,
            position=position,
            velocity=velocity,
        )
    
    def calculate_radius(self, bh_type="schwarzschild"):
        if bh_type == "schwarzschild":
            pass
        elif bh_type == "ergosphere":
            pass
        elif bh_type == "photon_sphere":
            pass

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
    def __init__(self, internal_processes={}, **kwargs):
        # `AstroObject` の親クラスを呼び出して基本的な初期化を行う
        super().__init__(**kwargs)
        
        self.internal_processes = internal_processes

    def __str__(self):
        parent_str = super().__str__()  # Get string from the parent class
        black_hole_info = (
        )
        internal =f"Internal Processes: {', '.join(str(proc) for proc in self.internal_processes) if self.internal_processes else 'None'}"
        return f"{parent_str}\nBlack Hole Specifics:\n{black_hole_info}\n{internal}"

class PrimordialBlackHole(BlackHole, DarkMatter):
    """Primordial Black Hole, which is an astronomical object, black hole, and dark matter entity."""
    # Class variables for tracking Hawking radiation calculation
    _hawking_calculated_flag = False
    _hawking_calculated_params = {"mass": 1e+18*u.kg, "spin": 0, "charge": 0}
    hawking_radiation = None

    def __init__(self, mass=1e+15*u.kg, spin=0, charge=0, internal_processes={"HawkingRadiation":BlackHawk}, **kwargs):
        # `BlackHole` と `DarkMatter` の親クラスを呼び出して基本的な初期化を行う
        BlackHole.__init__(self, mass=mass, spin=spin, charge=charge, internal_processes=internal_processes, **kwargs)
        DarkMatter.__init__(self, internal_processes=internal_processes, **kwargs)
        
        #時間を蒸発した時間として使用
        self.state_history.rename_column("time", "evaporated_time")
        
        #履歴に物理パラメータも追加
        self.state_history["mass"] = self.mass
        self.state_history["spin"] = self.spin
        self.state_history["charge"] = self.charge
        
        self._run_hawking_radiation_process()
        
    def __str__(self):
        parent_str = super().__str__()  # Get string from the parent class
        primordial_info = "Primordial Black Hole Specifics: (additional properties can be added here)"
        internal = f"Internal Processes: {', '.join(str(proc) for proc in self.internal_processes) if self.internal_processes else 'None'}"
        return f"{parent_str}\n{primordial_info}\n{internal}"
        
    @classmethod
    def generate_random_pbh(
        cls,
        epoch_range=("2000-01-01", "2010-01-01"),
        position_range={"RA": (0, 360), "Dec": (-90, 90), "distance": (0.01, 1)},
        velocity_range={"proper_motion_ra": (0, 0), "proper_motion_dec": (0, 0), "radial_velocity": (0, 0)},
        mass_range=(1e+12, 1e+30),
        spin_range=(0, 0),
        charge_range=(0, 0),
        internal_processes={"HawkingRadiation":BlackHawk}
    ):
        '''
        # Example usage
        # Generate a list of 5 random PrimordialBlackHole instances
        pb_holes = [PrimordialBlackHole.generate_random_instance() for _ in range(5)]

        # Create AstroObjects with the generated Black Holes as the source
        astro_objects = [AstroObject(source=pb_hole, coordinate_system="spherical", position={"RA": 10 * u.deg, "Dec": 10 * u.deg}) for pb_hole in pb_holes]
        '''
        astro_obj = AstroObject.generate_random_distribution(
            epoch_range=epoch_range,
            position_range=position_range,
            velocity_range=velocity_range
        )
        epoch=astro_obj.epoch.iso
        position={"RA": astro_obj.ra, "Dec": astro_obj.dec, "distance": astro_obj.distance}
        velocity={"proper_motion_ra":astro_obj.proper_motion_ra, "proper_motion_dec":astro_obj.proper_motion_dec, "radial_velocity": astro_obj.radial_velocity}
        
        # ブラックホールの特異なパラメータ（質量、スピン、電荷）をランダムに生成
        log_mass = np.random.uniform(np.log10(1e+12), np.log10(1e+30))  # 対数値を生成
        mass = 10 ** log_mass * u.kg  # 線形スケールに戻す
        spin = np.random.uniform(spin_range[0], spin_range[1])
        charge = np.random.uniform(charge_range[0], charge_range[1])
        
        # ダークマターもあれば

        return cls(mass=mass, spin=spin, charge=charge, internal_processes=internal_processes, epoch=epoch, position=position, velocity=velocity)

    def calculate_lifetime(self):
        '''Ukwatta et al. 2016'''
        self.lifetime = []
        lifetime = (4.8 * 10**2 *pow((c.k_B*self.temperature()/u.TeV).to(u.Unit(1)), -3)) * u.s
        logger.debug(f'Lifetime of the Blackhole: {lifetime:1.1E}')
        return lifetime

    def caluculate_temperature(self, MASS):
        '''This method calculates the temperature of the Blackhole.'''
        # Assuming c from scipy.constants for speed of light
        temp = []
        for mass in MASS:
            temp.append(( c.hbar * c.c**3 / (8 * np.pi * c.G * mass.to(u.kg) * c.k_B) ).to(u.K))
            logger.debug(f'Temperature of the Blackhole: {temp:1.1E}')
        return temp
    
    def calculate_entropy(self):
        pass

    def _run_hawking_radiation_process(self):
        """Calculate Hawking radiation if not already done."""
        if not PrimordialBlackHole._hawking_calculated_flag:

            if "HawkingRadiation" in self.internal_processes.keys():
                # HawkingRadiationキーが含まれている場合
                process_method = self.internal_processes["HawkingRadiation"]
                
                if issubclass(process_method, BlackHawk):
                    bhawk = process_method(Path('/Users/omooon/pbh_explorer/blackhawk_v2.3'), Path('/Users/omooon/pbh_explorer/pbh_explorer/objects/blackhole/blackhawk_params/parameters_PYTHIA_present_epoch.txt'))
                    #bhawk.launch_tot()
                    bhawk.read_results(particles = ['photon_primary', 'photon_secondary'])
                    # 実行したインスタンスを渡す
                    PrimordialBlackHole.hawking_radiation = bhawk
                else:
                    #他のモデルがあれば
                    pass
                
                # 計算後、フラッグをTrueにする
                PrimordialBlackHole._hawking_calculated_flag = True
                        
            else:
                # HawkingRadiationキーが含まれていない場合
                print("HawkingRadiation is not present in internal_processes.")
        else:
            print("Hawking radiation already calculated, skipping.")

    def get_time_window_average_models(
        self,
        distance=0.1*u.pc,
        tstart="2000-01-01",
        tstop="2010-01-01",
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
        pbh_initial_lifetime = Time(self.state_history["evaporated_time"][0]) - Time(tstart)
        
        if Time(tstart)+pbh_initial_lifetime < Time(tstart):
            pass
        else:
            str_idx = self.hawking_radiation._dts['rt'][
                        self.hawking_radiation._dts['rt'] >= pbh_initial_lifetime
                    ].argmin()
            
            if self.hawking_radiation._dts['rt'][str_idx] == pbh_initial_lifetime:
                time_factor = 0
            else:
                excess_time = self.hawking_radiation._dts['rt'][str_idx] - pbh_initial_lifetime
                time_step = self.hawking_radiation._dts['rt'][str_idx] - self.hawking_radiation._dts['rt'][str_idx+1]
                time_factor = (time_step - excess_time) / time_step
    
        # particleごとのスペクトルモデルの作成
        for i, particle in enumerate(particle_list):
            if "photon" not in particle.lower():
                logger.error("Only photons can be modeled now! %s is not a photon.", particle)
        
            # primary成分とsecondary成分ごとにエネルギーとフラックスを抽出
            component_energy_array = {}
            component_spectra_array = {}
            for component in ["primary","secondary"]:
                spectra_table = \
                    self.hawking_radiation.integral_radiation_spectra[f"{particle}_{component}"][str_idx:]
                
                energy_bins = np.array([
                    u.Quantity(colname).to(u.GeV).value for colname in spectra_table.colnames[1:]
                ]) * u.GeV
                component_energy_array[component] = energy_bins
                
                spectra_list = []
                for colname in spectra_table.colnames[1:]:
                    #TODO:合ってるか確認
                    # Convert from 1/(GeV cm3) to count number/GeV
                    # 最初の行（str_idx）の値にはtime_factorをかける
                    col_zeroed = (spectra_table[colname][0] * time_factor * u.cm**3).to(u.GeV**(-1)).value
                    # 最初の行を除外して合計
                    col_after_zeroed = np.sum((spectra_table[colname][1:] * u.cm**3).to(u.GeV**(-1)).value)
                    
                    spectra_list.append(
                        (col_zeroed + col_after_zeroed)
                    )
                component_spectra_array[component] = np.array(spectra_list).T * u.GeV**(-1)

            
            # creationg short-term signal skymodel for PBH
            time_interval = Time(tstop) - Time(tstart)
            distance_factor = 4 * np.pi * (distance.to(u.cm))**2
            signal_skymodels = []
            
            primary_spectral_model = TemplateSpectralModel(
                energy = component_energy_array["primary"].to(u.MeV),
                values = component_spectra_array["primary"].to(1/u.MeV) / distance_factor / (time_interval.sec * u.s)
            )
            secondary_spectral_model = TemplateSpectralModel(
                energy = component_energy_array["secondary"].to(u.MeV),
                values = component_spectra_array["secondary"].to(1/u.MeV) / distance_factor / (time_interval.sec * u.s)
            )
            
            total_spectral_model = CompoundSpectralModel(
                primary_spectral_model,
                secondary_spectral_model,
                operator=lambda x, y: x + y
            )
            spatial_model = PointSpatialModel(
                lon_0=self.ra,
                lat_0=self.dec,
                frame="icrs"
            )
            signal_skymodels.append(
                                        SkyModel(
                                                    spectral_model=total_spectral_model,
                                                    spatial_model=spatial_model
                                                    #name=None
                                                )
                                    )
        return Models(signal_skymodels)

    def _create_evolution_models(
        self,
        particle_list=["photon"],
        decay_functions_list=[{}],
        rest_mass_list=[0],
        charge_list=[0],
        flavour_list=[None],
        spin_list=[1],
        ndof_list=[2]
    ):
        """
        Generate SkyModel objects from PBH data, considering observation start time.
        
        Parameters:
        - observation_start_time (str): ISO format string representing the start of observation.
        """
        distance_factor = 4 * np.pi * (self.distance.to(u.cm))**2

        for i, particle in enumerate(particle_list):
            if "photon" not in particle.lower():
                logger.error("Only photons can be modeled now! %s is not a photon.", particle)
        
            # primary成分とsecondary成分ごとにエネルギーとフラックスを抽出
            component_energy_array = {}
            component_spectra_array = {}
            for component in ["primary","secondary"]:
                spectra_table = self.hawking_radiation.integral_radiation_spectra[f"{particle}_{component}"]
                
                energy_bins = np.array([
                    u.Quantity(colname).to(u.GeV).value for colname in spectra_table.colnames[1:]
                ]) * u.GeV
                
                spectra_list = []
                for colname in spectra_table.colnames[1:]:
                    # Convert from 1/(GeV cm3) to count number/GeV
                    #TODO:合ってるか確認
                    spectra_list.append(
                        (spectra_table[colname] * u.cm**3).to(u.GeV**(-1)).value
                    )
            
                component_spectra_array[component] = np.array(spectra_list).T * u.GeV**(-1)
            
            # creationg short-term signal skymodel for PBH
            signal_skymodels = []
            for (primary_spectrum, secondary_spectrum) in zip(component_spectra_array["primary"],
                                                              component_spectra_array["secondary"]):
                primary_spectral_model = TemplateSpectralModel(
                    energy = energy_bins.to(u.MeV),
                    values = primary_spectrum.to(1/u.MeV) / distance_factor / u.s
                )
                secondary_spectral_model = TemplateSpectralModel(
                    energy = energy_bins.to(u.MeV),
                    values = secondary_spectrum.to(1/u.MeV) / distance_factor / u.s
                )
                
                total_spectral_model = CompoundSpectralModel(
                    primary_spectral_model,
                    secondary_spectral_model,
                    operator=lambda x, y: x + y
                )
                spatial_model = PointSpatialModel(
                    lon_0=self.ra,
                    lat_0=self.dec,
                    frame="icrs"
                )
                signal_skymodels.append(
                                            SkyModel(
                                                        spectral_model=total_spectral_model,
                                                        spatial_model=spatial_model
                                                        #name=None
                                                    )
                                        )
        return Models(signal_skymodels)
        
    def plot_particle_flux(
        self,
        particle_list=["photon"],
        decay_functions_list=[{}],
        rest_mass_list=[0],
        charge_list=[0],
        flavour_list=[None],
        spin_list=[1],
        ndof_list=[2]
    ):
        evolution_models = self._create_evolution_models(
            particle_list=particle_list,
            decay_functions_list=decay_functions_list,
            rest_mass_list=rest_mass_list,
            charge_list=charge_list,
            flavour_list=flavour_list,
            spin_list=spin_list,
            ndof_list=ndof_list
        )

        fig = plt.figure(figsize=(12, 8))
        #self._bhawk.plot_evolution(fig_evol=fig)
            
        # axis for time to evaporation spectrum
        ax_spec = plt.subplot2grid(
            (6, 2), (2, 0),
            rowspan=3, colspan=2,
            fig=fig
        )
        
        # slider-axis for time to evaporation
        ax_evaporation = fig.add_axes(
            [0.3, 0.08, 0.4, 0.02],
            facecolor='lightsteelblue'
        )
        
        evaporation_slider = Slider(
            ax_evaporation,
            'log10(Time to evaporation [s])',
            self._time_to_evaporation_log[-1],
            self._time_to_evaporation_log[0],
            valstep=np.flip(self._time_to_evaporation_log),
            valinit=self._time_to_evaporation[0],
            initcolor='hotpink'
        )

        # time to evaporation colorbar
        cmap_time = plt.cm.get_cmap('rainbow')
        norm_time = mpl.colors.LogNorm(
            vmax=10**self._time_to_evaporation_to_zero_log[0],
            vmin=10**self._time_to_evaporation_to_zero_log[-1]
            )
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm_time, cmap=cmap_time),
            ax=ax_spec,
            label='Time to evaporation [s]'
            )
            
        # reset button
        ax_flux_reset = fig.add_axes([0.8, 0.08, 0.1, 0.02])
        button_flux_reset = Button(
            ax_flux_reset,
            'Reset',
            color='g', hovercolor='r'
            )

        def show_spectrum(val):
            '''Draws primary/secondary photon spectra of the blackhole (for 4pi sr).'''
            
            # 現在のスライダーに対応するインデックスの情報をそれぞれ取得
            ival = np.absolute(self._time_to_evaporation_log - val).argmin()
            ph_radiation1 = self._radiation_profiles['photon_primary'].spectra[ival]
            ph_radiation2 = self._radiation_profiles['photon_secondary'].spectra[ival]
            dt = self._delta_times[ival]
            t_to_evap = self._time_to_evaporation[ival]
            
            # スペクトルのプロット
            signal_models = self.create_short_term_signal_skymodels(ph_radiation1, ph_radiation2, dt, t_to_evap)
            signal_models.spectral_model.model1.plot(
                ax=ax_spec,
                energy_bounds=(1e-6, 100) * u.TeV,
                sed_type='dnde',
                ls='-', lw=1, marker='o', ms=2, alpha=1.0,
                color=cmap_time(norm_time(t_to_evap)),
                label='photon primary at {0:1.2E} s'.format(t_to_evap),
                )
            signal_models.spectral_model.model2.plot(
                ax=ax_spec,
                energy_bounds=(1e-6, 100) * u.TeV,
                sed_type='dnde',
                ls='-', lw=1, marker=',', ms=0, alpha=0.5,
                color=cmap_time(norm_time(t_to_evap)),
                label='photon_secondary at {0:1.2E} s'.format(t_to_evap),
                )
            # Add vertical dashed lines
            energy_axis = self._config.getEnergyAxis()
            emin = energy_axis.bounds.to(u.TeV).min().value
            emax = energy_axis.bounds.to(u.TeV).max().value
            ax_spec.axvline(x=emin,
                            color='gray',
                            linestyle='--',
                            )
            ax_spec.axvline(x=emax,
                            color='gray',
                            linestyle='--'
                            )
            ax_spec.legend()
            evaporation_slider.poly.set(facecolor=cmap_time(norm_time(t_to_evap)))
            fig.canvas.draw_idle()
            
        def reset_flux(event):
            evaporation_slider.reset()
            ax_spec.clear()

        evaporation_slider.on_changed(show_spectrum)
        button_flux_reset.on_clicked(reset_flux)
        
        plt.tight_layout()
        plt.show()


        



'''
        for i, particle in enumerate(particle_list):
            if "photon" not in particle.lower():
                logger.error("Only photons can be modeled now! %s is not a photon.", particle)
            
            # 引数で与えられたparticleの名前を元に、Particleインスタンスの作成
            PARTICLE = Particle.Particle(
                particle, decay_functions_list[i], rest_mass_list[i], charge_list[i], flavour_list[i], spin_list[i], ndof_list[i]
            )
            
            # primary成分とsecondary成分をまとめる
            radiations = {}
            for component in ["primary","secondary"]:
                spectra_table = self.hawking_radiation.integral_radiation_spectra[f"{particle}_{component}"]
                
                energy_bins = np.array([
                    u.Quantity(colname).to(u.GeV).value for colname in spectra_table.colnames[1:]
                ]) * u.GeV
                
                spectra_list = []
                for colname in spectra_table.colnames[1:]:
                    # Convert from 1/(GeV cm3) to count number/GeV
                    #TODO:合ってるか確認
                    spectra_list.append(
                        (spectra_table[colname] * u.cm**3).to(u.GeV**(-1)).value
                    )
                spectra_qarray = np.array(spectra_list).T * u.GeV**(-1)
        
                # さっきのリストと何が違う？
                spectra_list = []
                for irow in reversed(range(len(spectra_table))):
                    spectrum_values = spectra_qarray[irow]
                    particle_graph_set = (spectrum_values, energy_bins)
                    spectra_list.append(
                        Particle.ParticleDistribution(
                            particle=PARTICLE,
                            position=(self.ra, self.dec, self.distance),
                            spectrum=particle_graph_set,  # particle_hist
                            name=f"{particle}_{component}",
                            spec_hist=False,
                        )
                    )
                radiations[f"{particle}_{component}"] = \
                    Particle.ParticleDistributionProfile(spectra_list, energy_bins)
        
        
            distance_factor = 4 * np.pi * (self.distance.to(u.cm))**2
            signal_skymodels = []
            for irow, (ph_radiation1, ph_radiation2) in \
                enumerate(zip(radiations[f"{particle}_primary"].spectra, radiations[f"{particle}_secondary"].spectra)):
                """ph_radiation1 and ph_radiation2 must be a ParticleDistribution,
                which has attributes of .spectrum and .position."""
                
                primary_flux = ph_radiation1.spectrum[0].to(1/u.MeV) / distance_factor / u.s
                primary_spectral_model = TemplateSpectralModel(
                    energy = ph_radiation1.spectrum[1].to(u.MeV),
                    values = primary_flux
                )
                
                secondary_flux = ph_radiation2.spectrum[0].to(1/u.MeV) / distance_factor / u.s
                secondary_spectral_model = TemplateSpectralModel(
                    energy = ph_radiation2.spectrum[1].to(u.MeV),
                    values = secondary_flux
                )
                
                total_spectral_model = CompoundSpectralModel(
                    primary_spectral_model,
                    secondary_spectral_model,
                    operator=lambda x, y: x + y
                )
                spatial_model = PointSpatialModel(
                    lon_0=self.ra,
                    lat_0=self.dec,
                    frame="icrs"
                )
                
                # creationg short-term signal skymodel for PBH
                signal_skymodels.append(
                                            SkyModel(
                                                        spectral_model=total_spectral_model,
                                                        spatial_model=spatial_model
                                                        #name=None
                                                    )
                                        )
        return Models(signal_skymodels)
'''
