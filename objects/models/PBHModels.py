#/usr/Astro/bh-simulator/
from pathlib import Path
import os
import sys
import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mpl_colors
from matplotlib.widgets import Slider, Button, RangeSlider
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from gammapy.maps import RegionGeom
from regions import CircleSkyRegion
from gammapy.modeling.models import (
    TemplateSpectralModel,
    CompoundSpectralModel,
    SkyModel,
    Models,
    PointSpatialModel,
)
from gammapy.datasets import MapDataset
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

from gammapy.maps import MapAxis, WcsGeom, HpxGeom, WcsNDMap, HpxNDMap

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'object')
    )
)
from objects import AstroObject
from objects import Particle

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'hawking')
    )
)
from hawking.BlackHawk import BlackHawk


from logging import getLogger, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel('INFO')
logger.addHandler(handler)
logger.setLevel('INFO')

class PBHModels:
    def __init__(
        self,
        blackhole=None,
        evolutions=None,
        hawking_spectra=None,
    ):
        """        
        Parameters:
        ----------
        blackhole : `~ hawking.AstroObject.BlackHole`
            ブラックホールに関する設定パラメータの辞書
            ブラックホールの質量、距離、その他の関連パラメータを含む
            カウントする領域の中心座標や半径を指定する

        hawking_spectra : `~ astropy.table.QTable`
            radiation_spectra_table_dict.get("photon_primary")およびradiation_spectra_tables.get("photon_secondary")
            のデータのみが使われる
        それぞれが対応するリストの数を保持している
        """
    def __init__(self):
        self._blackhole = blackhole
        self._evolutions = evolutions
        self._hawking = hawking_spectra
    
    
    def spatial_model(self):
        pass
    
    def spectrum_model(self):
        pass
        
    def temporal_model(self):
        pass

    def get_short_term_signal_models(self, coordsys="GAL"):
        '''ph_radiation1 and ph_radiation2 must be a ParticleDistribution,
        which has attributes of .spectrum and .position.'''
    
        #logger.info(f'Creating short term signal models: \n Lifetime: {t_to_evap} s \n Duration: {dt} \n Distance: {self.blackhole.position["distance"]}')

        # creating short-time signal spectrum model for PBH
        distance_factor = 4 * np.pi * (self._blackhole.position["distance"].to(u.cm))**2

        pbh_skymodels = []
        for irow, photon_total in \
            zip(self._hawking_spectra["photon_primary"].spectra,
                self._hawking_spectra["photon_secondary"].spectra):
            
            dt = self._delta_times_to_evap[irow]
            photon_total_energy = photon_total.spectrum[1].to(u.MeV)
            print(photon_total_energy)
            photon_total_flux = photon_total.spectrum[0].to(1/u.MeV) / (distance_factor * dt.to(u.s))
            
            spectral_model   = TemplateSpectralModel(
                                    energy = photon_total_energy,
                                    values = photon_total_flux
                                )
            if coordsys == "GAL":
                spatial_model = PointSpatialModel(
                                    lon_0= self._blackhole.sky_coord().transform_to('galactic').l,
                                    lat_0= self._blackhole.sky_coord().transform_to('galactic').b,
                                    frame="galactic"
                                )
            if coordsys == "CEL":
                spatial_model = PointSpatialModel(
                                    lon_0= self._blackhole.sky_coord().ra,
                                    lat_0= self._blackhole.sky_coord().dec,
                                    frame="icrs"
                                )
            
            # creationg short-term signal skymodel for PBH
            pbh_skymodels.append(
                                SkyModel(
                                            spectral_model=spectral_model,
                                            spatial_model=spatial_model,
                                            name= f'{self._times_to_evap[irow]}',
                                        )
                                )
        return Models(pbh_skymodels)


    def plot_angular_displacement(self, angular=0.5):
        """
        地球の公転速度とポテンシャルエネルギーによるビリアル定理
        詳しくは、Search for Gamma-Ray Emission from Local Primordial Black Holes with the Fermi Large Area Telescopeの2節参照
        """
        # 定数
        perpendicular_velocity = 300 * u.km / u.s  # km/s
        alpha = angular * np.pi / 180  # 1 degree -> radians

        # 観測時間 (秒)
        one_day = 1 * u.day
        ten_year = 10 * u.year
        observation_time = np.logspace(np.log10(one_day.to(u.s).value), np.log10(ten_year.to(u.s).value), 100)  # 1日から10年まで -> second

        # 距離 R (パーセク)
        distance_to_pbh = (perpendicular_velocity.to(u.pc/u.s) * observation_time) / alpha

        # プロット
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # 左側のy軸 (秒)
        ax1.plot(distance_to_pbh, observation_time, label="Angular displacement = 1°")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Distance to PBH [pc]")
        ax1.set_ylabel("Observation Time [sec]")
        ax1.tick_params(axis='y')
        ax1.grid(True, which="both", ls="--")

        # 右側のy軸 (年)
        ax2 = ax1.twinx()
        observation_time_years = (observation_time * u.s).to(u.year).value
        ax2.plot(distance_to_pbh, observation_time_years)
        ax2.set_yscale('log')
        ax2.set_ylabel("Observation Time [years]")
        ax2.tick_params(axis='y')

        # タイトルと凡例
        plt.title("Distance to PBH vs. Observation Time")
        fig.tight_layout()
        plt.show()
"""
球面全体をHealPixで1degに分割
    gammapy.HpxMap or SkyDivisionを使う

PBHAnalyzerを使ってmapdataset, バックグラウンドのカウント数を算出（とりあえずはnpredでいいけど、最終的にはポアソン揺らぎを考慮する必要があるから、fakeカウント等を使って発生させた値をカウんと）
→この時の条件は一つ（観測時間とう）に絞って、何回も乱数を振って(例えば、fakeを何回も実行して)作ったバックグラウンドカウントデータを用意する（Qtable）
バックグラウンドによる揺らぎを再現する


全天 4π(sr)より全天を1degに分割したら、4π/π(π/180)**2 = 1.3*10**4個程度に分けられる
5σが6*10**-7より、

4π/π(π/180)**2 * 5σ = 8*10**-3　の確率で5σの事象が発生する（大体1%ぐらい）
ということはバックグラウンドの揺らぎによって偽の5σが1%ぐらいは発生する
FAR = 0.1 回/年

↑↑↑↑↑↑ここまではバックグラウンドの話

↓↓↓↓↓↓PBH蒸発率(発生率)をどうやって見積もるか

PBHシグナルに唯一必要なパラメータはPBH蒸発率ρ←1cubic per secあたりに何回バーストが起きるかの指標

ランダムにPBHを発生させるには、
・いつ蒸発するか
・どの位置で蒸発するか
の二つが必要。


一番簡単な例
質量（寿命）と観測時間は固定する、横軸距離、縦軸個数の分布
                            横軸明るさ、縦軸個数の分布

"""
/Users/omooon/pbh-search/pbh-analyzer/irfs/ftools

class SkyDistribution:
    pass
    
