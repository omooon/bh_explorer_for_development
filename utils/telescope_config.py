import os
import numpy as np

import gt_apps
from GtApp import GtApp
import subprocess
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord


class FermiConfig:
    def __init__(
        self, 
        center_coord=None, 
        lon_width=5, lat_width=5, binsz=0.1, proj="TAN"
    ):
        """
        Parameters:
        ----------
        center_coord : `~astropy.coordinates.SkyCoord`
            作成例：
            from astropy.coordinates import SkyCoord
            center_coord = SkyCoord(0, 45, unit="deg", frame="galactic")
        
        map_params : dict  
            出力されるマップを指定する辞書変数、マップの中心位置と出力する座標系は、デフォルトでは、roi_regionになる
            RoI の中心位置と範囲の指定 (lon_width, lat_width)
            解像度の指定 (binsz)、マップタイプの指定
        """
        self.TELESCOPE = "FERMI"

        # unit
        self.ENERGY_UNIT = "MeV"
        self.SPACE_UNIT = "degree"
        self.TIME_UNIT = "second"

        # energy for map
        self.ENERGY_MIN = 20
        self.ENERGY_MAX = int(3e+5)
        self.ENERGY_BINS_PER_DEC = 8

        # energy_true for map (IRFを作るときにエネルギー範囲にも使われる)
        self.ENERGY_MIN_TRUE = 0.1 * self.ENERGY_MIN
        self.ENERGY_MAX_TRUE = 10 * self.ENERGY_MAX
        self.ENERGY_BINS_PER_DEC_TRUE = 10

        # space for map
        def setRoI_for_MapSpace(lon, lat):
            if frame == "galactic":
                self.SPACE_L = lon
                self.SPACE_B = lat
                galactic_coords = SkyCoord(lon*u.deg, lat*u.deg, frame="galactic")
                equatorial_coords = galactic_coords.transform_to("icrs")
                self.SPACE_RA = equatorial_coords.ra.deg
                self.SPACE_DEC = equatorial_coords.dec.deg
            if frame == "icrs":
                self.SPACE_RA = lon
                self.SPACE_DEC = lat
                equatorial_coords = SkyCoord(lon*u.deg, lat*u.deg, frame="icrs")
                galactic_coords = equatorial_coords.transform_to("galactic")
                self.SPACE_L = galactic_coords.l.deg
                self.SPACE_B = galactic_coords.b.deg
        setRoI_for_MapSpace(lon, lat)
        self.SPACE_LON_WIDTH = lon_width
        self.SPACE_LAT_WIDTH = lat_width
        self.SPACE_BINSZ = binsz
        self.SPACE_PROJ = proj
        self.FRAME = frame

    def get_IRF_Parameter(self, evclass=128, evtype=3, irfs="P8R3_SOURCE_V3", zmax=90):
        """ 
        map roi のパラメータをもとに作成するIRFの（エネルギー、空間、時間に関する）パラメータを設定する
        """
        irf_params = {"evclass": evclass,
                      "evtype": evtype,
                      "irfs": irfs,
                      "zmax": zmax}

        # map RoI で設定した energy_true を使う
        irf_params["emin"] = self.ENERGY_MIN_TRUE
        irf_params["emax"] = self.ENERGY_MAX_TRUE
        irf_params["enumbins"] = int(np.ceil(np.log10(self.ENERGY_MAX_TRUE / self.ENERGY_MIN_TRUE) * self.ENERGY_BINS_PER_DEC_TRUE))

        # map RoI で設定した width より 5　倍広くとる
        # TODO: とりあえずでやってるので、何倍取ればいいかはもう少しちゃんと検討した方がいい
        irf_params["space_ra"] = self.SPACE_RA
        irf_params["space_dec"] = self.SPACE_DEC
        irf_params["space_l"] = self.SPACE_L
        irf_params["space_b"] = self.SPACE_B
        irf_params["proj"] = self.SPACE_PROJ
        irf_params["space_lon_width"] = min(5 * self.SPACE_LON_WIDTH, 360) # 経度は360度を超えないようにする
        irf_params["space_lat_width"] = min(5 * self.SPACE_LAT_WIDTH, 180) # 緯度は180度を超えないようにする
        irf_params["space_binsz"] = self.SPACE_BINSZ

        return irf_params
        
    def get_Fullsky_Parameter(self, frame="galactic"):
        # unit
        self.ENERGY_UNIT = "MeV"
        self.SPACE_UNIT = "degree"
        self.TIME_UNIT = "second"

        # energy for map
        self.ENERGY_MIN = 20
        self.ENERGY_MAX = int(3e+5)
        self.ENERGY_BINS_PER_DEC = 8

        # energy_true for map (IRFを作るときにエネルギー範囲にも使われる)
        self.ENERGY_MIN_TRUE = 0.1 * self.ENERGY_MIN
        self.ENERGY_MAX_TRUE = 10 * self.ENERGY_MAX
        self.ENERGY_BINS_PER_DEC_TRUE = 10

        # space for map
        def setRoI_for_MapSpace(lon=0, lat=0):
            if frame == "galactic":
                self.SPACE_L = lon
                self.SPACE_B = lat
                galactic_coords = SkyCoord(lon*u.deg, lat*u.deg, frame="galactic")
                equatorial_coords = galactic_coords.transform_to("icrs")
                self.SPACE_RA = equatorial_coords.ra.deg
                self.SPACE_DEC = equatorial_coords.dec.deg
            if frame == "icrs":
                self.SPACE_RA = lon
                self.SPACE_DEC = lat
                equatorial_coords = SkyCoord(lon*u.deg, lat*u.deg, frame="icrs")
                galactic_coords = equatorial_coords.transform_to("galactic")
                self.SPACE_L = galactic_coords.l.deg
                self.SPACE_B = galactic_coords.b.deg
        setRoI_for_MapSpace()
        self.SPACE_LON_WIDTH = 360
        self.SPACE_LAT_WIDTH = 180
        self.SPACE_BINSZ = 1
        self.SPACE_PROJ = "AIT"


class CTAConfig:
    def __init__(self):
        super().__init__()
