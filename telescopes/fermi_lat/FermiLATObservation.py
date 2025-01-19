import os
import re
from pathlib import Path
from pbh_explorer.utils import PathChecker as PathC

from astropy import units as u
from gammapy.data import GTI
import glob
import subprocess
from GtApp import GtApp
import gt_apps
import os
import shutil
from pathlib import Path
import json
import numpy as np

import yaml

from gammapy.datasets import MapDataset
from gammapy.maps import Map, MapAxis, WcsGeom, WcsNDMap, HpxGeom
from gammapy.irf import PSFMap, EDispKernelMap

from astropy.time import Time
import pandas as pd

script_dirpath = os.path.dirname(__file__)

from astropy import units as u
from astropy.coordinates import SkyCoord

# %matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from gammapy.data import EventList
from gammapy.datasets import Datasets, MapDataset
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map, MapAxis, WcsGeom, HpxGeom, WcsNDMap, HpxNDMap
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    Models,
    PointSpatialModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)


from logging import getLogger, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)

class WeeklyData:
    def __init__(self):
        # Timestamps used uniformly by Fermi satellites. (https://heasarc.gsfc.nasa.gov/W3Browse/fermi/fermilweek.html#week_number)
        reference_time = Time("2001-01-01 00:00:00", scale="utc")
        
        # The start time, in UTC, for data in the weekly photon file.
        utc_obs_start = Time("2008-08-04 15:43:36", scale="utc")
        
        # The stop time (current time), in UTC, for data in the weekly photon file.
        utc_obs_end = Time.now()
        
        # The observation start time given in Mission Elapsed Time (MET). Mission Elapsed Time is measured in seconds from 2001.0.
        self._met_obs_start = (utc_obs_start - reference_time).sec
        
        # The observation stop time given in Mission Elapsed Time (MET). Mission Elapsed Time is measured in seconds from 2001.0.
        self._met_obs_end   = (utc_obs_end - reference_time).sec
        
        # 1週間の時間（秒）
        one_week_seconds = 7 * 24 * 60 * 60

        # Fermiの「週の開始」は木曜日の00:00:00 UTCなので、基準日から最初の木曜日を求める
        # まず基準時間から最も近い木曜日を計算
        days_since_start = (utc_obs_start - reference_time).jd  # Julian Day（ユリウス日）で経過日数を取得
        days_since_start_int = int(np.floor(days_since_start))  # 整数部分を取得

        # 日付を曜日に基づいて調整（木曜日は曜日番号3）
        weekday_start = (days_since_start_int + 3) % 7  # 木曜日への補正
        adjusted_start_time = utc_obs_start - TimeDelta(weekday_start, format='jd')

        # 最初の木曜日から週数を計算
        weeks_since_start = (utc_obs_start - adjusted_start_time).sec / one_week_seconds

        # 週番号を整数に切り上げる（最初の週を0からカウント）
        week_number = int(np.floor(weeks_since_start))
        
        
        # The Fermi mission week number. Each week begins Thursdays at 00:00:00 UTC.
        #nweeks =

    def split_data_file(lat_photon_weekly_file, time_resolution="daily"):
        pass

    def combine_datafiles(lat_photon_weekly_files):
        # combine the weekly files into a single file
        #!ls lat_photon_weekly* > filelist.txt
        subprocess.run(["punlearn", "gtselect"])
        
        # Combining the files takes 10-15 minutes for the full dataset.
        gt_apps.filter["evclass"] = "INDEF"
        gt_apps.filter["evtype"]  = "INDEF"
        gt_apps.filter["infile"]  = "@filelist.txt"
        gt_apps.filter["outfile"] = "lat_alldata.fits"
        gt_apps.filter["ra"]      = 0
        gt_apps.filter["dec"]     = 0
        gt_apps.filter["rad"]     = 180
        gt_apps.filter["emin"]    = 30
        gt_apps.filter["emax"]    = 1000000
        gt_apps.filter["tmin"]    = "INDEF"
        gt_apps.filter["tmax"]    = "INDEF"
        gt_apps.filter["zmax"]    = 180
        gt_apps.filter["chatter"] = 4

from gammapy.data import EventList

class PartialSkyRegion:
    def __init__(
        self,
        photon_filename="/Users/omooon/pbh_explorer/weekly/lat_photon_weekly_w843_p305_v001.fits",
        spacecraft_filename="/Users/omooon/pbh_explorer/weekly/lat_spacecraft_weekly_w843_p310_v001.fits",
        galdiff_filename="/Users/omooon/miniconda3/envs/fermipy/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits",
        isodiff_filename="/Users/omooon/miniconda3/envs/fermipy/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt",
        catalogs=[],
        region_params=('lon', 'lat', 'coordsys', 'width', 'binsz'),
        energy_params=('emin', 'emax', 'binsperdec'),
        loglevel=None,
        **kwargs
    ):
        if loglevel:
            logger.setLevel(loglevel)
            for handler in logger.handlers:
                handler.setLevel(loglevel)
                
        self.photon_filename = photon_filename
        self.spacecraft_filename = spacecraft_filename
        self.galdiff_filename = galdiff_filename
        self.isodiff_filename = isodiff_filename
        self.catalogs = catalogs

        if region_params[2] == "galactic":
            galactic_skycoord = SkyCoord(
                l=region_params[0]*u.deg,
                b=region_params[1]*u.deg,
                frame=region_params[2]
            )
            icrs_skycoord = galactic_skycoord.icrs

            center_l = region_params[0]
            center_b = region_params[1]
            center_ra = icrs_skycoord.ra.value.item()
            center_dec = icrs_skycoord.dec.value.item()
            coordsys = "GAL"
        elif region_params[2] == "icrs":
            icrs_skycoord = SkyCoord(
                ra=region_params[0]*u.deg,
                dec=region_params[1]*u.deg,
                frame=region_params[2]
            )
            galactic_skycoord = icrs_skycoord.galactic
        
            center_ra = region_params[0]
            center_dec = region_params[1]
            center_l = galactic_skycoord.l.value.item()
            center_b = galactic_skycoord.b.value.item()
            coordsys = "CEL"

        gti = GTI.read(photon_filename)
        met_start, met_stop = gti.met_start[0].value.item(), gti.met_start[-1].value.item()

        # Extract numbers following ‘w’ in regular expressions
        match1 = re.search(r'w\d+', os.path.basename(self.photon_filename))
        match2 = re.search(r'w\d+', os.path.basename(self.spacecraft_filename))
        if not match1.group() == match2.group():
            return -1
        work_dir = os.getcwd()
        outdir = f"{work_dir}/weekly/{match1.group()}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.config = None
        self._init_config(
            (center_ra, center_dec, center_l, center_b, coordsys, region_params[3], region_params[4]),
            (met_start, met_stop),
            energy_params,
            outdir,
            **kwargs
        )
        
    def _init_config(
        self,
        region_params,
        time_duration,
        energy_params,
        outdir,
        evtype=3,
        evclass=128,
        zmax=90,
        roicut="no",
        filter="(DATA_QUAL>0)&&(LAT_CONFIG==1)",
        irfs="P8R3_SOURCE_V3"
    ):
        self.config = {
            "data": {
                "evfile": self.photon_filename,
                "scfile": self.spacecraft_filename,
            },
            "binning": {
                "binsperdec": energy_params[2],
                "binsz": region_params[6],
                "coordsys": region_params[4],
                "proj":"CAR",
                "projtype": "WCS",
                "roiwidth": region_params[5],
            },
            "selection": {
                "ra": region_params[0],
                "dec": region_params[1],
                "glon": region_params[2],
                "glat": region_params[3],
                "emin": energy_params[0],
                "emax": energy_params[1],
                "zmax": zmax,
                "evclass": evclass,
                "evtype": evtype,
                "roicut": roicut,
                "tmin": time_duration[0],
                "tmax": time_duration[1],
                "filter": filter,
            },
            "gtlike": {
                "edisp": True,
                "irfs": irfs,
                "edisp_disable": ["isodiff", "galdiff"],
            },
            "model": {
                "src_roiwidth": region_params[5] + 5,
                "galdiff": self.galdiff_filename,
                "isodiff": self.isodiff_filename,
                "catalogs": self.catalogs,
            },
            "fileio": {
                "outdir": outdir,
            },
        }
        logger.info(self.config)

    def delete_files_in_directory(self):
        # ディレクトリ内のすべてのファイルを取得
        files = glob.glob(os.path.join(outdir, '*'))
        
        # ファイルを削除
        for file in files:
            try:
                if os.path.isdir(file):  # ディレクトリの場合
                    os.rmdir(file)  # サブディレクトリを削除（空の場合）
                else:
                    os.remove(file)  # ファイルを削除
                print(f"{file} を削除しました。")
            except Exception as e:
                print(f"削除エラー: {file} - {e}")

    def fermipy_binned_likelihood_analysis(self, clear_outdir=True):
    
        if clear_outdir:
            self.delete_files_in_directory()

        with open("config.yaml", 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
        
        from fermipy.gtanalysis import GTAnalysis
        self.gta = GTAnalysis("config.yaml", logging={'verbosity' : 3})
        
        self.gta.setup()
        self.gta.optimize()

        # Free all parameters of isotropic and galactic diffuse components
        self.gta.free_source("galdiff")
        self.gta.free_source("isodiff")

        self.gta.free_sources(minmax_ts=[10,None],pars='norm')              # Free sources with TS > 10
        self.gta.free_sources(minmax_ts=[None,10],free=False,pars='norm')   # Fix sources with TS < 10
        #self.gta.free_sources(minmax_npred=[10,100],free=False,pars='norm') # Fix sources with 10 < Npred < 100
        
        self.gta.fit()
        self.gta.print_roi()

    def fermitools_binned_likelihood_analysis(self, clear_outdir=True):
        if clear_outdir:
            self.delete_files_in_directory()

        with open("config.yaml", 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

        logger.info(f"\nRunning gtselect...")
        subprocess.run(["punlearn", "gtselect"])
        gt_apps.filter["infile"]  = self.config["data"]["evfile"]
        gt_apps.filter["outfile"] = os.path.join(self.config["fileio"]["outdir"], "ft1_00.fits")
        gt_apps.filter["ra"]      = self.config["selection"]["ra"]
        gt_apps.filter["dec"]     = self.config["selection"]["dec"]
        gt_apps.filter["rad"]     = self.config["binning"]["roiwidth"]
        gt_apps.filter["zmax"]    = self.config["selection"]["zmax"]
        gt_apps.filter["emin"]    = self.config["selection"]["emin"]
        gt_apps.filter["emax"]    = self.config["selection"]["emax"]
        gt_apps.filter["tmin"]    = self.config["selection"]["tmin"]
        gt_apps.filter["tmax"]    = self.config["selection"]["tmax"]
        gt_apps.filter["evclass"] = self.config["selection"]["evclass"]
        gt_apps.filter["evtype"]  = self.config["selection"]["evtype"]
        gt_apps.filter.run()
        
        print(f"\nRunning gtmktime... ")
        subprocess.run(["punlearn", "gtmktime"])
        gt_apps.maketime["evfile"]  = os.path.join(self.config["fileio"]["outdir"], "ft1_00.fits")
        gt_apps.maketime["outfile"] = os.path.join(self.config["fileio"]["outdir"], "ft1_filtered_00.fits")
        gt_apps.maketime["scfile"]  = self.config["data"]["scfile"]
        gt_apps.maketime["filter"]  = self.config["selection"]["filter"]
        gt_apps.maketime["roicut"]  = self.config["selection"]["roicut"]
        gt_apps.maketime.run()


class AllSkyRegion:
    def __init__(
        self,
        photon_filename="/Users/omooon/pbh_explorer/weekly/lat_photon_weekly_w843_p305_v001.fits",
        spacecraft_filename="/Users/omooon/pbh_explorer/weekly/lat_spacecraft_weekly_w843_p310_v001.fits",
        galdiff_filename="/Users/omooon/miniconda3/envs/fermipy/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits",
        isodiff_filename="/Users/omooon/miniconda3/envs/fermipy/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt",
        catalogs=[],
        coordsys="galactic",
        binsz=1,
        energy_params=(100, 1000000, 8),
        loglevel=None,
        **kwargs
    ):
        if loglevel:
            logger.setLevel(loglevel)
            for handler in logger.handlers:
                handler.setLevel(loglevel)

        partial = PartialSkyRegion(
            photon_filename=photon_filename,
            spacecraft_filename=spacecraft_filename,
            galdiff_filename=galdiff_filename,
            isodiff_filename=isodiff_filename,
            catalogs=catalogs,
            region_params=(0, 0, "galactic", 180, binsz),
            energy_params=(energy_params[0], energy_params[1], energy_params[2]),
            loglevel=None,
        )
        self.config = partial.config

    def duplicate(self):
        # 重複イベントを処理するために使うマップ
        # これをfermipyで実行したマップにしてみる？
        duplicate_geom = WcsGeom.create(
            binsz=binsz,
            proj="CAR",
            frame=coordsys,
            axes=None,
            skydir=(0, 0),
            width=(360, 180)
        )
        self._duplicate_map= WcsNDMap(
            duplicate_geom,
            data=np.zeros(duplicate_geom.data_shape)
        )
    
        self.partial_skyregions = []
        map_size = 30  # 30°×30° のマップ
        for i in range(72): #72
            # 位置を決定 (i番目の30°×30°の領域)
            # 横方向 (RA): 0, 30, 60, ..., 330
            # 縦方向 (Dec): -90, -60, -30, 0, 30, 60, 90
            row = i // 12  # 縦方向のインデックス (0〜5)
            col = i % 12  # 横方向のインデックス (0〜11)
            logger.info(f"Processing region {i+1}/72 - Row: {row}, Column: {col}")

            # lは0〜360の範囲なので、colを使って計算
            l_center = col * map_size + map_size / 2  # 中心を計算 (0° から 330°)
            # bは-90〜90の範囲なので、rowを使って計算
            b_center = (row - 3) * map_size + map_size / 2  # 中心を計算 (-90° から 90°)
            logger.info(f"l:{l_center}, b:{b_center}")

            # マップの重なりを補正するためのマップ,放射線強度を1で初期化
            partial_duplicate_geom = WcsGeom.create(
                binsz=binsz,
                proj="CAR",
                frame=coordsys,
                axes=None,
                skydir=(l_center, b_center),
                width=(map_size, map_size)
            )
            partial_duplicate_map = WcsNDMap(
                partial_duplicate_geom,
                data=np.ones(partial_duplicate_geom.data_shape)
            )
            
            # リプロジェクトして全体マップに加算
            self._duplicate_map.data += partial_duplicate_map.reproject_to_geom(duplicate_geom).data

            # joukino条件でのpatialregionをfermipyで作成
            partial_skyregion = PartialSkyRegion(
                photon_filename=photon_filename,
                spacecraft_filename=spacecraft_filename,
                galdiff_filename=galdiff_filename,
                isodiff_filename=isodiff_filename,
                catalogs=catalogs,
                region_params=(l_center, b_center, "galactic", map_size, binsz),
                energy_params=energy_params,
                loglevel=loglevel,
                **kwargs
            )
            self.partial_skyregions.append(partial_skyregion)

    def counts_map(self):
        partial_skyregion = self.partial_skyregions[0]
        
        logger.info(f"\nRunning gtselect...")
        subprocess.run(["punlearn", "gtselect"])
        gt_apps.filter["infile"]  = partial_skyregion.config["data"]["evfile"]
        gt_apps.filter["outfile"] = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_00.fits")
        gt_apps.filter["ra"]      = 0
        gt_apps.filter["dec"]     = 0
        gt_apps.filter["rad"]     = 180
        gt_apps.filter["zmax"]    = partial_skyregion.config["selection"]["zmax"]
        gt_apps.filter["emin"]    = partial_skyregion.config["selection"]["emin"]
        gt_apps.filter["emax"]    = partial_skyregion.config["selection"]["emax"]
        gt_apps.filter["tmin"]    = partial_skyregion.config["selection"]["tmin"]
        gt_apps.filter["tmax"]    = partial_skyregion.config["selection"]["tmax"]
        gt_apps.filter["evclass"] = partial_skyregion.config["selection"]["evclass"]
        gt_apps.filter["evtype"]  = partial_skyregion.config["selection"]["evtype"]
        gt_apps.filter.run()
        
        print(f"\nRunning gtmktime... ")
        subprocess.run(["punlearn", "gtmktime"])
        gt_apps.maketime["evfile"]  = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_00.fits")
        gt_apps.maketime["outfile"] = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_filtered_00.fits")
        gt_apps.maketime["scfile"]  = partial_skyregion.config["data"]["scfile"]
        gt_apps.maketime["filter"]  = partial_skyregion.config["selection"]["filter"]
        gt_apps.maketime["roicut"]  = partial_skyregion.config["selection"]["roicut"]
        gt_apps.maketime.run()
        
        events = EventList.read(os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_filtered_00.fits"))
        if partial_skyregion.config["binning"]["coordsys"] == "GAL":
            frame="galactic"
        elif partial_skyregion.config["binning"]["coordsys"] == "CEL":
            frame="icrs"
        counts_geom = WcsGeom.create(
            binsz=partial_skyregion.config["binning"]["binsz"],
            proj="CAR",
            frame=frame,
            axes=None,
            skydir=(0, 0),
            width=(360, 180)
        )
        self.counts_map= WcsNDMap(
            counts_geom,
            data=np.zeros(counts_geom.data_shape)
        )
        self.counts_map.fill_events(events)
    
    def counts_map_each_subregion(self):
        partial_skyregion = self.partial_skyregions[0]

        if partial_skyregion.config["binning"]["coordsys"] == "GAL":
            frame="galactic"
        elif partial_skyregion.config["binning"]["coordsys"] == "CEL":
            frame="icrs"
        counts_geom = WcsGeom.create(
            binsz=partial_skyregion.config["binning"]["binsz"],
            proj="CAR",
            frame=frame,
            axes=None,
            skydir=(0, 0),
            width=(360, 180)
        )
        self.counts_map_sub= WcsNDMap(
            counts_geom,
            data=np.zeros(counts_geom.data_shape)
        )

        for i, partial_skyregion in enumerate(self.partial_skyregions):
            logger.info(f"\nRunning gtselect...")
            subprocess.run(["punlearn", "gtselect"])
            gt_apps.filter["infile"]  = partial_skyregion.config["data"]["evfile"]
            gt_apps.filter["outfile"] = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_00.fits")
            gt_apps.filter["ra"]      = partial_skyregion.config["selection"]["ra"]
            gt_apps.filter["dec"]     = partial_skyregion.config["selection"]["dec"]
            gt_apps.filter["rad"]     = partial_skyregion.config["binning"]["roiwidth"]
            gt_apps.filter["zmax"]    = partial_skyregion.config["selection"]["zmax"]
            gt_apps.filter["emin"]    = partial_skyregion.config["selection"]["emin"]
            gt_apps.filter["emax"]    = partial_skyregion.config["selection"]["emax"]
            gt_apps.filter["tmin"]    = partial_skyregion.config["selection"]["tmin"]
            gt_apps.filter["tmax"]    = partial_skyregion.config["selection"]["tmax"]
            gt_apps.filter["evclass"] = partial_skyregion.config["selection"]["evclass"]
            gt_apps.filter["evtype"]  = partial_skyregion.config["selection"]["evtype"]
            gt_apps.filter.run()
            
            print(f"\nRunning gtmktime... ")
            subprocess.run(["punlearn", "gtmktime"])
            gt_apps.maketime["evfile"]  = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_00.fits")
            gt_apps.maketime["outfile"] = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_filtered_00.fits")
            gt_apps.maketime["scfile"]  = partial_skyregion.config["data"]["scfile"]
            gt_apps.maketime["filter"]  = partial_skyregion.config["selection"]["filter"]
            gt_apps.maketime["roicut"]  = partial_skyregion.config["selection"]["roicut"]
            gt_apps.maketime.run()
            
            events = EventList.read(os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_filtered_00.fits"))
            counts_map_sub= WcsNDMap(
                counts_geom,
                data=np.zeros(counts_geom.data_shape)
            )
            counts_map_sub.fill_events(events)
            data = counts_map_sub.data
            data[data >= 1] = 1
            self.counts_map_sub.data += data
                
        
    def wcs_allsky_analysis_each_subregion(self):
        for i, partial_skyregion in enumerate(self.partial_skyregions):
            partial_skyregion.fermipy_binned_likelihood_analysis(clear_outdir=True)
            
            counts_map = partial_skyregion.gta.counts_map()
            model_counts_map = partial_skyregion.gta.model_counts_map()
            
            if i == 0:
                energy_axis = MapAxis.from_energy_edges(model_counts_map.geom.axes[0].edges)
                if partial_skyregion.config["binning"]["coordsys"] == "GAL":
                    frame="galactic"
                elif partial_skyregion.config["binning"]["coordsys"] == "CEL":
                    frame="icrs"
                all_sky_geom = WcsGeom.create(
                    binsz=partial_skyregion.config["binning"]["binsz"],
                    proj="CAR",
                    frame=frame,
                    axes=[energy_axis],
                    skydir=(0, 0),
                    width=(360, 180)
                )
                all_sky_map = WcsNDMap(
                    all_sky_geom,
                    data=np.zeros(all_sky_geom.data_shape),
                )
                self.counts_map = all_sky_map.copy()
                self.model_counts_map = all_sky_map.copy()
            
            self.counts_map.data += counts_map.reproject_to_geom(all_sky_geom).data
            self.model_counts_map.data += model_counts_map.reproject_to_geom(all_sky_geom).data
        
        self.counts_map.data = self.counts_map.data / self._duplicate_map.data
        self.model_counts_map.data = self.model_counts_map.data / self._duplicate_map.data
        
        self.exposure_map = WcsNDMap.read(
            os.path.join(partial_skyregion.config["fileio"]["outdir"], "bexpmap_00.fits")
        )
        
        subprocess.run(["punlearn", "gtpsf"])
        gtpsf = GtApp("gtpsf", "Likelihood")
        gtpsf["expcube"] = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ltcube_00.fits")
        gtpsf["outfile"] = os.path.join(partial_skyregion.config["fileio"]["outdir"], "psfmap_00.fits")
        gtpsf["irfs"] = partial_skyregion.config["gtlike"]["irfs"]
        gtpsf["evtype"] = partial_skyregion.config["selection"]["evtype"]
        gtpsf["ra"] = 0
        gtpsf["dec"] = 0
        gtpsf["emin"] = partial_skyregion.config["selection"]["emin"]
        gtpsf["emax"] = partial_skyregion.config["selection"]["emax"]
        gtpsf["nenergies"] = self.exposure_map.geom.axes[0].nbin
        gtpsf["thetamax"] = 30
        gtpsf["ntheta"] = 300
        gtpsf.run()
        self.psf_map = PSFMap.read(gtpsf["outfile"], format="gtpsf")

    def create_wcs_all_sky_analysis(self):
        logger.info(f"\nRunning gtselect...")
        subprocess.run(["punlearn", "gtselect"])
        gt_apps.filter["infile"]  = partial_skyregion.config["data"]["evfile"]
        gt_apps.filter["outfile"] = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_00.fits")
        gt_apps.filter["ra"]      = 0
        gt_apps.filter["dec"]     = 0
        gt_apps.filter["rad"]     = 180
        gt_apps.filter["zmax"]    = partial_skyregion.config["selection"]["zmax"]
        gt_apps.filter["emin"]    = partial_skyregion.config["selection"]["emin"]
        gt_apps.filter["emax"]    = partial_skyregion.config["selection"]["emax"]
        gt_apps.filter["tmin"]    = partial_skyregion.config["selection"]["tmin"]
        gt_apps.filter["tmax"]    = partial_skyregion.config["selection"]["tmax"]
        gt_apps.filter["evclass"] = partial_skyregion.config["selection"]["evclass"]
        gt_apps.filter["evtype"]  = partial_skyregion.config["selection"]["evtype"]
        gt_apps.filter.run()
        
        print(f"\nRunning gtmktime... ")
        subprocess.run(["punlearn", "gtmktime"])
        gt_apps.maketime["evfile"]  = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_00.fits")
        gt_apps.maketime["outfile"] = os.path.join(partial_skyregion.config["fileio"]["outdir"], "ft1_filtered_00.fits")
        gt_apps.maketime["scfile"]  = partial_skyregion.config["data"]["scfile"]
        gt_apps.maketime["filter"]  = partial_skyregion.config["selection"]["filter"]
        gt_apps.maketime["roicut"]  = partial_skyregion.config["selection"]["roicut"]
        gt_apps.maketime.run()

        # Calculating the instrument livetime for the entire sky
        subprocess.run(["punlearn", "gtltcube"])
        gt_apps.expCube["evfile"]    = "gtmktime.fits"
        gt_apps.expCube["outfile"]   = "gtltcube.fits"
        gt_apps.expCube["scfile"]    = self._ft2_filename
        gt_apps.expCube["zmax"]      = 90
        gt_apps.expCube["dcostheta"] = 0.025
        gt_apps.expCube["binsz"]     = self._config["space"]["binsz"]
        gt_apps.expCube["chatter"]   = 4
        
        print(f"\nRunning gtltcube...")
        gt_apps.expCube.run()

        # Convolving the livetime with the IRF
        subprocess.run(["punlearn", "gtexpcube2"])
        gtexpCube2 = GtApp("gtexpcube2", "Likelihood")
        gtexpCube2["infile"]    = "gtltcube.fits"
        gtexpCube2["outfile"]   = f"gtexpcube2_{algorithm}.fits"
        gtexpCube2["cmap"]      = f"gtmktime_{algorithm}.fits"
        gtexpCube2["irfs"]      = self._config["event"]["irf_name"]
        gtexpCube2["evtype"]    = self._config["event"]["evtype"]
        gtexpCube2["bincalc"]   = "EDGE"
        gtexpCube2["chatter"]   = 4
        
        print(f"\nRunning gtexpcube2...")
        gtexpCube2.run()
        
        #
        subprocess.run(["punlearn", "gtpsf"])
        gtpsf = GtApp("gtpsf", "Likelihood")
        gtpsf["expcube"]   = "gtltcube.fits"
        gtpsf["outfile"]   = "gtpsf.fits"
        gtpsf["irfs"]      = self._config["event"]["irf_name"]
        gtpsf["evtype"]    = self._config["event"]["evtype"]
        gtpsf["ra"]        = self._config["space"]["center"]["ra"]
        gtpsf["dec"]       = self._config["space"]["center"]["dec"]
        gtpsf["emin"]      = self._config["energy"]["emin"]
        gtpsf["emax"]      = self._config["energy"]["emax"]
        gtpsf["nenergies"] = self._config["energy"]["enumbins"]
        gtpsf["thetamax"]  = 30
        gtpsf["ntheta"]    = 300
        
        print(f"\nRunning gtpsf...")
        gtpsf.run()


    def delete_files_in_directory(self, outdir):
        # ディレクトリ内のすべてのファイルを取得
        files = glob.glob(os.path.join(outdir, '*'))
        
        # ファイルを削除
        for file in files:
            try:
                if os.path.isdir(file):  # ディレクトリの場合
                    os.rmdir(file)  # サブディレクトリを削除（空の場合）
                else:
                    os.remove(file)  # ファイルを削除
                print(f"{file} を削除しました。")
            except Exception as e:
                print(f"削除エラー: {file} - {e}")

    def create_map_dataset(self, mask_fit=None, mask_safe=None):
        self.delete_files_in_directory(self.config["fileio"]["outdir"])
    
        logger.info(f"\nRunning gtselect...")
        subprocess.run(["punlearn", "gtselect"])
        gt_apps.filter["infile"]  = self.config["data"]["evfile"]
        gt_apps.filter["outfile"] = os.path.join(self.config["fileio"]["outdir"], "ft1_00.fits")
        gt_apps.filter["ra"]      = 0
        gt_apps.filter["dec"]     = 0
        gt_apps.filter["rad"]     = 180
        gt_apps.filter["zmax"]    = self.config["selection"]["zmax"]
        gt_apps.filter["emin"]    = self.config["selection"]["emin"]
        gt_apps.filter["emax"]    = self.config["selection"]["emax"]
        gt_apps.filter["tmin"]    = self.config["selection"]["tmin"]
        gt_apps.filter["tmax"]    = self.config["selection"]["tmax"]
        gt_apps.filter["evclass"] = self.config["selection"]["evclass"]
        gt_apps.filter["evtype"]  = self.config["selection"]["evtype"]
        gt_apps.filter.run()
        
        print(f"\nRunning gtmktime... ")
        subprocess.run(["punlearn", "gtmktime"])
        gt_apps.maketime["evfile"]  = os.path.join(self.config["fileio"]["outdir"], "ft1_00.fits")
        gt_apps.maketime["outfile"] = os.path.join(self.config["fileio"]["outdir"], "ft1_filtered_00.fits")
        gt_apps.maketime["scfile"]  = self.config["data"]["scfile"]
        gt_apps.maketime["filter"]  = self.config["selection"]["filter"]
        gt_apps.maketime["roicut"]  = self.config["selection"]["roicut"]
        gt_apps.maketime.run()
    
        events = EventList.read(
            os.path.join(self.config["fileio"]["outdir"], "ft1_filtered_00.fits")
        )
        energy_axis = MapAxis.from_energy_bounds(
            f'{self.config["selection"]["emin"]} MeV',
            f'{self.config["selection"]["emax"]} MeV',
            nbin=1,
            per_decade=True,
            name="energy",
        )
        gc_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
        
        wcs_counts = Map.create(
            skydir=gc_pos,
            npix=(360, 180),
            proj="CAR",
            frame="galactic",
            binsz=1,
            axes=[energy_axis],
            dtype=float,
        )
        wcs_counts.fill_events(events)
        
        hpx_geom = HpxGeom(64, nest=False, frame='galactic', axes=[energy_axis])
        hpx_counts = HpxNDMap(hpx_geom, dtype=float)
        hpx_counts.fill_events(events)

        subprocess.run(["punlearn", "gtltcube"])
        gt_apps.expCube["evfile"]    = os.path.join(self.config["fileio"]["outdir"], "ft1_filtered_00.fits")
        gt_apps.expCube["outfile"]   = os.path.join(self.config["fileio"]["outdir"], "ltcube_00.fits")
        gt_apps.expCube["scfile"]    = self.config["data"]["scfile"]
        gt_apps.expCube["zmax"]      = self.config["selection"]["zmax"]
        gt_apps.expCube["dcostheta"] = 0.025
        gt_apps.expCube["binsz"]     = 1#self.config["binning"]
        gt_apps.expCube["chatter"]   = 4
        print(f"\nRunning gtltcube...")
        gt_apps.expCube.run()

        subprocess.run(["punlearn", "gtexpcube2"])
        gtexpCube2 = GtApp("gtexpcube2", "Likelihood")
        gtexpCube2["infile"]    = os.path.join(self.config["fileio"]["outdir"], "ltcube_00.fits")
        gtexpCube2["cmap"]      = 'none'
        gtexpCube2["outfile"]   = os.path.join(self.config["fileio"]["outdir"], "bexpmap_00.fits")
        gtexpCube2["irfs"]      = self.config["gtlike"]["irfs"]
        gtexpCube2["evtype"]    = self.config["selection"]["evtype"]
        gtexpCube2['nxpix'] = 360 # 3600
        gtexpCube2['nypix'] = 180
        gtexpCube2['binsz'] = 1
        gtexpCube2['coordsys'] = 'GAL'
        gtexpCube2['xref'] = 0
        gtexpCube2['yref'] = 0
        gtexpCube2['axisrot'] = 0
        gtexpCube2['proj'] = 'CAR'
        gtexpCube2['ebinalg'] = 'LOG'
        gtexpCube2['emin'] = self.config["selection"]["emin"]
        gtexpCube2['emax'] = self.config["selection"]["emax"]
        gtexpCube2['enumbins'] = 12#37
        gtexpCube2["bincalc"]   = "EDGE"
        gtexpCube2["chatter"]   = 4
        print(f"\nRunning gtexpcube2...")
        gtexpCube2.run()
        
        exposure = Map.read(
            os.path.join(self.config["fileio"]["outdir"], "bexpmap_00.fits")
        )
        energy_axis_true = MapAxis.from_edges(
            exposure.geom.axes[0].center,
            name="energy_true",
            interp="log"
        )
        wcs_exp_geom = WcsGeom(
            wcs=wcs_counts.geom.wcs,
            npix=wcs_counts.geom.npix,
            axes=[energy_axis_true]
        )
        wcs_exposure = exposure.interp_to_geom(wcs_exp_geom)
        
        hpx_exp_geom = HpxGeom(
            64,
            nest=False,
            frame='galactic',
            axes=[energy_axis_true]
        )
        hpx_exposure = exposure.interp_to_geom(hpx_exp_geom)
        
        # psf
        subprocess.run(["punlearn", "gtpsf"])
        gtpsf = GtApp("gtpsf", "Likelihood")
        gtpsf["expcube"] = os.path.join(self.config["fileio"]["outdir"], "ltcube_00.fits")
        gtpsf["outfile"] = os.path.join(self.config["fileio"]["outdir"], "psfmap_00.fits")
        gtpsf["irfs"] = self.config["gtlike"]["irfs"]
        gtpsf["evtype"] = self.config["selection"]["evtype"]
        gtpsf["ra"] = 0
        gtpsf["dec"] = 0
        gtpsf["emin"] = self.config["selection"]["emin"]
        gtpsf["emax"] = self.config["selection"]["emax"]
        gtpsf["nenergies"] = 10
        gtpsf["thetamax"] = 30
        gtpsf["ntheta"] = 300
        gtpsf.run()
        
        psf = PSFMap.read(
            os.path.join(self.config["fileio"]["outdir"], "psfmap_00.fits"),
            format="gtpsf"
        )

        # edisp
        edisp = EDispKernelMap.from_diagonal_response(
            energy_axis=energy_axis,
            energy_axis_true=energy_axis_true,
        )

        #bkg_events = EventList.read(self.rcdiff_filename)
        #bkg_events = EventList(bkg_events.table[bkg_events.table['ZENITH_ANGLE'] <= 90])
        #bkg_counts_map = HpxNDMap(hpx_geom, dtype=float)
        #bkg_counts_map.fill_events(bkg_events)

        template_diffuse = TemplateSpatialModel.read(
            filename=self.config["model"]["galdiff"],
            normalize=False
        )
        diffuse_iem = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=template_diffuse,
            name="diffuse-iem",
        )
        diffuse_iso = create_fermi_isotropic_diffuse_model(
            filename=self.config["model"]["isodiff"],
            interp_kwargs={"extrapolate": True}
        )
        bkg_models = Models([diffuse_iem, diffuse_iso])
        
        gti = GTI.read(
            os.path.join(self.config["fileio"]["outdir"], "ft1_filtered_00.fits")
        )
        
        self.wcs_dataset = MapDataset(
            models=None,
            counts=wcs_counts,
            exposure=wcs_exposure,
            psf=psf,
            edisp=edisp,
            background=None,
            gti=gti,
            mask_fit=mask_fit,
            mask_safe=mask_safe,
        )
        
        self.hpx_dataset = MapDataset(
            models=None,
            counts=hpx_counts,
            exposure=hpx_exposure,
            psf=psf,
            edisp=edisp,
            background=None,
            gti=gti,
            mask_fit=mask_fit,
            mask_safe=mask_safe,
        )
