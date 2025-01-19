from astropy import constants as c
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from astropy.table import QTable

from tqdm import tqdm

import itertools
import time
from timeit import default_timer as timer

from pathlib import Path
from gammapy.maps import Map, MapAxis, WcsNDMap, HpxNDMap
from gammapy.modeling.models import Models
import numpy as np

from logging import getLogger, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

from matplotlib.widgets import Button

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

class FermiLAT:
    def __init__(self,
        map_dataset,
        loglevel=None,
    ):
        """
        ObsSnapFermiLAT holds information on a short-term observation for Fermi-LAT Telescope.
        Here "short-term" means the IRF can be considered to be constant.'
        
        Parameters:
        ----------
        map_dataset : `~ gammapy.datasets.MapDataset`

        outdir : string
        """
        # loglevel setting
        if loglevel:
            logger.setLevel(loglevel)
            for handler in logger.handlers:
                handler.setLevel(loglevel)
            
        # Checking whether map_dataset.counts, map_dataset.exposure, and map_dataset.background
        # are of the same map type (WcsNDMap or HpxNDMap)
        missing_attributes = []
        if map_dataset.counts is None:
            missing_attributes.append("counts")
        if map_dataset.exposure is None:
            missing_attributes.append("exposure")
        if map_dataset.psf is None:
            missing_attributes.append("psf")
        if map_dataset.edisp is None:
            missing_attributes.append("edisp")
        if map_dataset.gti is None:
            missing_attributes.append("gti")
        if missing_attributes:
            raise ValueError(f"The following attributes are None: {', '.join(missing_attributes)}. The map_dataset must contain {', '.join(missing_attributes)}.")

        # Checking whether map_dataset.counts, map_dataset.exposure, and map_dataset.background
        # are of the same map type (WcsNDMap or HpxNDMap)
        invalid_attributes = []
        if map_dataset.models:
            invalid_attributes.append("models")
        if map_dataset.background:
            missing_attributes.append("background")
        if map_dataset.background_model:
            invalid_attributes.append("background_model")
        if invalid_attributes:
            raise ValueError(f"The following attributes cannot be included: {', '.join(invalid_attributes)}. The map_dataset must 'NOT' contain {', '.join(missing_attributes)}.")

        # Checking whether map_dataset.counts, map_dataset.exposure, and map_dataset.background
        # are of the same map type (WcsNDMap or HpxNDMap)
        if isinstance(map_dataset.counts, type(map_dataset.exposure)) and \
           isinstance(map_dataset.counts, type(map_dataset.background)) and \
           isinstance(map_dataset.counts, (WcsNDMap, HpxNDMap)):
            logger.debug(f"The input MapDataset is based on {type(map_dataset.counts)}.")
        else:
            logger.warn(f"The input MapDataset has mixed types: "
                        f"counts = {type(map_dataset.counts)}, "
                        f"exposure = {type(map_dataset.exposure)}, "
                        f"background = {type(map_dataset.background)}.")
        
        # setting for analysis
        self._set_analysis_data(map_dataset)

    def _set_analysis_data(self, map_dataset):
        logger.debug("=== Starting _set_analysis_data ===")
        logger.debug(f"Input map_dataset:\n{map_dataset}")
    
        # コピーして作業用データをセットアップ
        self.map_dataset = map_dataset.copy()
        logger.debug(f"map_dataset was copied {id(map_dataset)} to {id(self.map_dataset)}")
    
        # データ初期化
        self.data = {}
        self.data["counts"] = self.map_dataset.counts
        self.data["npred_sig"] = None
        self.data["npred_bkg"] = None
        self.data["npred_sig_comp"] = []
        self.data["npred_bkg_comp"] = []
        logger.debug(f"Initialized counts data: {np.sum(self.data['counts'])}")
        logger.debug(f"Initialized npred_sig data: {np.sum(self.data['npred_sig'])}")
        logger.debug(f"Initialized npred_bkg data: {np.sum(self.data['npred_bkg'])}")
    
        # metadata
        self.metadata = {"maptype":{}}
        self.metadata["maptype"]["counts"] = type(self.map_dataset.counts).__name__
        self.metadata["maptype"]["exposure"] = type(self.map_dataset.exposure).__name__
        self.metadata["maptype"]["background"] = type(self.map_dataset.background).__name__
        self.metadata["maptype"]["psf"] = type(self.map_dataset.psf).__name__
        self.metadata["frame"] = self.map_dataset._geom.frame
        if hasattr(self.map_dataset._geom, 'nside') and self.map_dataset._geom.nside is not None:
            self.metadata["resolution"] = f"nside{list(self.map_dataset._geom.nside)}"
        else:
            self.metadata["resolution"] = f"binsz{list(self.map_dataset._geom.pixel_scales.deg)}"
        time_start = self.map_dataset.gti.time_start[0]
        time_stop = self.map_dataset.gti.time_stop[-1]
        self.metadata["obs_duration"] = [time_start.iso, time_stop.iso]
        self.metadata["npred_sig_name"] = []
        self.metadata["npred_bkg_name"] = []
        logger.debug("=== Completed _set_analysis_data ===")
    
    def evaluate_ps_signal(self, skymodels, pixel_scan=True):
        for skymodel in skymodels:
            energy_true_centers = self.map_dataset.geoms["geom_exposure"].axes["energy_true"].center
            energy_true_edges = self.map_dataset.geoms["geom_exposure"].axes["energy_true"].edges

            psf_containment = self.map_dataset.psf.containment(1*u.deg, energy_true_centers)
            integral = skymodel.spectral_model.integral(energy_true_edges[:-1], energy_true_edges[1:]).to(u.TeV/u.TeV/u.s/u.cm**2)
            edisp_kernel = self.map_dataset.edisp.get_edisp_kernel().data

            result = self.map_dataset.exposure.data * psf_containment[:, np.newaxis] * integral[:, np.newaxis]
            
            # result は (12, 49152)、edisp_kernel は (12, 4)なので、結果の形は (4, 49152) になる
            npred_with_edisp = np.dot(edisp_kernel.T, result)
            npred_scan = self.data["counts"].copy()
            npred_scan.data = npred_with_edisp.value
            self.data["npred_sig_comp"].append(npred_scan)
            self.metadata["npred_sig_name"].append(skymodel.name)

    def evaluate_ps_background(self, skymodels, fit=False):
        logger.debug("=== Before evaluate_ps_background ===")
        
        skymodel_names = [skymodel.name for skymodel in skymodels]
        for bkg_name in self.metadata["npred_bkg_name"]:
            if bkg_name in skymodel_names:
                raise ValueError(f"{bkg_name} model already exists.")
            else:
                pass

        # Add models to map_dataset
        logger.debug(f"Before self.map_dataset:\n{self.map_dataset}")
        self.map_dataset.models = (
            Models(skymodels)
            if self.map_dataset.models is None
            else self.map_dataset.models + Models(skymodels)
        )
        
        if fit:
            # TODO: fit前にlon_0, lat_0をfrozenにしておく?
            fit = Fit()
            fit.run(datasets=[self.map_dataset])
            
        bkg_maps_to_add = []
        model_names_to_remove = []
        for skymodel in skymodels:
            bkg_maps_to_add.append(
                self.map_dataset.npred_signal(model_names=[skymodel.name])
            )
            model_names_to_remove.append(skymodel.name)
            
        # bkgスカイモデルをdataset.modelsから削除する
        current_models = self.map_dataset.models
        updated_models = Models([model for model in current_models if model.name not in model_names_to_remove])
        self.map_dataset.models = updated_models
        
        # bkgスカイモデルから作られたマップをdataset.backgroundに移す
        from functools import reduce
        import operator
        self.map_dataset.background = (
            reduce(operator.add, bkg_maps_to_add)
            if self.map_dataset.background is None
            else self.map_dataset.background + reduce(operator.add, bkg_maps_to_add)
        )
        logger.info("ps bkg skymodels info were moved to self.map_dataset.models to self.map_dataset.background")
        logger.debug(f"After self.map_dataset:\n{self.map_dataset}")
        
        # NOTE: append(extend)する順番に注意
        # appendは単一の要素をリストに追加する際、
        # extendはリストやイテラブルを既存のリストに追加する際に
        self.metadata["npred_bkg_name"].extend(model_names_to_remove)
        self.data["npred_bkg_comp"].extend(bkg_maps_to_add)
        self.data["npred_bkg"] = self.map_dataset.background
        logger.debug(f"Current counts data: {np.sum(self.data['counts'])}")
        logger.debug(f"Current npred_sig data: {np.sum(self.data['npred_sig'])}")
        logger.debug(f"Current npred_bkg data: {np.sum(self.data['npred_bkg'])}")
        logger.debug("=== Completed evaluate_ps_background ===")

    def evaluate_diffuse_background(
        self,
        galdiff_filename="/Users/omooon/miniconda3/envs/fermipy/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits",
        isodiff_filename="/Users/omooon/miniconda3/envs/fermipy/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt",
        wcs_map_dataset=None
    ):
        logger.debug("=== Starting evaluate_diffuse_background ===")
        for bkg_name in self.metadata["npred_bkg_name"]:
            if bkg_name == "fermi-diffuse-iem":
                raise ValueError("Galactic diffuse model already exists.")
            if bkg_name == "fermi-diffuse-iso":
                raise ValueError("Isotropic diffuse model already exists.")
        
        logger.info(f"Loading galactic diffuse model from {galdiff_filename}.")
        template_diffuse = TemplateSpatialModel.read(filename=galdiff_filename, normalize=False)
        diffuse_iem_skymodel = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=template_diffuse,
            name="fermi-diffuse-iem"
        )
    
        logger.info(f"Loading isotropic diffuse model from {isodiff_filename}.")
        diffuse_iso_skymodel = create_fermi_isotropic_diffuse_model(
            filename=isodiff_filename,
            interp_kwargs={"extrapolate": True}
        )

        logger.debug(f"Before self.map_dataset:\n{self.map_dataset}")
        if wcs_map_dataset:
            logger.info("Fitting background components with provided WCS map dataset.")
            wcs_map_dataset.models = Models([diffuse_iem_skymodel, diffuse_iso_skymodel])
            fit = Fit()
            fit.run(datasets=[wcs_map_dataset])

            diffuse_iem_map = (
                wcs_map_dataset.npred_signal(model_names=["fermi-diffuse-iem"])
                .reproject_to_geom(
                    self.map_dataset.geoms["geom"],
                    preserve_counts=True
                )
            )
            diffuse_iso_map = (
                wcs_map_dataset.npred_signal(model_names=["fermi-diffuse-iso"])
                .reproject_to_geom(
                    self.map_dataset.geoms["geom"],
                    preserve_counts=True
                )
            )
        else:
            logger.info("Fitting background components with self.map_dataset.")
            
            self.map_dataset.models = (
                Models([diffuse_iem_skymodel, diffuse_iso_skymodel])
                if self.map_dataset.models is None
                else self.map_dataset.models + Models([diffuse_iem_skymodel, diffuse_iso_skymodel])
            )
            fit = Fit()
            fit.run(datasets=[self.map_dataset])
            
            diffuse_iem_map = self.map_dataset.npred_signal(model_names=["fermi-diffuse-iem"])
            diffuse_iso_map = self.map_dataset.npred_signal(model_names=["fermi-diffuse-iso"])
            
            # diffuseモデルをdataset.modelsから削除する
            current_models = self.map_dataset.models
            model_names_to_remove = ["fermi-diffuse-iem", "fermi-diffuse-iso"]
            updated_models = Models([model for model in current_models if model.name not in model_names_to_remove])
            self.map_dataset.models = updated_models
        
        # diffuseモデルから計算されたマップをbackgroundに入れる
        self.map_dataset.background = (
            diffuse_iem_map + diffuse_iso_map
            if self.map_dataset.background is None
            else self.map_dataset.background + diffuse_iem_map + diffuse_iso_map
        )
        logger.info("fermi-diffuse-iem and fermi-diffuse-iso skymodels info were moved to self.map_dataset.models to self.map_dataset.background")
        logger.debug(f"After self.map_dataset:\n{self.map_dataset}")
            
        # NOTE: append(extend)する順番に注意
        # appendは単一の要素をリストに追加する際、
        # extendはリストやイテラブルを既存のリストに追加する際に
        self.metadata["npred_bkg_name"].extend([diffuse_iem_skymodel.name, diffuse_iso_skymodel.name])
        self.data["npred_bkg_comp"].extend([diffuse_iem_map, diffuse_iso_map])
        self.data["npred_bkg"] = self.map_dataset.background
        logger.debug(f"Current counts data: {np.sum(self.data['counts'])}")
        logger.debug(f"Current npred_sig data: {np.sum(self.data['npred_sig'])}")
        logger.debug(f"Current npred_bkg data: {np.sum(self.data['npred_bkg'])}")
        logger.debug("=== Completed evaluate_diffuse_background ===")
        
        
        
        
        
        
        
        
        
        
    '''
    def evaluate_signal_detection(*param_ranges, algorithm="cash"):
        # get observed counts
        counts = obs_snap.data["counts"]

        # get diffuse background npred
        npred_bkgs = None
        for npred_bkg in obs_snap.data["npred_bkg"]:
            if npred_bkgs is None:
                npred_bkgs = npred_bkg
            else:
                npred_bkgs += npred_bkg

        #
        sig_skymodel_names = [i.name for i in obs_snap.data["sig_skymodel"]]
        #lifetime = [f"lifetime{i:.2f}s" for i in np.arange(20, 21, 1)]
        param_ranges = (sig_skymodel_names, *param_ranges)

        # 全ての組み合わせを生成して結果を格納するための多次元行列
        matrix_shape = tuple(len(r) for r in param_ranges)
        matrix = np.empty(matrix_shape, dtype=object)
        for index in np.ndindex(matrix_shape):
            matrix[index] = np.zeros(
                np.shape(obs_snap.data["counts"].data)
            )

        if algorithm == "cash":
            from gammapy.stats import cash
            cash_null = cash(counts.data, npred_bkgs.data)
            
            cash_full_matrix = matrix.copy()
            for idx, params in enumerate( list(itertools.product(*param_ranges)) ):
                print(idx)
                match = re.match(r'distance([0-9.]+)([a-zA-Z]+)', params[1])
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    distance = value * u.Unit(unit)
                else:
                    print("Invalid format")
                
                params[0]
                distance = params[1]
                print(params)
    
    distance = [f"distance{i:.2f}pc" for i in np.arange(1, 10, 1)]
    evaluate_signal_detection(distance)
    '''
    
    def _evaluate_signal_detection(self, stat_type ="cash"):
        
        if stat_type == "cash":
            from gammapy.stats import cash
            stat_null = cash(
                self.data["counts"].data,
                self.data["npred_bkg"].data
            )
            stat_full = cash(
                self.data["counts"].data,
                self.data["npred_sig"].data + self.data["npred_bkg"].data
            )
        elif stat_type == "loglikely":
            pass

        empty_map = self.data["counts"].copy()
        empty_map.data = 0
        
        stat_null_map = empty_map.copy()
        stat_null_map.data = stat_null
        
        stat_full_map = empty_map.copy()
        stat_full_map.data = stat_full

        #ts_matrix = np.clip(stat_null - stat_full, 0, None)
        ts_map = empty_map.copy()
        ts_map.data = np.clip(stat_null - stat_full, 0, None)
        
        self.results["stat_null"] = stat_null_map
        self.results["stat_full"] = stat_full_map
        self.results["ts"] = ts_map

    def _split_galactic_and_off_plane(self, lat_threshold=10*u.deg):
        # Get pixel coordinates
        pixel_skycoords = self.map_dataset._geom.get_coord().skycoord
        if self.metadata["frame"] == "galactic":
            lon_cube_qarray = pixel_skycoords.l
            lat_cube_qarray = pixel_skycoords.b
        elif self.metadata["frame"] == "icrs":
            lon_cube_qarray = pixel_skycoords.ra
            lat_cube_qarray = pixel_skycoords.dec
        lon_map_qarray = lon_cube_qarray[0].flatten()
        lat_map_qarray = lat_cube_qarray[0].flatten()
        
        # Create masks for galactic plane and off-plane regions
        galactic_plane_array = np.abs(lat_map_qarray) <= lat_threshold
        off_plane_array = ~galactic_plane_array
        return galactic_plane_array, off_plane_array
        
    def extract_filtered_table(
        self,
        nphotons_min=10,
        ts_min=25,
        use_galactic_plane=True,
        use_off_plane=True,
        lat_threshold=10*u.deg
    ):
        from functools import reduce
        table = []
        
        # それぞれのピクセル位置
        lon_data = self.map_dataset._geom.get_coord().lon[0].flatten()
        lat_data = self.map_dataset._geom.get_coord().lat[0].flatten()
        if self.metadata["frame"] == "galactic":
            lon_key = "l"
            lat_key = "b"
        elif self.metadata["frame"] == "icrs":
            lon_key = "ra"
            lat_key = "dec"
        
        # galactic planeとoff planeに分ける
        galactic_plane_array, _ = \
            self._split_galactic_and_off_plane(lat_threshold=lat_threshold)
        
        # 各エネルギーウィンドウ毎に解析
        energy_edges = self.map_dataset.geoms["geom"].axes["energy"].edges
        
        org_counts = self.data["counts"]
        org_npred_sig = self.data["npred_sig"]
        org_npred_bkg = self.data["npred_bkg"]
        for eidx, emin in enumerate(energy_edges):
            if emin == energy_edges[-1]:
                break
            
            self.data["counts"] = org_counts\
                                  .slice_by_idx({"energy": slice(eidx, len(energy_edges))})\
                                  .sum_over_axes()
        

            self.data["npred_sig"] = org_npred_sig\
                                     .slice_by_idx({"energy": slice(eidx, len(energy_edges))})\
                                     .sum_over_axes()

            self.data["npred_bkg"] = org_npred_bkg\
                                     .slice_by_idx({"energy": slice(eidx, len(energy_edges))})\
                                     .sum_over_axes()


            self._evaluate_signal_detection()
        
            filterd_table = QTable({
                "plane": np.where(galactic_plane_array, "galactic", "off"),
                lon_key: lon_data,
                lat_key: lat_data,
                "obs_start": [self.metadata["obs_duration"][0]] * len(lon_data),
                "obs_stop": [self.metadata["obs_duration"][1]] * len(lon_data),
                "emin": ([round(energy_edges[eidx].value)] * len(lon_data)) * energy_edges[0].unit,
                "emax": ([round(energy_edges[len(energy_edges)-1].value)] * len(lon_data)) * energy_edges[len(energy_edges)-1].unit,
                "counts": self.data["counts"].data.flatten(),
                "npred_sig": self.data["npred_sig"].data.flatten(),
                "npred_bkg": self.data["npred_bkg"].data.flatten(),
                "ts": self.results["ts"].data.flatten(),
                "stat_full": self.results["stat_full"].data.flatten(),
                "stat_null": self.results["stat_null"].data.flatten()
            })

            filterd_table = filterd_table[
                (filterd_table["counts"] >= nphotons_min)
                & (filterd_table["ts"] >= ts_min)
            ]

            if use_galactic_plane and use_off_plane:
                pass
            elif use_galactic_plane and not use_off_plane:
                filterd_table = filterd_table[filterd_table["plane"] == "galactic"]
            elif not use_galactic_plane and use_off_plane:
                filterd_table = filterd_table[filterd_table["plane"] == "off"]
            elif not use_galactic_plane and not use_off_plane:
                raise ValueError("")
        
            table.append(filterd_table)

        self.data["counts"] = org_counts
        self.data["npred_sig"] = org_npred_sig
        self.data["npred_bkg"] = org_npred_bkg
        from astropy.table import vstack
        return vstack(table)
        
    def plot_stat_2d_histogram(self, stat_full):
        pass

    def plot_pixel_distribution(
        self,
        result_dict,
        hist_type="ObsCounts",
        spilit_plane=False,
        bins=100,
        log_scale=True,
        ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered", category=RuntimeWarning
            )
            
            if not spilit_plane:
                ax.hist(
                    result_dict[hist_type],
                    bins=bins,
                    histtype="step",
                    color="black",
                    log=log_scale,
                )
                ax.set(
                    xlabel=f"Counts per Pixel",
                    ylabel="Number of Pixels",
                    ylim=(9e-1, None) if log_scale else (0, None)
                )
                ax.grid(True)
                ax.set_title(hist_type)
            
            if spilit_plane:
                planes = ["galactic", "off"]
                for plane in planes:
                    use_plane_array = result_dict['Plane'] == plane
            
                    ax.hist(
                        np.where(use_plane_array, result_dict[hist_type], np.nan),
                        bins=bins,
                        histtype="step",
                        log=log_scale,
                        label=plane
                    )
                    ax.set(
                        xlabel=f"Counts per Pixel",
                        ylabel="Number of Pixels",
                        ylim=(9e-1, None) if log_scale else (0, None)
                    )
                    ax.grid(True)
                    ax.legend()
                    ax.set_title(hist_type)
        return ax

    def plot_pixel_map(
        self,
        result_dict,
        hist_type="ObsCounts",
        use_galactic_plane=False,
        use_off_plane=True,
        ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if not use_galactic_plane and not use_off_plane:
            raise ValueError("All data has been masked. At least one of use_galactic_plane and use_off_plane must be True.")
        elif use_galactic_plane and use_off_plane:
            cube_array = result_dict[hist_type]
        elif use_galactic_plane and not use_off_plane:
            use_gal_plane_array = result_dict['Plane'] == "galactic"
            cube_array = np.where(use_gal_plane_array, result_dict[hist_type], np.nan)
        elif not use_galactic_plane and use_off_plane:
            use_off_plane_array = result_dict['Plane'] == "off"
            cube_array = np.where(use_off_plane_array, result_dict[hist_type], np.nan)
    
        ccube = Map.from_geom(self.geom, dtype=float)
        ccube.data = cube_array
    
        vmax = np.nanmax(ccube.data[0])
        ccube.plot(
            ax=ax,
            stretch="log",
            cmap="jet",
            add_cbar=True,
            #vmin=0,
            #vmax= 1 if vmax == 0 else vmax
        )
        ax.set(
            xticks=[],
            yticks=[],
            #title="Distribution of pixel events passing the condition.\n(Mask Area (White), Passed Pixel (Yellow-Green), Failed Pixel (Black))"
        )
        return ax

    def plot_pixel_residual_map(
        self,
        result_dict,
        use_galactic_plane=False,
        use_off_plane=True,
        ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if not use_galactic_plane and not use_off_plane:
            raise ValueError("All data has been masked. At least one of use_galactic_plane and use_off_plane must be True.")
        elif use_galactic_plane and use_off_plane:
            cube_array = result_dict["ObsCounts"] - result_dict["BkgNpred"]
        elif use_galactic_plane and not use_off_plane:
            use_gal_plane_array = result_dict['Plane'] == "galactic"
            cube_array = np.where(use_gal_plane_array, result_dict["ObsCounts"] - result_dict["BkgNpred"], np.nan)
        elif not use_galactic_plane and use_off_plane:
            use_off_plane_array = result_dict['Plane'] == "off"
            cube_array = np.where(use_off_plane_array, result_dict["ObsCounts"] - result_dict["BkgNpred"], np.nan)
    
        ccube = Map.from_geom(self.geom, dtype=float)
        ccube.data = cube_array
    
        vmax = np.nanmax(ccube.data[0])
        ccube.plot(
            ax=ax,
            stretch="log",
            cmap="jet",
            add_cbar=True,
            #vmin=0,
            #vmax= 1 if vmax == 0 else vmax
        )
        ax.set(
            xticks=[],
            yticks=[],
            #title="Distribution of pixel events passing the condition.\n(Mask Area (White), Passed Pixel (Yellow-Green), Failed Pixel (Black))"
        )
        return ax

    def evaluate_count_distribution(self, ax=None, signal_models=None, **kwargs):
        # ===================================================
        # Prepare figure and axes, supporting external axes.
        # ===================================================
        if ax:
            fig = ax.figure
            bbox = ax.get_position()  # parent_axの位置を取得 (Figure基準)
            gs = GridSpec(
                2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig,
                left=bbox.x0, right=bbox.x0 + bbox.width,  # parent_axの横幅
                bottom=bbox.y0, top=bbox.y0 + bbox.height  # parent_axの縦幅
            )
        else:
            fig = plt.figure(figsize=(12, 6))
            gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)
        
        # split up within figs according to gs
        axes = {
            "Left": fig.add_subplot(gs[:, 0]),
            "Top Right": fig.add_subplot(gs[0, 1]),
            "Bottom Right": fig.add_subplot(gs[1, 1]),
        }
        
        # ===================================================================
        # Screening of count and npred maps according to specified criteria.
        # ===================================================================
        if signal_models:
            self.map_dataset.models = signal_models
            #self.map_dataset
        sig_npred_map = self.map_dataset.npred_signal()
        bkg_npred_map = self.map_dataset.npred_background()

        energy_window_kwargs = {key: kwargs[key] for key in ("energy_window",) if key in kwargs}
        gamma_min_kwargs = {key: kwargs[key] for key in ("gamma_min",) if key in kwargs}
        n_sigma_kwargs = {key: kwargs[key] for key in ("n_sigma",) if key in kwargs}
        
        # energy -> count -> galactic and off plane
        count_map = self.map_dataset.counts
        energy_reduced_count_map = self.filter_by_energy_window(count_map, **energy_window_kwargs)
        masked_count_map = self.filter_by_signal_event_count(energy_reduced_count_map, **gamma_min_kwargs)
        masked_count_map = self.filter_by_signal_to_noise_significance(
            masked_count_map,
            self.filter_by_energy_window(sig_npred_map, **energy_window_kwargs),
            self.filter_by_energy_window(bkg_npred_map, **energy_window_kwargs),
            **n_sigma_kwargs
        )
        
        # screening of `self.map_dataset.npred_signal()`
        npred_map = self.map_dataset.npred()
        energy_reduced_npred_map = self.filter_by_energy_window(npred_map, **energy_window_kwargs)
        masked_npred_map = self.filter_by_signal_event_count(energy_reduced_npred_map, **gamma_min_kwargs)
        masked_npred_map = self.filter_by_signal_to_noise_significance(
            masked_npred_map,
            self.filter_by_energy_window(sig_npred_map, **energy_window_kwargs),
            self.filter_by_energy_window(bkg_npred_map, **energy_window_kwargs),
            **n_sigma_kwargs
        )
        
        # ======================
        # Plotting the results.
        # ======================
        # Function to update the plots based on ON/OFF state
        current_model_mode = ["Counts"]  # mutable object to track state
        log_scale=False

        def get_plot_data(mode):
            if mode == "Counts":
                hist_data = masked_count_map.data.flatten()
                passed_data = np.where(np.isnan(masked_count_map), 0, 1)
                if isinstance(masked_count_map, WcsNDMap):
                    passed_map = WcsNDMap(masked_count_map.geom, data=passed_data)
                elif isinstance(masked_count_map, HpxNDMap):
                    passed_map = HpxNDMap(masked_count_map.geom, data=passed_data)
            else:  # mode == "Npred"
                hist_data = masked_npred_map.data.flatten()
                passed_data = np.where(np.isnan(masked_npred_map), 0, 1)
                if isinstance(masked_npred_map, WcsNDMap):
                    passed_map = WcsNDMap(masked_npred_map.geom, data=passed_data)
                elif isinstance(masked_npred_map, HpxNDMap):
                    passed_map = HpxNDMap(masked_npred_map.geom, data=passed_data)
            return hist_data, passed_map
        
        def update_plot(mode):
            for ax in axes.values():
                ax.clear()
            hist_data, passed_map = get_plot_data(mode)
            bins = np.arange(self.current_conditions["gamma_min"], np.nanmax(count_map) + 1)
            self._plot_pixel_count_distribution(hist_data, bins=bins, log_scale=log_scale, ax=axes["Left"])
            self._plot_passed_count_map(passed_map, ax=axes["Bottom Right"])
            fig.canvas.draw()
        
        # Event handler for ON/OFF button
        ax_button_onoff = fig.add_axes([0.75, 0.01, 0.1, 0.05])
        button_onoff = Button(ax_button_onoff, 'Counts/Npred')
        def on_onoff_clicked(event):
            modes = ["Counts", "Npred"]
            current_index = modes.index(current_model_mode[0])
            current_model_mode[0] = modes[(current_index + 1) % len(modes)]
            update_plot(current_model_mode[0])
        button_onoff.on_clicked(on_onoff_clicked)
        
        # Event handler for Log/Linear button
        left_bbox = axes["Left"].get_position()  # get axes["Left"] position
        button_width = 0.3 * left_bbox.width  # set the width of the button to 30% of the width of axes[‘Left’]
        button_height = 0.05 * left_bbox.height  # set the height of the button to 5% of the height of axes[‘Left’]
        button_x = left_bbox.x0 + 0.65 * left_bbox.width  # 65% from the left edge
        button_y = left_bbox.y0 + 0.02 * left_bbox.height  # slightly above the bottom edge
        ax_button_log = fig.add_axes([button_x, button_y, button_width, button_height])
        button_log = Button(ax_button_log, 'Log/Linear')
        def on_log_clicked(event):
            nonlocal log_scale
            log_scale = not log_scale
            update_plot(current_model_mode[0])
        button_log.on_clicked(on_log_clicked)
        
        # Initial plot
        update_plot(current_model_mode[0])
        plt.show()
    

    def plot_signal_model_map(self, width=Angle("10 deg")):
        """
        シグナルモデルを中心にしたマップデータセット
        """
        signal_skymodel_list = self._get_point_source_signal_skymodel_list_per_pixel(
            spatial_skycoords=SkyCoord(l=np.array([[0]])*u.deg,b=np.array([[0]])*u.deg,frame="galactic")
        )

        self.map_dataset.models = Models(signal_skymodel_list)
        
        sig_npred_ccube = self.map_dataset.npred_signal().sum_over_axes()
        sig_npred_ccube.plot(add_cbar="log")

    def get_signal_model_spectrum(
        self,
        radius=Angle("1 deg"),
        only_spectrum=False
    ):
        """
        on_center_coords=[SkyCoord(0, 0, unit="deg", frame="galactic")],
        は、データセットのsignalmodelsから取得する
        """
        if not self.map_dataset.models:
            return -1
            
        sigon_spectrum_datasets = []
        for sky_model in self.map_dataset.models:
            lon = sky_model.spatial_model.lon_0.quantity
            lat = sky_model.spatial_model.lat_0.quantity
            frame = sky_model.spatial_model.frame
            # fermiの場合にはgalacticに
            center = SkyCoord(lon, lat, frame=frame)
        
            sigon_region = CircleSkyRegion(center=center, radius=radius)
            sigon_spectrum_dataset = \
                self.map_dataset.to_spectrum_dataset(sigon_region)
            # 元々あったsignal_modelsとbackground_modelを適用
            sigon_spectrum_dataset.models = \
                self.map_dataset.models
                            
            # リストに追加
            sigon_spectrum_datasets.append(sigon_spectrum_dataset)
        return sigon_spectrum_datasets

































    def evaluate_signal_detection_using_tsmapestimator(self, skymodel_list, energy_window=None, **kwargs):
        from gammapy.estimators import TSMapEstimator

        # デフォルトのエネルギーウィンドウを設定
        energy_edges = self.map_dataset._geom.axes["energy"].edges
        energy_window = energy_window or (energy_edges[0], energy_edges[-1])
        logger.debug(f"Energy window set to: {energy_window}")
        # エネルギーウィンドウの最小値と最大値を取得
        energy_min, energy_max = map(u.Quantity, energy_window)
        idx_emin = np.argmin(np.abs(energy_edges - energy_min))
        idx_emax = np.argmin(np.abs(energy_edges - energy_max))
        # エネルギーウィンドウがエッジと一致するかを確認
        if not np.isclose(energy_edges[idx_emin], energy_min) or not np.isclose(energy_edges[idx_emax], energy_max):
            raise ValueError(
                f"Energy window ({energy_min}, {energy_max}) does not match any of the energy edges: {energy_edges}"
            )
        energy_edges = [energy_edges[idx_emin], energy_edges[idx_emax]]
            
        for skymodel in skymodel_list:
            estimator = TSMapEstimator(
                model=skymodel,
                kernel_width="1 deg",
                energy_edges=energy_edges,
                **kwargs
            )
            self.metadata["npred_sig_name"].append(skymodel.name)
            
            self.data["npred_sig"].append(estimator.run(self.map_dataset))
            
        # updata
        self.data["counts"] = self.map_dataset.counts\
                              .slice_by_idx({"energy": slice(idx_emin, idx_emax)})\
                              .sum_over_axes()
        for idx, npred_bkg in enumerate(self.data["npred_bkg"]):
            self.data["npred_bkg"][idx] = npred_bkg\
                                          .slice_by_idx({"energy": slice(idx_emin, idx_emax)})\
                                          .sum_over_axes()
        self.data["exposure"] = self.map_dataset.exposure\
                                .slice_by_idx({"energy": slice(idx_emin, idx_emax)})\
                                .sum_over_axes()
        self.data["psf"] = self.map_dataset.psf
        self.data["edisp"] = self.map_dataset.edisp





        
    def evaluate_ts_distribution(self, ax=None, energy_window=None, log_scale=True):
        if ax:
            fig = ax.figure
            bbox = ax.get_position()  # parent_axの位置を取得 (Figure基準)
            gs = GridSpec(
                2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig,
                left=bbox.x0, right=bbox.x0 + bbox.width,  # parent_axの横幅
                bottom=bbox.y0, top=bbox.y0 + bbox.height  # parent_axの縦幅
            )
        else:
            fig = plt.figure(figsize=(12, 6))
            gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)
        
        # split up within figs according to gs
        axes = {
            "Left": fig.add_subplot(gs[:, 0]),
            "Top Right": fig.add_subplot(gs[0, 1]),
            "Bottom Right": fig.add_subplot(gs[1, 1]),
        }
        
        self._reduce_by_energy_window(energy_window=energy_window) # 興味のあるエネルギー帯にする
        ts_array, _, _ = self._calculate_test_statistic() # tsデータを取得
        sqrt_ts_array = np.sqrt(ts_array)
        
        # tsの値分布
        import warnings
        with warnings.catch_warnings():
            #if logger
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            axes["Left"].hist(sqrt_ts_array.flatten(), bins="auto", histtype="step", color="black", log=log_scale)
            axes["Left"].set(
                xlabel="Counts per Pixel",
                ylabel="Number of Pixels",
                ylim=(9e-1, None) if log_scale else (0, None)
            )
            axes["Left"].grid(True)
        
        # テーブルのデータ
        table_data = [
            ["10", "15", "20"],
            ["5", "25", "30"],
            ["35", "10", "5"],
        ]
        column_labels = ["Column 1", "Column 2", "Column 3"]
        row_labels = ["Row 1", "Row 2", "Row 3"]
        self.plot_summary_table(
            data=table_data,
            column_labels=column_labels,
            row_labels=row_labels,
            ax=axes["Top Right"],
            title="Table with Row Labels"
        )
        
        # tsマップ
        reduced_geom = self.count_map.geom
        if self.map_type is WcsNDMap:
            ts_map = WcsNDMap(reduced_geom, data=sqrt_ts_array)
        elif self.map_type is HpxNDMap:
            ts_map = HpxNDMap(reduced_geom, data=sqrt_ts_array)
            print(ts_map)
        else:
            raise ValueError("Unsupported map type now. Expected WcsNDMap or HpxNDMap.")
        from matplotlib import cm
        cmap = cm.get_cmap("hot").copy()
        cmap.set_bad(color="gray", alpha=0.3)
        
        ts_map.plot(ax=axes["Bottom Right"], stretch="linear", add_cbar=True, cmap=cmap)
        axes["Bottom Right"].set(xticks=[], yticks=[], title="Sqrt(TS)")
        return axes
        
        
    def count_map_filtering(
        self,
        count_map,
        ewindow=None,
        mask_galactic_plane=True,
        mask_off_plane=False,
        **kwargs
    ):
        """
        Evaluates the counts distribution within the dataset.

        Parameters:
        -----------
        ewindow : tuple or None
            The energy range to consider (e.g., ("1 GeV", "1 TeV")).
            If None, defaults to the full range in the dataset.
        cwindow : tuple or None
            The range of counts per pixel to consider (e.g., (-np.inf, np.inf)).
            If None, defaults to the minimum and maximum counts in the dataset.
        mask_galactic_plane : bool
            Whether to include the galactic plane region.
        mask_off_plane : bool
            Whether to include the off-plane region.
        lat_threshold : float, optional
            Galactic latitude threshold in degrees to define the galactic plane. Default is 10°.
        """
        #============================#
        # エネルギーウィンドウに関する処理 #
        #============================#
        # Default ewindow based on the dataset if not provided
        if ewindow is None:
            energy_edges = count_map.geom.axes["energy"].edges
            ewindow = (energy_edges[0], energy_edges[-1])
            logger.debug(f"Default energy window set to: {ewindow}")
        else:
            ewindow = (u.Quantity(ewindow[0]), u.Quantity(ewindow[1]))
            logger.debug(f"Input energy window set to: {ewindow}")
    
        energy_min, energy_max = (u.Quantity(ewindow[0]), u.Quantity(ewindow[1]))
        energy_edges = count_map.geom.axes["energy"].edges
        idx_emin = np.where(np.isclose(energy_edges.to(u.MeV).value, energy_min.to(u.MeV).value))[0]
        idx_emax = np.where(np.isclose(energy_edges.to(u.MeV).value, energy_max.to(u.MeV).value))[0]
        if len(idx_emin) == 0 or len(idx_emax) == 0:
            raise ValueError(
                f"Energy window ({energy_min}, {energy_max}) do not match any of the energy edges: {energy_edges}"
            )
        ereduced_counts = \
            count_map.slice_by_idx({"energy": slice(int(idx_emin[0]), int(idx_emax[0]))}).sum_over_axes()

        #==========================#
        # カウントウィンドウに関する処理 #
        #==========================#
        condition_kwargs = {key: kwargs[key] for key in ("gamma_min", "n_sigma", "bkg_syst_fraction", "signal_npred_map", "background_npred_map") if key in kwargs}
        cmasked_counts, condition_passed_data = self.apply_signal_detection_conditions(count_map, **condition_kwargs)
        normed_cpassed_counts = np.where(np.isnan(cmasked_counts.data), 0, 1)
        
        #=======================#
        # RoIマスキングに関する処理 #
        #=======================#
        # 銀河面の領域 (|b| < 10°) と銀河面以外の領域 (|b| > 10°) に分けたマップを作成
        cmasked_gplane_counts, cmasked_oplane_counts = self.split_galactic_plane(
            cmasked_counts,
            kwargs["lat_threshold"],
            keep_geometry=True,
            mask_value=np.nan
        )
        normed_cpassed_gplane_counts, normed_cpassed_oplane_counts = self.split_galactic_plane(
            normed_cpassed_counts,
            kwargs["lat_threshold"],
            keep_geometry=True,
            mask_value=np.nan
        )

        if mask_galactic_plane and mask_off_plane:
            return ValueError("All data has been masked. At least one of mask_galactic_plane and mask_off_plane must be false.")
            
        elif mask_galactic_plane and not mask_off_plane:
            cmasked_roi_counts = cmasked_oplane_counts
            cpassed_roi_counts_dict = {
                "map": normed_cpassed_oplane_counts,
                "npixels": normed_cpassed_counts.data[0].size,
                "galactic plane region": {"flag":"not using", "npixels":np.isnan(normed_cpassed_oplane_counts.data[0]).sum(), "lat_band_width":lat_threshold},
                "off plane region": {"flag":"using", "npixels":np.isnan(normed_cpassed_gplane_counts.data[0]).sum()}
            }
        elif not mask_galactic_plane and mask_off_plane:
            cmasked_roi_counts = cmasked_gplane_counts
            cpassed_roi_counts_dict = {
                "map": normed_cpassed_gplane_counts,
                "npixels": normed_cpassed_counts.data[0].size,
                "galactic plane region": {"flag":"using", "npixels":np.isnan(normed_cpassed_oplane_counts.data[0]).sum(), "lat_band_width":lat_threshold},
                "off plane region": {"flag":"not using", "npixels":np.isnan(normed_cpassed_gplane_counts.data[0]).sum()}
            }
    
        else:
            cmasked_roi_counts = cmasked_counts
            cpassed_roi_counts_dict = {
                "map": normed_cpassed_counts,
                "npixels": normed_cpassed_counts.data[0].size,
                "galactic plane region": {"flag":"using", "npixels":np.isnan(normed_cpassed_oplane_counts.data[0]).sum(), "lat_band_width":lat_threshold},
                "off plane region": {"flag":"using", "npixels":np.isnan(normed_cpassed_gplane_counts.data[0]).sum()}
            }
            
        windows = {"energy":ewindow, "count":cwindow, "time":0, "space":0}
        filtered_counts = cmasked_roi_counts
        passed_counts_dict = cpassed_roi_counts_dict
        return windows, filtered_counts, passed_counts_dict
