from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from pathlib import Path
from gammapy.maps import MapAxis, WcsNDMap, HpxNDMap
from gammapy.modeling.models import Models
import numpy as np
from astropy.table import QTable

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

class ObsSnapFermiLAT:
    def __init__(self, dataset, outdir=Path('.')):
        """
        ObsSnapFermiLAT holds information on a short-term observation for Fermi-LAT Telescope.
        Here "short-term" means the IRF can be considered to be constant.'
        
        Parameters:
        ----------
        dataset : `~ gammapy.datasets.MapDataset`

        outdir : string
        """
        missing_attributes = []
        if dataset.counts is None:
            missing_attributes.append("counts")
        if dataset.exposure is None:
            missing_attributes.append("exposure")
        if dataset.psf is None:
            missing_attributes.append("psf")
        if dataset.edisp is None:
            missing_attributes.append("edisp")
        #if dataset.background is None:
        #    missing_attributes.append("background")
        #if dataset.models is None:
        #    missing_attributes.append("models")
        if missing_attributes:
            raise ValueError(f"The following attributes are None: {', '.join(missing_attributes)}.")

        if isinstance(dataset.counts, type(dataset.exposure)) and isinstance(dataset.counts, (WcsNDMap, HpxNDMap)):
            # dataset.counts と dataset.exposure が同じ型（WcsNDMap または HpxNDMap）で一致している場合
            print(f"The input MapDataset is based on {type(dataset.exposure)}.")
        else:
            logger.warn(f"The input MapDataset is a mixture of {type(dataset.counts)} and {type(dataset.exposure)}.")
        
        #dataset.background
        #fdataset.models
        self._mapdataset = dataset
        
    def regions_masking(
        self,
        regions=[],
        #スペーしゃるモデルがpointsourceの場合のみ,
    ):
        '''
        Parameters:
        ----------
        regions : list
        
        catalog_file : path or str
        '''
        pass
        
        
    def add_bkg_models(
    ):
        """
        Parameters:
        ----------
        models : modelsかmapもいける。（マップの場合はnpredが固定されることにちゅうい）

        Returns:
            masked map: しきい値より小さいピクセルをマスクしたマップ
        """
        pass

    def add_signal_models(
        self,
        signal_models = Models(),
        on_center_coords = SkyCoord(0, 45, unit="deg", frame="galactic"),
        on_region_radius = Angle("1 deg"),
    ):
        """
        +-- Map RoI --------------+
        |                         |
        |        Count RoI        |
        |      ###       ###      |
        |    ##             ##    |
        |   ##      * <- BH  ##   |
        |    ##             ##    |
        |      ###       ###      |
        |         #######         |
        |                         |
        +-------------------------+
        
        Returns:
            masked map: しきい値より小さいピクセルをマスクしたマップ
        """
        self._obs_map_dataset.models = target_models
        
        on_region = CircleSkyRegion(
            center=on_center_coords,
            radius=on_region_radius
        )
        on_spectrum_dataset = \
            self._obs_map_dataset.to_spectrum_dataset(on_region)
        
    def split_galactic_plane(self, map, lat_threshold, keep_geometry=True, mask_value=np.nan):
        """
        Split a given map into galactic plane (|b| < latitude_threshold)
        and off-plane regions (|b| >= latitude_threshold). b is defined to latitude_threshold.

        Parameters
        ----------
        map : `~gammapy.maps.WcsNDMap` or `~gammapy.maps.HpxNDMap`
        
        lat_threshold : float, optional
            Galactic latitude threshold in degrees to define the galactic plane.
            Default is 10°.

        keep_geometry : bool, optional
            If True, keeps the original geometry of the self._dataset. So masking. If False,
            a new geometry is created that corresponds to the split regions.
            Default is True.
            
        mask_value :

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'galactic_plane_counts_map': Counts map for galactic plane region.
            - 'off_plane_counts_map': Counts map for off-plane region.
            - 'galactic_plane_counts_data': Data array for galactic plane region.
            - 'off_plane_counts_data': Data array for off-plane region.
        """
        # Get pixel coordinates
        pixel_coords = map.geom.get_coord()
        
        # Create masks for galactic plane and off-plane regions
        galactic_plane_mask = np.abs(pixel_coords["lat"]) < lat_threshold.value
        off_plane_mask = ~galactic_plane_mask

        # Create the counts maps for each region
        if isinstance(map, WcsNDMap):
            map_cls = WcsNDMap
        elif isinstance(map, HpxNDMap):
            map_cls = HpxNDMap
        else:
            raise ValueError("Unsupported map type. Expected WcsNDMap or HpxNDMap.")

        # Apply mask and create maps
        if keep_geometry:
            galactic_plane_map = map_cls(map.geom, data=np.where(galactic_plane_mask, map.data, mask_value))
            off_plane_map = map_cls(map.geom, data=np.where(off_plane_mask, map.data, mask_value))
        else:
            galactic_plane_map = map_cls(map.geom.select_bands(mask=galactic_plane_mask),
                                                       data=np.where(galactic_plane_mask, map.data, mask_value))
            off_plane_map = map_cls(map.geom.select_bands(mask=off_plane_mask),
                                                   data=np.where(off_plane_mask, map.data, mask_value))

        return galactic_plane_map, off_plane_map

    def evaluate_counts_distribution(
        self,
        ewindow=None,
        cwindow=None,
        mask_galactic_plane=False,
        mask_off_plane=False,
        lat_threshold=10*u.deg,
        adjust_pixel_size=False,
        new_binsz=None,
        new_nside=None,
    ):
        """
        Evaluates the counts distribution within the dataset.

        Parameters:
        -----------
        energy_window : tuple or None
            The energy range to consider (e.g., ("1 GeV", "1 TeV")).
            If None, defaults to the full range in the dataset.
        count_window : tuple or None
            The range of counts per pixel to consider (e.g., (-np.inf, np.inf)).
            If None, defaults to the minimum and maximum counts in the dataset.
        galactic_plane : bool
            Whether to include the galactic plane region.
        off_plane : bool
            Whether to include the off-plane region.
        adjust_pixel_size : bool
            Whether to adjust the pixel size of the map.
        binsz : float or None
            The new bin size, if adjusting the pixel size.
        nside : int or None
            The HEALPix Nside parameter, if using HEALPix geometry.
        """
        import inspect
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Method `{method_name}` started.")
        
        #============================#
        # エネルギーウィンドウに関する処理 #
        #============================#
        # Default ewindow based on the dataset if not provided
        if ewindow is None:
            energy_edges = self._mapdataset.counts.geom.axes["energy"].edges
            ewindow = (energy_edges[0], energy_edges[-1])
            print(f"Default energy window set to: {ewindow}")
        else:
            print(f"Input energy window set to: {ewindow}")
    
        energy_min, energy_max = (u.Quantity(ewindow[0]), u.Quantity(ewindow[1]))
        energy_edges = self._mapdataset.counts.geom.axes["energy"].edges
        idx_emin = np.where(np.isclose(energy_edges.to(u.MeV).value, energy_min.to(u.MeV).value))[0]
        idx_emax = np.where(np.isclose(energy_edges.to(u.MeV).value, energy_max.to(u.MeV).value))[0]
        if len(idx_emin) == 0 or len(idx_emax) == 0:
            raise ValueError(
                f"Energy window ({energy_min}, {energy_max}) do not match any of the energy edges: {energy_edges}"
            )
        ereduced_counts = \
            self._mapdataset.slice_by_idx({"energy": slice(int(idx_emin[0]), int(idx_emax[0]))}).counts.sum_over_axes()

        #==========================#
        # カウントウィンドウに関する処理 #
        #==========================#
        # Default cwindow based on the dataset if not provided
        if cwindow is None:
            counts_data = self._mapdataset.counts.data
            cwindow = (counts_data.min(), counts_data.max())
            print(f"Default count window set to: {cwindow}")
        else:
            print(f"Input count window set to: {cwindow}")
        
        masked_count_window_data = \
            np.where((ereduced_counts.data >= cwindow[0]) & (ereduced_counts.data <= cwindow[1]), ereduced_counts.data, np.nan)
        passed_count_window_data = \
            np.where((ereduced_counts.data >= cwindow[0]) & (ereduced_counts.data <= cwindow[1]), 1, 0)
        
        if isinstance(ereduced_counts, WcsNDMap):
            cmasked_counts = WcsNDMap(ereduced_counts.geom, data=masked_count_window_data)
            cpassed_counts = WcsNDMap(ereduced_counts.geom, data=passed_count_window_data)
        elif isinstance(ereduced_counts, HpxNDMap):
            cmasked_counts = HpxNDMap(ereduced_counts.geom, data=masked_count_window_data)
            cpassed_counts = HpxNDMap(ereduced_counts.geom, data=passed_count_window_data)
        else:
            raise ValueError("Unsupported map type now. Expected WcsNDMap or HpxNDMap.")
        
        #=======================#
        # RoIマスキングに関する処理 #
        #=======================#
        if mask_galactic_plane or mask_off_plane:
            
            if mask_galactic_plane and mask_off_plane:
                return ValueError("All data has been masked.")
                
            # 銀河面の領域 (|b| < 10°) と銀河面以外の領域 (|b| > 10°) に分けたマップを作成
            cmasked_gplane_counts, cmasked_oplane_counts = self.split_galactic_plane(
                cmasked_counts,
                lat_threshold,
                keep_geometry=True,
                mask_value=np.nan
            )
            
            cpassed_gplane_counts, cpassed_oplane_counts = self.split_galactic_plane(
                cpassed_counts,
                lat_threshold,
                keep_geometry=True,
                mask_value=np.nan
            )
            
            if mask_galactic_plane and not mask_off_plane:
                cmasked_roi_counts = cmasked_oplane_counts
                cpassed_roi_counts = cpassed_oplane_counts
                
            if not mask_galactic_plane and mask_off_plane:
                cmasked_roi_counts = cmasked_gplane_counts
                cpassed_roi_counts = cpassed_gplane_counts
        
        else:
            cmasked_roi_counts = cmasked_counts
            cpassed_roi_counts = cpassed_counts

        #=======================#
        # ピクセルサイズに関する処理 #
        #=======================#
        if adjust_pixel_size:
            if isinstance(cmasked_roi_counts, WcsNDMap):
                # map.geom.to_binszを使う
                # 元のマップをロードまたは作成original_map = WcsNDMap.create(binsz=0.1, width=(5, 5))
                # 元のジオメトリを取得original_geom = original_map.geom
                # 新しい binsz を指定してジオメトリを作成new_binsz = 0.2  # 新しいビンサイズnew_geom = original_geom.to_binsz(new_binsz)
                # 新しいマップを作成new_map = WcsNDMap.from_geom(new_geom)
                # 元のデータを新しいマップにリサンプリングする例（必要に応じて）resampled_map = original_map.interp_to_geom(new_geom)
                pass
            elif isinstance(cmasked_roi_counts, HpxNDMap):
                # Upsample or downsample the map to a given nside.
                cmasked_roi_counts = cmasked_roi_counts.to_nside(nside, preserve_counts=True)
                cpassed_roi_counts = cpassed_roi_counts.to_nside(nside, preserve_counts=True)
          
        #=========#
        # プロット #
        #=========#
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)

        axes = {
            "Left": fig.add_subplot(gs[:, 0]),
            "Top Right": fig.add_subplot(gs[0, 1]),
            "Bottom Right": fig.add_subplot(gs[1, 1])
        }

        plot_hists = [
            {"ax": "Left", "data": cmasked_roi_counts.data.flatten(), "color": "black", "log": False},
        ]
        plot_maps = [
            {"ax": "Top Right", "data": cmasked_roi_counts, "title": "", "stretch": "log"},
            {"ax": "Bottom Right", "data": cpassed_roi_counts, "title": "Distribution of events through energy and counting windows", "stretch": "log"},
        ]
        
        for hist in plot_hists:
            ax = axes[hist["ax"]]
            ax.hist(hist["data"], bins="auto", histtype="step", color=hist["color"], log=hist["log"])
            ax.set_xlabel("Counts per Pixel")
            ax.set_ylabel("Number of Pixels")
            ax.grid(True)

        for map in plot_maps:
            ax_map = axes[map["ax"]]
        
            # プロットの際には、np.nanだとマップ外側の領域と同じ色になって見えにくいのでnp.nanを0に変更してプロットする
            # Bug: galacticとかでのマスクまで見えてしまう。
            if map["ax"] == "Top Right":
                top_right_map = map["data"].copy()
                top_right_map.data = np.nan_to_num(masked_count_window_data, nan=0)
                map["data"] = top_right_map
            
            map["data"].plot(ax=ax_map, stretch=map["stretch"], add_cbar=True)
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            ax_map.set_title(map["title"])

        plt.tight_layout()
        plt.show()


    def fit_map():
        pass
    def fit_spectrum():
        pass
    def get_residual_map(self, test_models_name, simlation=False):
        """使う前にモデルの追加が必要"""
        pass
    def test_statistics(self, test_models_name):
        pass
