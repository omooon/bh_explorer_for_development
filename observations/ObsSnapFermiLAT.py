from astropy import constants as c
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from astropy.table import QTable

from pathlib import Path
from gammapy.maps import MapAxis, WcsNDMap, HpxNDMap
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
        if dataset.gti is None:
            missing_attributes.append("gti")
        if missing_attributes:
            raise ValueError(f"The following attributes are None: {', '.join(missing_attributes)}.")

        invalid_attributes = []
        if dataset.background:
            invalid_attributes.append("background")
        if dataset.models:
            invalid_attributes.append("models")
        if invalid_attributes:
            raise ValueError(f"The following attributes cannot be included: {', '.join(invalid_attributes)}")

        # dataset.counts と dataset.exposure が同じ型（WcsNDMap または HpxNDMap）かどうかの確認
        if isinstance(dataset.counts, type(dataset.exposure)) and isinstance(dataset.counts, (WcsNDMap, HpxNDMap)):
            logger.info(f"The input MapDataset is based on {type(dataset.exposure)}.")
        else:
            logger.warn(f"The input MapDataset is a mixture of {type(dataset.counts)} and {type(dataset.exposure)}.")
        
        self._map_dataset = dataset
        
    @property
    def map_dataset(self):
        return self._map_dataset
    
    # エネルギー軸を可変する関数
    # ピクセルサイズを可変する関数
    # 時間窓でフィルタリングして可変する関数
        
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
        mask_galactic_plane=True,
        mask_off_plane=False,
        lat_threshold=10*u.deg,
        adjust_pixel_size=False,
        new_binsz=None,
        new_nside=None,
        show=True,
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
            energy_edges = self._map_dataset.counts.geom.axes["energy"].edges
            ewindow = (energy_edges[0], energy_edges[-1])
            print(f"Default energy window set to: {ewindow}")
        else:
            print(f"Input energy window set to: {ewindow}")
    
        energy_min, energy_max = (u.Quantity(ewindow[0]), u.Quantity(ewindow[1]))
        energy_edges = self._map_dataset.counts.geom.axes["energy"].edges
        idx_emin = np.where(np.isclose(energy_edges.to(u.MeV).value, energy_min.to(u.MeV).value))[0]
        idx_emax = np.where(np.isclose(energy_edges.to(u.MeV).value, energy_max.to(u.MeV).value))[0]
        if len(idx_emin) == 0 or len(idx_emax) == 0:
            raise ValueError(
                f"Energy window ({energy_min}, {energy_max}) do not match any of the energy edges: {energy_edges}"
            )
        ereduced_counts = \
            self._map_dataset.slice_by_idx({"energy": slice(int(idx_emin[0]), int(idx_emax[0]))}).counts.sum_over_axes()

        #==========================#
        # カウントウィンドウに関する処理 #
        #==========================#
        # Default cwindow based on the dataset if not provided
        if cwindow is None:
            counts_data = self._map_dataset.counts.data
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
                cmasked_roi_counts = cmasked_roi_counts.to_nside(new_nside, preserve_counts=True)
                cpassed_roi_counts = cpassed_roi_counts.to_nside(new_nside, preserve_counts=True)
          
        #=========#
        # プロット #
        #=========#
        if show:
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
                map["data"].plot(ax=ax_map, stretch=map["stretch"], add_cbar=True)
                ax_map.set_xticks([])
                ax_map.set_yticks([])
                ax_map.set_title(map["title"])

            plt.tight_layout()
            plt.show()
        
        else:
            return cmasked_roi_counts, cpassed_roi_counts


class ObsCampaignFermiLAT:
    '''ObsCampaign stands for a project to derive a specific result
    from a series of ObsSnap.
    It holds information on the target objects
    and the ObsSnap series.
    The IRF is assumed to be different for each ObsSnap.
    ObsCampaign can be composed of multiple observation
    runs toward one or more celestical regions.
    '''
    def __init__(
            self,
            obs_snaps={},
            outdir=Path('.'),
            reference_time=Time("2000-01-01 00:00:00")
        ):
        """Initialize an ObsCampaign instance.

        Parameters:
        ----------
        obs_snaps : dict, default={}
            A dictionary of ObsSnap instances to be analysed in the campaign.
            The key must be the time starting of the ObsSnap.
        """
        self.obs_snaps = obs_snaps
        self.reference_time = reference_time

    def timeres_onoffspectra(self):
        timeres_onoffspectra = []
        for obs_snap in self.obs_snaps.values():
            timeres_onoffspectra.append(
                obs_snap.onoff_spectrum_dataset
            )
        return timeres_onoffspectra

    def timeres_cubes(self):
        timeres_cubes = []
        for obs_snap in self.obs_snaps.values():
            timeres_cubes.append(
                obs_snap.map_dataset.counts
            )
        return timeres_cubes



    def list_obs_snaps(self):
        """Returns a list of ObsSnap objects.

        Returns:
        -------
        obs_snaps: list
            A list of ObsSnap objects.
        """
        return list(self.obs_snaps.values())

    def on_regions(self):
        """Returns a list of on regions for each observation.

        Returns:
        -------
        on_regions: list
            A list of on regions for each observation.
        """
        on_regions = []
        for obs_snap in self.obs_snaps.values():
            on_regions.append(obs_snap.oncount_geom.region)
        return on_regions
