from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from pathlib import Path
from gammapy.maps import MapAxis, WcsNDMap, HpxNDMap
from gammapy.modeling.models import Models
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from astropy.table import QTable

from logging import getLogger, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)



class ObsSnapFermiLAT:
    def __init__(
        self,
        dataset,
        outdir=Path('.')
    ):
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
        if dataset.background is None:
            missing_attributes.append("background")
        if dataset.models is None:
            missing_attributes.append("models")
        if missing_attributes:
            raise ValueError(f"The following attributes are None: {', '.join(missing_attributes)}.")

        if isinstance(dataset.counts, type(dataset.exposure)) and isinstance(dataset.counts, (WcsNDMap, HpxNDMap)):
            # dataset.counts と dataset.exposure が同じ型（WcsNDMap または HpxNDMap）で一致している場合
            print(f"The input MapDataset is based on {type(dataset.exposure)}.")
        else:
            logger.warn(f"The input MapDataset is a mixture of {type(dataset.counts)} and {type(dataset.exposure)}.")
        
        dataset.background
        dataset.models
        
        self._dataset = dataset
        
        # 二つを合わせた辞書にして、キーを設定しておく
        #self._on_regions = []
        #self._on_spectrum_datasets = []
        
    def mask_regions_from_catalog(
        self,
        catalog_file,
        #マスクする半径,
        #スペーしゃるモデルがpointsourceの場合のみ,
    ):
        '''
        Parameters:
        ----------
        catalog_file : path or str
        '''
        pass
        
    def add_models_from_catalog(
        self,
        catalog_file,
        #モデルに入れる基準,
        #self変数で読み込んだカタログは保持しておく？
    ):
        '''
        Parameters:
        ----------
        catalog_file : path or str
        '''
        pass
        
    def add_bkg_models(
    ):
        """
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
        
        self._on_regions.append(on_region)
        self._on_spectrum_datasets.append(on_spectrum_dataset)
        
        
    def run_fitting(
        self,
        fit_models,
        energy_edges,
        mode="binned",
    ):
        #bkg models
        # gammapyでやるのではなく、fermitoolsでやった方がいいのでは？
        fit = Fit()
        result = fit.run(datasets=[dataset])

    def get_residual_map(self, test_models_name, simlation=False):
        pass

    def get_test_statistics(self, test_models_name):
        on_spectrum = self.on_spectrum_dataset.copy()
        on_datasets = Datasets()
        on_datasets.append(on_spectrum)
        on_datasets.models = test_models
        #  Check statistics
        #  logger.info('''Cash: {0}'''.format(
        #  cash(n_on=on_spectrum.counts,
        #  mu_on=on_spectrum.npred()).sum()))
        #  logger.info('stat_sum: {0}'.format(on_datasets.stat_sum()))
        return on_datasets.stat_sum()

    def get_counts_map(
        self,
        count_type="npred"
    ):
        # 集めるカウント領域の設定
        if not count_radius:
            count_roi = self._count_roi
        else:
            count_roi = CircleSkyRegion(
                                            center=self._blackhole.sky_coord(),
                                            radius=count_radius*u.deg
                                        )

        # 各エネルギーを任意のlifetimeで足し合わせたカウント数、マップの初期化
        if isinstance(self._map_geom, WcsGeom):
            sig_counts = WcsNDMap.create(
                                skydir=self._map_geom.center_skydir,
                                frame=self._map_geom.frame,
                                width=(self._map_geom.width[0][0].value,
                                        self._map_geom.width[1][0].value
                                        ),
                                proj=self._map_geom.projection,
                                binsz=self._map_geom.pixel_scales[0].deg,
                                axes=[self._energy_axis],
                                dtype=float,
                            )
            bkg_counts = sig_counts.copy()
        elif isinstance(self._map_geom, HpxGeom):
            pass

        if count_type == "counts":
            self.get_map_dataset()
        elif count_type == "npred":
            pass
        elif count_type == "fake":
            pass
            
    def mask_low_counts(self, map, counts_threshold=10, mask_value=np.nan):
        """
        `counts_threshold` より小さいピクセルをマスクする関数
        Args:
            map (np.ndarray or Map): ピクセルのカウントが格納されたマップデータ
            counts_threshold (float): カウントのしきい値

        Returns:
            masked map: しきい値より小さいピクセルをマスクしたマップ
        """
        counts = map.data
        masked_data = np.where(counts >= counts_threshold, counts, mask_value)

        # Create the counts maps for each region
        if isinstance(map, WcsNDMap):
            return WcsNDMap(map.geom, data=masked_data)
        elif isinstance(map, HpxNDMap):
            return HpxNDMap(map.geom, data=masked_data)
        else:
            raise ValueError("Unsupported map type. Expected WcsNDMap or HpxNDMap.")
            
    def split_galactic_plane(self, map, latitude_threshold=10*u.deg, keep_geometry=False, mask_value=np.nan):
        """
        Split a given map into galactic plane (|b| < latitude_threshold)
        and off-plane regions (|b| >= latitude_threshold). b is defined to latitude_threshold.

        Parameters
        ----------
        map : `~gammapy.maps.WcsNDMap` or `~gammapy.maps.HpxNDMap`
        
        latitude_threshold : float, optional
            Galactic latitude threshold in degrees to define the galactic plane.
            Default is 10°.

        keep_geometry : bool, optional
            If True, keeps the original geometry of the dataset. If False,
            a new geometry is created that corresponds to the split regions.
            Default is True.

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
        galactic_plane_mask = np.abs(pixel_coords["lat"]) < latitude_threshold.value
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

        return {"map": galactic_plane_map, "mask": galactic_plane_mask}, {"map": off_plane_map, "mask": off_plane_mask}

    def extract_pixel_info_as_qtable(self, map):
        """
        Extract pixel-wise information from a map and return it as an Astropy QTable.
        
        Parameters
        ----------
        map : `~gammapy.maps.Map`
            The input map object (WcsNDMap or HpxNDMap).
        
        Returns
        -------
        qtable : `~astropy.table.QTable`
            A QTable containing pixel-wise information: coordinates, energy (if available), and data values.
        """
        # Get pixel center coordinates
        pixel_coords = map.geom.get_coord()

        # Extract basic coordinate information
        lon = pixel_coords["lon"][0] * u.deg
        lat = pixel_coords["lat"][0] * u.deg
        data = {"lon": lon, "lat": lat}
        
        # Extract non-spatial axes information
        non_spatial_axes_name = map.geom.axes.names
        for axis_name in non_spatial_axes_name:
            for idx in range(map.geom.axes[axis_name].nbin):
                table_name = f"counts_{axis_name}{map.geom.axes[axis_name].edges[idx].value:.0e}_{map.geom.axes[axis_name].edges[idx+1].value:.0e}{str(map.geom.axes[axis_name].unit)}"
                data[table_name] = map.data[idx]
        
        return QTable(data)

    def evaluate_counts_distribution(
        self,
        counts_type="counts",
        nfakes=0,
        energy_window=("1 GeV", "1 TeV"),
        time_range=None,
        space_window=(),
        counts_threshold=0,
        latitude_threshold=10*u.deg,
        show_dist="counts"
    ):
        """
        Parameters
        ----------
        nfakes : int
            カウント分布の評価方法の指定
            - `-1` : モデル予測値 (`npred`) に基づく分布を評価
            - `0` : 実測データ (`real data`) を使用して分布を評価
            - `1以上` : 指定した数だけ疑似データ (`fake counts`) を生成し、それに基づく分布を評価
            デフォルト値は `0`
            
        energy_window : tuple of str or tuple of
            エネルギー範囲 (例: ("1 GeV", "10 GeV"))。astropy単位付き。

        show : bool
            `True` の場合、結果を可視化するためにプロットを表示
            デフォルト値は `False`

        Returns
        -------
        各ピクセルごとに集めたカウントマップデータのテーブル
        ピクセルの座標も一緒に記録しておきたい
        """
        energy_min, energy_max = (u.Quantity(energy_window[0]), u.Quantity(energy_window[1]))
        energy_edges = self._map_dataset.counts.geom.axes["energy"].edges
        idx_emin = np.where(np.isclose(energy_edges.to(u.MeV).value, energy_min.to(u.MeV).value))[0]
        idx_emax = np.where(np.isclose(energy_edges.to(u.MeV).value, energy_max.to(u.MeV).value))[0]
        if len(idx_emin) == 0 or len(idx_emax) == 0:
            raise ValueError(
                f"Energy window ({energy_min}, {energy_max}) do not match any of the energy edges: {energy_edges}"
            )
            
        reduced_dataset = \
            self._map_dataset.slice_by_idx({"energy": slice(int(idx_emin[0]), int(idx_emax[0]))})

        if nfakes==0:
            counts_map = reduced_dataset.counts.sum_over_axes()
            counts_map = self.mask_low_counts(
                            counts_map,
                            counts_threshold=counts_threshold,
                            mask_value=np.nan
                        )
            counts_data = counts_map.data.flatten()

            # 銀河面の領域 (|b| < 10°) と銀河面以外の領域 (|b| > 10°) を抽出
            galactic_plane, off_plane = self.split_galactic_plane(
                                            counts_map,
                                            latitude_threshold=latitude_threshold,
                                            keep_geometry=True,
                                            mask_value=np.nan
                                        )
            
            galactic_plane_counts_map = galactic_plane["map"]
            #Warning: mask_valueがnp.nanだとどっちでもいいけど、0にすると、上じゃないと0countsのエントリー数が増える
            #galactic_plane_counts_data = galactic_plane["map"].data[galactic_plane["mask"]].flatten()
            galactic_plane["map"].data.flatten()

            off_plane_counts_map = off_plane["map"]
            #off_plane_counts_data = off_plane["map"].data[off_plane["mask"]].flatten()
            off_plane["map"].data.flatten()

            counts_qtable = self.extract_pixel_info_as_qtable(counts_map)
            galactic_plane_counts_qtable = self.extract_pixel_info_as_qtable(galactic_plane_counts_map)
            off_plane_counts_qtable = self.extract_pixel_info_as_qtable(off_plane_counts_map)
            
            show_suptitle = f"Real MapDataset\n" \
                            f"(nside: {reduced_dataset.counts.geom.nside[0]}," \
                            f" energy: {round(reduced_dataset.counts.geom.axes['energy'].edges[0].to(u.GeV).value)} - " \
                            f"{round(reduced_dataset.counts.geom.axes['energy'].edges[-1].to(u.GeV).value)} GeV," \
                            f" time: {round(reduced_dataset.gti.met_stop[-1].value - reduced_dataset.gti.met_start[0].value)} sec)"
            
        elif nfakes>=1:
            counts_data_list = []
            galactic_plane_counts_data_list = []
            off_plane_counts_data_list = []
            
            counts_qtables = []
            galactic_plane_counts_qtables = []
            off_plane_counts_qtables = []
            
            for i in range(nfakes):
                # create fake counts
                reduced_dataset.fake()
                
                counts_map = reduced_dataset.counts.sum_over_axes()
                counts_map = self.mask_low_counts(
                                counts_map,
                                counts_threshold=counts_threshold,
                                mask_value=np.nan
                            )
                counts_data_list.append(
                    counts_map.data.flatten()
                )

                # 銀河面の領域 (|b| < 10°) と銀河面以外の領域 (|b| > 10°) を抽出
                galactic_plane, off_plane = self.split_galactic_plane(
                                                counts_map,
                                                latitude_threshold=latitude_threshold,
                                                keep_geometry=True,
                                                mask_value=np.nan
                                            )
                
                galactic_plane_counts_map = galactic_plane["map"]
                galactic_plane_counts_data_list.append(
                    #galactic_plane["map"].data[galactic_plane["mask"]].flatten()
                    galactic_plane["map"].data.flatten()
                )
                
                off_plane_counts_map = off_plane["map"]
                off_plane_counts_data_list.append(
                    #off_plane["map"].data[off_plane["mask"]].flatten()
                    off_plane["map"].data.flatten()
                )
    
                counts_qtables.append( self.extract_pixel_info_as_qtable(counts_map) )
                galactic_plane_counts_qtables.append( self.extract_pixel_info_as_qtable(galactic_plane_counts_map) )
                off_plane_counts_qtables.append( self.extract_pixel_info_as_qtable(off_plane_counts_map) )

            # list 内の全てのデータを1つに結合
            counts_data= np.concatenate([counts for counts in counts_data_list])
            galactic_plane_counts_data= np.concatenate([counts for counts in galactic_plane_counts_data_list])
            off_plane_counts_data= np.concatenate([counts for counts in off_plane_counts_data_list])
            
            show_suptitle = f"{nfakes} Fake MapDataset\n" \
                            f"(nside: {reduced_dataset.counts.geom.nside[0]}," \
                            f" energy: {round(reduced_dataset.counts.geom.axes['energy'].edges[0].to(u.GeV).value)} - " \
                            f"{round(reduced_dataset.counts.geom.axes['energy'].edges[-1].to(u.GeV).value)} GeV," \
                            f" time: {round(reduced_dataset.gti.met_stop[-1].value - reduced_dataset.gti.met_start[0].value)} sec)"
            show_tile = f"Total Pixels : {(np.size(reduced_dataset.counts.data) * nfakes):,}"

          
        if show_dist == "counts":
            fig = plt.figure(figsize=(15, 5))
            fig.suptitle(show_suptitle, fontsize=16)
            plot_hists = [
                {"data": counts_data, "color": "black", "log": True},
                {"data": galactic_plane_counts_data, "color": "black", "log": True},
                {"data": off_plane_counts_data, "color": "black", "log": True},
            ]
            inset_maps = [
                {"data": counts_map, "title": "All Sky Region", "stretch": "log"},
                {"data": galactic_plane_counts_map, "title": "Galactic Plane Region", "stretch": "log"},
                {"data": off_plane_counts_map, "title": "Off Plane Region", "stretch": "log"},
            ]
            
            for i, (plot, inset) in enumerate(zip(plot_hists, inset_maps)):
                ax = plt.subplot(1, 3, i+1)
                min_bin = np.floor(np.nanmin(plot["data"]))  # 最小値を切り下げ
                max_bin = np.ceil(np.nanmax(plot["data"]))  # 最大値を切り上げ
                bin_edges = np.arange(min_bin, max_bin + 1)
                ax.hist(plot["data"], bins=bin_edges, histtype="step", color=plot["color"], log=plot["log"])
                ax.set_xlabel("Counts per Pixel")
                ax.set_ylabel("Number of Pixels")
                ax.grid(True)
                
                ax_inset = inset_axes(ax, width="50%", height="50%", loc="upper right")
                inset["data"].plot(ax=ax_inset, stretch=inset["stretch"])
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                ax_inset.set_title(inset["title"])
            plt.show()
        
        elif show_dist == "pixels":
            return counts_map
            '''
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for qtable in off_plane_counts_qtables:
                non_nan_indices = ~np.isnan(qtable[qtable.colnames[2]])
                # NaNではない行に対応する 'lon' と 'lat' の値を取得
                lon = qtable['lon'][non_nan_indices].value
                lon = np.where(lon > 180, lon - 360, lon)
                lat = qtable['lat'][non_nan_indices].value
                ax.scatter(lon, lat, color='red', s=30)
    
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel('Galactic Longitude (deg)')
            ax.set_ylabel('Galactic Latitude (deg)')
            ax.set_title('Locations on the Celestial Sphere')
            ax.grid(True)
            plt.show()
            '''
    def evaluate_5sigma_events(self):
        #TODO: FARの計算、各ピクセルごとにヒストグラムの結果に対して何σのイベントだったかを計算、各ピクセルのカウント数を計算
        #二つのconditionを通過したピクセルを調べる。
        pass
    
    def evaluate_sn_ratio(self, show=False):
        for t_to_evap in self._times_to_evap[tmin_idx:tmax_idx]:
            t_to_evap_str = '{0:0=11}ms'.format(int(t_to_evap*1000))
            if count_type == 'counts':
                path_to_sig_filename = f"{self.RESULT_DIR}/simulation/{t_to_evap_str}/sig_count_map.fits.gz"
                path_to_bkg_filename = f"{self.RESULT_DIR}/simulation/{t_to_evap_str}/bkg_count_map.fits.gz"
            elif count_type == 'npred':
                path_to_sig_filename = f"{self.RESULT_DIR}/simulation/{t_to_evap_str}/sig_npred_map.fits.gz"
                path_to_bkg_filename = f"{self.RESULT_DIR}/simulation/{t_to_evap_str}/bkg_npred_map.fits.gz"
            sig_count_map += self.load_map(path_to_sig_filename).interp_to_geom(geom)
            bkg_count_map += self.load_map(path_to_bkg_filename).interp_to_geom(geom)


        # 各エネルギーを任意のlifetimeで足し合わせたカウント数
        diff_energy_sig_counts = sig_count_map.to_region_nd_map(region=count_roi).data[:,0,0]
        diff_energy_bkg_counts = bkg_count_map.to_region_nd_map(region=count_roi).data[:,0,0]

        int_hienergy_sig_counts = np.flip(np.cumsum(np.flip(diff_energy_sig_counts)))
        int_hienergy_bkg_counts = np.flip(np.cumsum(np.flip(diff_energy_bkg_counts)))
        
        diff_eSN = diff_energy_sig_counts / np.sqrt(diff_energy_bkg_counts)
        int_hi_esn = int_hienergy_sig_counts / np.sqrt(int_hienergy_bkg_counts)
        #return photon_count_table
