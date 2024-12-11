from astropy import constants as c
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from astropy.table import QTable

from gammapy.maps import MapAxis, WcsNDMap, HpxNDMap
from gammapy.modeling.models import Models, SkyModel

class ObsOnOff:
    def __init__(self, obs):
        """
        campaignでもsnapでもどっちでもできるように
        """
        import copy
        self._obs_on = copy.deepcopy(obs)
        self._obs_off = copy.deepcopy(obs)

    def mask_from_catalog(
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
        
    def clear_masked_regions(self):
        pass
    
    def add_fix_bkg_models(self, bkg_map):
        self._obs_on.map_dataset.background = bkg_map
        print(self._obs_on.map_dataset)
        
        self._obs_off.map_dataset.background = bkg_map
        print(self._obs_off.map_dataset)
        
    def add_free_bkg_models(self, bkg_models):
        self._obs_on.map_dataset.background_model = self._obs.map_dataset.models + bkg_models
        print(self._map_dataset)

    def clear_bkg_models(self, names=None):
        if not names:
            self._obs_on.map_dataset.background = None
            self._obs_on.map_dataset.background_model = None
            
            self._obs_off.map_dataset.background = None
            self._obs_off.map_dataset.background_model = None

    def add_signal_models(self, sig_models):
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
            
            from gammapy.modeling.models import (
                Models,
                PointSpatialModel,
                PowerLawSpectralModel,
                SkyModel,
            )
            spatial_model = PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic")
            spectral_model = PowerLawSpectralModel(
                index=2.7, amplitude="5.8e-10 cm-2 s-1 TeV-1", reference="100 GeV"
            )

            source = SkyModel(
                spectral_model=spectral_model,
                spatial_model=spatial_model,
                name="source-gc",
            )
            sig_models = Models([source])
        """
        if not self._obs_on.map_dataset.models:
            org_dataset_sig_models = Models([])
        else:
            org_dataset_sig_models = Models([])
            for sky_model in self._obs_on.map_dataset.models[:]:
                org_dataset_sig_models.append(sky_model)
        
        self._obs_on.map_dataset.models = org_dataset_sig_models + sig_models
        print(self._obs_on.map_dataset)
        
    def clear_signal_models(self, names=None):
        if not names:
            self._obs_on.map_dataset.models = None
                
    def get_circle_region_spectra(
        self,
        radius=Angle("1 deg"),
    ):
        """
        on_center_coords=[SkyCoord(0, 0, unit="deg", frame="galactic")],
        は、データセットのsignalmodelsから取得する
        """
        on_region_spectra = []
        off_region_spectra = []
        for sky_model in self._obs_on.map_dataset.models:
            lon = sky_model.spatial_model.lon_0.quantity
            lat = sky_model.spatial_model.lat_0.quantity
            frame = sky_model.spatial_model.frame
            
            on_region = \
                CircleSkyRegion(
                    center=SkyCoord(lon, lat, frame=frame),
                    radius=radius
                )
            
            on_region_spectra.append(
                self._obs_on.map_dataset.to_spectrum_dataset(on_region)
            )
            off_region_spectra(
                self._obs_off.map_dataset.to_spectrum_dataset(off_region)
            )
        return on_region_spectra, off_region_spectra
    
    def plot_on_region(self):
        pass

    def evaluate_sn_ratio(self):
        pass
    def evaluate_npred_map(self):
        pass

    def evaluate_fake_counts_distribution(
        self,
        nfakes=1,
        ewindow=None,
        cwindow=None,
        mask_galactic_plane=False,
        mask_off_plane=False,
        lat_threshold=10*u.deg,
        adjust_pixel_size=False,
        new_binsz=None,
        new_nside=None,
        show=True,
    ):
        """
        
        """
        
        cmasked_roi_counts, cpassed_roi_counts = \
            self.evaluate_counts_distribution(show=False)
        
        for ifake in range(nfakes - 1):
    
            self._map_dataset.fake()
            
            cmasked_roi_counts, cpassed_roi_counts = \
                self.evaluate_counts_distribution(show=False)
    
            cumsum_cmasked_roi_counts += cmasked_roi_counts
            cumsum_cpassed_roi_counts += cpassed_roi_counts
        
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
                {"ax": "Top Right", "data": cumsum_cmasked_roi_counts/nfakes, "title": "", "stretch": "log"},
                {"ax": "Bottom Right", "data": cumsum_cpassed_roi_counts, "title": "Distribution of events through energy and counting windows", "stretch": "log"},
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
        
        else:
            return cmasked_roi_counts, cpassed_roi_counts

    def evaluate_residual_map(self):
        #self.evaluate_counts_mapとself.evaluate_npred_mapを使って、そのさをみる
        pass




    def fit_map():
        pass
    def fit_spectrum():
        pass
    def get_residual_map(self, test_models_name, simlation=False):
        """使う前にモデルの追加が必要"""
        pass
    def test_statistics(self, test_models_name):
        pass
