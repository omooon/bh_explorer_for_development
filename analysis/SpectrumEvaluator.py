

class ObsModeling:
    def __inti__(self, obs_snap):
        self._obs_snap = obs_snap
        
    def add_bkg_models():
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
        signal_models=Models(),
        on_center_coords=SkyCoord(0, 45, unit="deg", frame="galactic"),
        on_region_radius=Angle("1 deg"),
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
        self._obs_snap.map_dataset.models = target_models
        
        on_region = CircleSkyRegion(
            center=on_center_coords,
            radius=on_region_radius
        )
        on_spectrum_dataset = \
            self._obs_map_dataset.to_spectrum_dataset(on_region)

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

    def evaluate_npred_map(self):
        pass
        
    def evaluate_residual_map(self):
        #self.evaluate_counts_mapとself.evaluate_npred_mapを使って、そのさをみる
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





    # 質量（光度）、距離、観測時間にオケる関係
    #観測時間を固定した、横軸距離、縦軸質量の2Dマップ、観測時間は横軸スライダーでエネルギー窓は縦軸スライダーで
    #2Dのグリッドごとにsensitivityを計算して、任意のエネルギー窓で積分
    
    
    # 時間窓、エネルギー窓ごとのSN比
