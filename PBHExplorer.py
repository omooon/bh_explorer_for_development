import copy
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider
from itertools import product
from tqdm import tqdm

from astropy import constants as c
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, Angle
from astropy.table import QTable

from gammapy.data import GTI
from gammapy.maps import MapAxis, WcsNDMap, HpxNDMap
from gammapy.modeling.models import Models, SkyModel
        
from pbh_explorer.telescopes.fermi_lat import FermiLATObservation #, CTAOObservation
from pbh_explorer.observations import ObsSnap, ObsCampaign
from pbh_explorer.objects import AstroObject
from concurrent.futures import ProcessPoolExecutor

class PBHExplorer:
    def __init__(
        self,
        files=[],
        telescope="fermi_lat",
        map_type="hpx",
        insert_wcs_dataset=True,
        loglevel="DEBUG"
    ):
        # 事前計算済みのマップデータセットの用意
        if telescope == "fermi_lat":
            self.fermi = FermiLATObservation.AllSkyRegion()
            self.fermi.create_map_dataset()
        
        # ObsSnapインスタンス
        if map_type == "wcs":
            obs_snap = ObsSnap.FermiLAT(
                self.fermi.wcs_dataset,
                loglevel=loglevel
            )
        elif map_type == "hpx":
            obs_snap = ObsSnap.FermiLAT(
                self.fermi.hpx_dataset,
                loglevel=loglevel
            )

        # ObsCampaignインスタンスでまとめる
        self.obs_campaign = ObsCampaign.FermiLAT(
            obs_snaps=[obs_snap],
            loglevel=loglevel
        )
        for obs_snap in self.obs_campaign.snaps:
            if map_type == "hpx" and insert_wcs_dataset is True:
                self.setup_obs_snap(
                    obs_snap,
                    method_kwargs={
                        "evaluate_diffuse_background": {
                            #"galdiff_filename": "path/to/galdiff.fits",
                            #"isodiff_filename": "path/to/isodiff.txt",
                            "wcs_map_dataset": self.fermi.wcs_dataset}
                    }
                )
            else:
                self.setup_obs_snap(
                    obs_snap,
                    method_kwargs={
                        "evaluate_diffuse_background": {"wcs_map_dataset": None}
                    }
                )
            
    def setup_obs_snap(self, obs_snap, method_kwargs={}):
        # method引数で指定した関数を動的に処理
        valid_methods = {
            "evaluate_ps_signal": obs_snap.evaluate_ps_signal,
            "evaluate_ps_background": obs_snap.evaluate_ps_background,
            "evaluate_diffuse_background": obs_snap.evaluate_diffuse_background,
        }

        for method, kwargs in method_kwargs.items():
            if method not in valid_methods:
                raise ValueError(f"Invalid method '{method}'. Valid methods are: {list(valid_methods.keys())}")
            func = valid_methods[method]
            print(f"Executing {method}...")
            func(**kwargs)

    def extract_filtered_table(
        self,
        nphotons_min=10,
        ts_min=25,
        energy_min=1*u.GeV,
        use_galactic_plane=True,
        use_off_plane=True,
        lat_threshold=10*u.deg
    ):
        table = []
        for obs_snap in self.obs_campaign.snaps:
            # 各ピクセル位置を取得
            lon_data = obs_snap.data["counts"].geom.get_coord().lon[0].flatten()
            lat_data = obs_snap.data["counts"].geom.get_coord().lat[0].flatten()
            if obs_snap.metadata["frame"] == "galactic":
                lon_key = "l"
                lat_key = "b"
            elif obs_snap.metadata["frame"] == "icrs":
                lon_key = "ra"
                lat_key = "dec"
            
            # galactic planeとoff planeに分ける
            def split_galactic_and_off_plane(lat_threshold=10*u.deg):
                # Get pixel coordinates
                pixel_skycoords = obs_snap.data["counts"].geom.get_coord().skycoord
                if obs_snap.metadata["frame"] == "galactic":
                    lon_cube_qarray = pixel_skycoords.l
                    lat_cube_qarray = pixel_skycoords.b
                elif obs_snap.metadata["frame"] == "icrs":
                    lon_cube_qarray = pixel_skycoords.ra
                    lat_cube_qarray = pixel_skycoords.dec
                lon_map_qarray = lon_cube_qarray[0].flatten()
                lat_map_qarray = lat_cube_qarray[0].flatten()
                
                # Create masks for galactic plane and off-plane regions
                galactic_plane_array = np.abs(lat_map_qarray) <= lat_threshold
                off_plane_array = ~galactic_plane_array
                return galactic_plane_array, off_plane_array
            galactic_plane_array, _ = split_galactic_and_off_plane(lat_threshold=lat_threshold)
            
            # エネルギーウィンドウの指定
            energy_edges = obs_snap.data["counts"].geom.axes["energy"].edges
            eidx_min = np.argmin(np.abs(energy_edges - energy_min))
            
            # 解析するmap
            counts_map = (
                obs_snap.data["counts"]
                .slice_by_idx({"energy": slice(eidx_min, len(energy_edges))})
                .sum_over_axes()
                .copy()
            )
            if obs_snap.data["npred_sig"] is None:
                npred_sig_map = counts_map.copy()
            else:
                npred_sig_map = (
                    obs_snap.data["npred_sig"]
                    .slice_by_idx({"energy": slice(eidx_min, len(energy_edges))})
                    .sum_over_axes()
                    .copy()
                )
            npred_bkg_map = (
                (obs_snap.data["npred_bkg"])
                .slice_by_idx({"energy": slice(eidx_min, len(energy_edges))})
                .sum_over_axes()
                .copy()
            )
            
            # ts検定
            def evaluate_test_statistics(
                counts_map, npred_sig_map, npred_bkg_map, stat_type="cash"
            ):
                if stat_type == "cash":
                    from gammapy.stats import cash
                    stat_null = cash(
                        counts_map.data,
                        npred_bkg_map.data
                    )
                    stat_full = cash(
                        counts_map.data,
                        npred_sig_map.data + npred_bkg_map.data
                    )
                elif stat_type == "loglikely":
                    pass
                
                stat_null_map = counts_map.copy()
                stat_null_map.data = stat_null
                
                stat_full_map = counts_map.copy()
                stat_full_map.data = stat_full
                
                ts_map = counts_map.copy()
                ts_map.data = np.clip(stat_null - stat_full, 0, None)
                
                return ts_map, stat_full_map, stat_null_map
            ts_map, stat_full_map, stat_null_map = evaluate_test_statistics(counts_map, npred_sig_map, npred_bkg_map)
        
            filterd_table = QTable({
                "plane": np.where(galactic_plane_array, "galactic", "off"),
                lon_key: lon_data,
                lat_key: lat_data,
                "obs_start": [obs_snap.metadata["obs_duration"][0]] * len(lon_data),
                "obs_stop": [obs_snap.metadata["obs_duration"][1]] * len(lon_data),
                "emin": ([round(energy_edges[eidx_min].value)] * len(lon_data)) * energy_edges[eidx_min].unit,
                "emax": ([round(energy_edges[len(energy_edges)-1].value)] * len(lon_data)) * energy_edges[len(energy_edges)-1].unit,
                "counts": counts_map.data.flatten(),
                "npred_sig": npred_sig_map.data.flatten(),
                "npred_bkg": npred_bkg_map.data.flatten(),
                "ts": ts_map.data.flatten(),
                "stat_full": stat_full_map.data.flatten(),
                "stat_null": stat_null_map.data.flatten()
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
            
        from astropy.table import vstack
        return vstack(table)

    def plot_pbh_spectrum(
        self,
        skymodels,
        spilit_component=True,
        ax=None
    ):
        '''
        pbh = AstroObject.PrimordialBlackHole(final_epoch="2000-01-01 00:00:00")
        skymodels = pbh.get_evolution_skymodels()
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
        # slider-axis for time to evaporation
        plt.subplots_adjust(bottom=0.2)
        ax_evaporation = fig.add_axes(
            [0.3, 0.08, 0.4, 0.02],
            facecolor='lightsteelblue'
        )

        # 各skymodelsの時間情報を取得
        time_to_evaporation = []
        for skymodel in skymodels:
            # lifetimeの値を抽出
            lifetime_str = skymodel.name.split("_")[0].replace("lifetime", "")
            lifetime = u.Quantity(lifetime_str)
            time_to_evaporation.append(lifetime.to(u.s).value)
            # distanceの値を抽出
            distance_str = skymodel.name.split("_")[1].replace("distance", "")
            distance = u.Quantity(distance_str)
        time_to_evaporation_log = np.log10(time_to_evaporation)
        time_to_evaporation_to_zero = np.append(time_to_evaporation, [0])
        time_to_evaporation_to_zero_log = np.log10(
            np.append(time_to_evaporation, [time_to_evaporation[-1]/10.])
        )
        evaporation_slider = Slider(
            ax_evaporation,
            'log10(Time to evaporation)',
            np.log10( (0.1*u.s).value ),
            np.log10( (10*u.year).to(u.s).value ),
            #time_to_evaporation_log[-1],
            #time_to_evaporation_log[0],
            valstep=np.flip(time_to_evaporation_log),
            #valinit=time_to_evaporation[0],
            valinit=np.log10( (0.1*u.s).value ),
            initcolor='hotpink'
        )
        #cmap_time = plt.cm.get_cmap('rainbow')
        cmap_time = mpl.colormaps.get_cmap('rainbow')
        norm_time = mpl.colors.LogNorm(
            vmax=10**np.log10( (10*u.year).to(u.s).value ),
            vmin=10**np.log10( (0.1*u.s).value )
        )
        tick_seconds = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
        tick_labels = [
            f"{t:.0e}s\n({(t * u.s).to(u.day).value:.1f} days)"
            for t in tick_seconds
        ]
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm_time, cmap=cmap_time),
            ax=ax,
            label='Time to evaporation [s]'
        )
        # カスタムティックをカラーバーに適用
        cbar.set_ticks(tick_seconds)
        cbar.set_ticklabels(tick_labels)

        def show_spectrum(val):
            '''Draws primary/secondary photon spectra of the blackhole (for 4pi sr).'''
            # 現在のスライダーに対応するインデックスの情報をそれぞれ取得
            ival = np.absolute(time_to_evaporation_log - val).argmin()
            
            # スペクトルのプロット
            t_to_evap = time_to_evaporation[ival]
            if spilit_component:
                skymodels[int(ival)].spectral_model.model1.plot(
                    ax=ax,
                    energy_bounds=(1e-6, 1e+6) * u.GeV,
                    sed_type='dnde',
                    ls='-', lw=1, marker='o', ms=2, alpha=1.0,
                    color=cmap_time(norm_time(t_to_evap)),
                    #label='photon primary at {0:1.2E} s'.format(t_to_evap),
                    )
                skymodels[int(ival)].spectral_model.model2.plot(
                    ax=ax,
                    energy_bounds=(1e-6, 1e+6) * u.GeV,
                    sed_type='dnde',
                    ls='-', lw=1, marker=',', ms=0, alpha=0.5,
                    color=cmap_time(norm_time(t_to_evap)),
                    #label='photon_secondary at {0:1.2E} s'.format(t_to_evap),
                    )
            elif not spilit_component:
                skymodels[int(ival)].spectral_model.plot(
                    ax=ax,
                    energy_bounds=(1e-6, 1e+6) * u.GeV,
                    sed_type='dnde',
                    ls='-', lw=1, marker=',', ms=0, alpha=0.5,
                    color=cmap_time(norm_time(t_to_evap)),
                    #label='photon at {0:1.2E} s'.format(t_to_evap),
                )
            
            # Add vertical dashed lines
            ax.axvline(x=100*u.MeV, color='gray', linestyle='--')
            ax.axvline(x=1*u.TeV, color='gray', linestyle='--')
            #ax.legend()
            evaporation_slider.poly.set(facecolor=cmap_time(norm_time(t_to_evap)))
            ax.set_title(f"distance: {distance.to(u.pc).value} parsec, lifetime: {t_to_evap} second")
            fig.canvas.draw_idle()
            
        def reset_flux(event):
            evaporation_slider.reset()
            ax.clear()
    
        ax_flux_reset = fig.add_axes([0.8, 0.08, 0.1, 0.02])
        button_flux_reset = Button(
            ax_flux_reset,
            'Reset',
            color='g', hovercolor='r'
        )

        evaporation_slider.on_changed(show_spectrum)
        button_flux_reset.on_clicked(reset_flux)
        plt.show()
        
    def plot_5sigma_detection(
        self,
        emin=1*u.GeV,
        emax=1*u.TeV,
        obs_duration=1*u.week,
        ref_data = np.array([
            [31.6227766017, 56.234132519, 3.34585556061e-12, 0],
            [56.234132519, 100.0, 1.63932302182e-12, 0],
            [100.0, 177.827941004, 9.77430679247e-13, 0],
            [177.827941004, 316.227766017, 6.66254775473e-13, 0],
            [316.227766017, 562.34132519, 4.52306699307e-13, 0],
            [562.34132519, 1000.0, 3.15655507706e-13, 0],
            [1000.0, 1778.27941004, 2.35356129532e-13, 0],
            [1778.27941004, 3162.27766017, 1.95280483884e-13, 0],
            [3162.27766017, 5623.4132519, 2.01383319924e-13, 0],
            [5623.4132519, 10000.0, 2.25904853343e-13, 0],
            [10000.0, 17782.7941004, 4.0298701638e-13, 0],
            [17782.7941004, 31622.7766017, 7.19442534414e-13, 0],
            [31622.7766017, 56234.132519, 1.25429910732e-12, 0],
            [56234.132519, 100000.0, 2.22279983578e-12, 0],
            [100000.0, 177827.941004, 3.93514424973e-12, 0],
            [177827.941004, 316227.766017, 6.96004573381e-12, 0],
            [316227.766017, 562341.32519, 1.24404292786e-11, 0],
            [562341.32519, 1000000.0, 2.30972760945e-11, 0],
        ])
    ):
        # エネルギー中央値 (MeV)
        ref_emin = ref_data[:, 0] * u.MeV
        ref_emax = ref_data[:, 1] * u.MeV
        ref_emid = np.sqrt(ref_emin * ref_emax)
        
        # 感度データ
        ref_sensitivity = ref_data[:, 2] * u.erg * u.cm**-2 * u.s**-1
        
        # 条件に一致するデータを抽出
        mask = (ref_emin >= emin.to(u.MeV)) & (ref_emax <= emax.to(u.MeV))
        sensitivity_in_range = ref_sensitivity[mask]
        # 平均感度の計算
        average_sensitivity_flux_threshold = np.mean(sensitivity_in_range)

    
        import numpy as np
        import matplotlib.pyplot as plt

        # 距離と光度の範囲を設定
        distance = np.logspace(20, 25, 500)  # 距離 [cm] (例: 10 kpc ~ 10 Mpc)
        
        pbh = AstroObject.PrimordialBlackHole()
        TIME_OF_EVAPORATION = Time(pbh.epoch_state["date"])
        
        obs_averaged_skymodel = pbh.get_observation_averaged_skymodels(
            [TIME_OF_OBS_START.iso, TIME_OF_OBS_STOP.iso],
            distance=1*u.pc,
            particle="photon",
            frame="galactic",
        )
        obs_states = pbh.track_states([TIME_OF_OBS_START.iso, TIME_OF_OBS_STOP.iso])
        
        luminosity = np.logspace(35, 45, 500)  # 光度 [erg/s] (例: 1e35 ~ 1e45 erg/s)

        # 2Dグリッドを生成
        D, L = np.meshgrid(distance, luminosity)

        # フラックス条件を計算
        flux = L / (4 * np.pi * D**2)

        # 条件を満たす箇所をマスク
        valid = flux >= average_sensitivity_flux_threshold

        # プロット
        plt.figure(figsize=(10, 8))
        plt.contourf(D / (3.086e18), L, valid, levels=[0, 0.5, 1], colors=["white", "blue"], alpha=0.7)
        plt.colorbar(label="Condition Met (1=True, 0=False)")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Distance [kpc]")
        plt.ylabel("Luminosity [erg/s]")
        plt.title("Luminosity vs Distance for Flux Condition")
        plt.grid(which="both", linestyle="--", alpha=0.5)
        plt.show()


    '''
    from pbh_explorer.objects import AstroObject
    obs_snap = detector.obs_campaign.snaps[0]
    
    import numpy as np
    from astropy import units as u
    distances = np.logspace(np.log10(0.01), np.log10(1), 5) * u.pc
    lifetimes = np.logspace(np.log10(1), np.log10(365 * 10), 5) * u.day

    from itertools import product
    from tqdm import tqdm
    
    from astropy.time import Time, TimeDelta
    TIME_OF_OBS_START = Time("1999-12-25 00:00:00")
    TIME_OF_OBS_STOP = Time("2000-01-01 00:00:00")
    
    X, Y = np.meshgrid(distances, lifetimes)
    Z = np.zeros_like(X.value)
    rows, cols = Z.shape
    skymodels = []
    with tqdm(total=rows * cols, desc="Initializing...") as progress_bar:
        for row, col in product(range(rows), range(cols)):
            # 各グリッドの値を取得
            distance = X[row, col]
            lifetime = Y[row, col]
            
            # 観測時間での平均的なモデルを作成
            TIME_OF_EVAPORATION = TIME_OF_OBS_START + TimeDelta(lifetime.to(u.s))
            pbh = AstroObject.PrimordialBlackHole.generate_random_profile_pbh(
                final_epoch_range=(TIME_OF_EVAPORATION.iso, TIME_OF_EVAPORATION.iso),
                position_range={
                    #"ra": (0, 360)*u.deg,
                    "ra": (0, 0)*u.deg,
                    #"dec": (-90, 90)*u.deg,
                    "dec": (0, 0)*u.deg,
                    "distance": (distance.to(u.pc).value, distance.to(u.pc).value)*u.pc
                },
                velocity_range={
                    "pm_ra_cosdec": (0, 0)*u.mas/u.yr,
                    "pm_dec": (0, 0)*u.mas/u.yr,
                    "radial_velocity": (0, 0)*u.km/u.s
                }
            )
            obs_averaged_skymodel = pbh.get_observation_averaged_skymodels(
                [TIME_OF_OBS_START.iso, TIME_OF_OBS_STOP.iso],
                distance=distance.to(u.pc),
                particle="photon",
                frame="galactic",
                addtional_name=f"fake"
            )
            skymodels += obs_averaged_skymodel
            # データセットに擬似PBHを生成
            obs_snap.map_dataset.models = Models(obs_averaged_skymodel)
            print(obs_snap.map_dataset)
            
            Z[row, col] = np.sum(obs_snap.map_dataset.npred_signal())
            
            # tqdm の desc を更新
            progress_bar.set_description(f"Distance: {distance:.3f}, Lifetime: {lifetime:.3f}")
            progress_bar.update(1)
    
    import matplotlib.pyplot as plt
    plt.pcolormesh(X.value, Y.value, Z, shading="auto", cmap='rainbow')
    plt.colorbar(label="Fraction of Detected-PBH")
    plt.xlabel(f"Distance ({X.unit})")
    plt.ylabel(f"Lifetime ({Y.unit})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(distances[0].value, distances[-1].value)
    plt.ylim(lifetimes[0].value, lifetimes[-1].value)
    plt.show()
    '''

class PBHSimulator:
    def __init__(
        self,
        files=[],
        telescope="fermi_lat",
        map_type="hpx",
        insert_wcs_dataset=True,
        loglevel="DEBUG"
    ):
        self._detector = PBHExplorer(
            files=files,
            telescope=telescope,
            map_type=map_type,
            insert_wcs_dataset=insert_wcs_dataset,
            loglevel=loglevel
        )
        self._loglevel = loglevel
        
        self.real_obs_campaign = copy.deepcopy(self._detector.obs_campaign)
        self.fake_obs_campaign = None
        
    def generate_fake_obs_campaign(
        self,
        real_obs_snap,
        nfakes=1,
        fake_pbh_params=None, # (distance, lifetime, npbhs)
        use_fake_counts_as_npred_bkg=False,
        #progress_bar=None  # 追加: プログレスバーの引数
    ):
        TIME_OF_OBS_START = Time(real_obs_snap.metadata["obs_duration"][0])
        TIME_OF_OBS_STOP = Time(real_obs_snap.metadata["obs_duration"][1])
        if fake_pbh_params:
            distance = fake_pbh_params[0]
            lifetime = fake_pbh_params[1]
            npbhs = fake_pbh_params[2]
            TIME_OF_EVAPORATION = TIME_OF_OBS_START + TimeDelta(lifetime.to(u.s))

        fake_obs_snaps = []
        with tqdm(total=nfakes, desc="Generating fake observations") as progress_bar:
            for fake_idx in range(nfakes):
                progress_bar.set_description(f"Fake ObsSnap {fake_idx+1}/{nfakes}")
                fake_obs_snap = copy.deepcopy(real_obs_snap)
                
                # backgroundのfakeカウント生成
                fake_obs_snap.map_dataset.models = None
                fake_obs_snap.map_dataset.fake()
                fake_obs_snap.data["counts"] = fake_obs_snap.map_dataset.counts.copy()
                if use_fake_counts_as_npred_bkg:
                    fake_obs_snap.data["npred_bkg"] = fake_obs_snap.map_dataset.counts.copy()
            
                # 擬似PBHのfakeカウント生成
                if fake_pbh_params:
                    obs_averaged_pbhs = []
                    for pbh_idx in range(npbhs):
                        progress_bar.set_description(f"Fake ObsSnap {fake_idx+1}/{nfakes}, Fake PBH SkyModel ({distance.to(u.pc):.3f}, {lifetime.to(u.day):.3f}) {pbh_idx + 1}/{npbhs}")
                        
                        pbh = AstroObject.PrimordialBlackHole.generate_random_profile_pbh(
                            final_epoch_range=(
                                TIME_OF_EVAPORATION.iso, TIME_OF_EVAPORATION.iso
                            ),
                            position_range={
                                "ra": (0, 360)*u.deg,
                                "dec": (-90, 90)*u.deg,
                                "distance": (distance.to(u.pc).value, distance.to(u.pc).value)*u.pc
                            },
                            velocity_range={
                                "pm_ra_cosdec": (0, 0)*u.mas/u.yr,
                                "pm_dec": (0, 0)*u.mas/u.yr,
                                "radial_velocity": (0, 0)*u.km/u.s
                            }
                        )
                        obs_averaged_skymodel, obs_states = pbh.get_observation_averaged_skymodels(
                            [TIME_OF_OBS_START.iso, TIME_OF_OBS_STOP.iso],
                            distance=distance.to(u.pc),
                            particle="photon",
                            frame="galactic"
                        )
                        obs_averaged_pbhs += obs_averaged_skymodel
                        print(obs_averaged_pbhs[-1])
                        
                    # 生成したモデルからfake関数を実行して擬似カウントを作る
                    fake_obs_snap.map_dataset.models = Models(obs_averaged_pbhs)
                    fake_obs_snap.map_dataset.background = None
                    fake_obs_snap.map_dataset.fake()
                    fake_obs_snap.data["counts"] += fake_obs_snap.map_dataset.counts.copy()
                    fake_obs_snap.map_dataset.counts += fake_obs_snap.data["npred_bkg"]
                    fake_obs_snap.map_dataset.background = fake_obs_snap.data["npred_bkg"]

                fake_obs_snaps.append(fake_obs_snap)
                progress_bar.update(1)
        
        # 上記で作成されたfake_obs_snapをobs_campaignでまとめる
        self.fake_obs_campaign = ObsCampaign.FermiLAT(
            obs_snaps=fake_obs_snaps,
            loglevel=self._loglevel
        )

    def fraction_of_detected_pbh(
        self,
        real_obs_snap,
        nfakes=52,
        distances = np.linspace(0.02, 0.02, 2) * u.pc,
        lifetimes = np.logspace(np.log10(0.1), np.log10(365*5), 2) * u.day,
        local_evaporation_rate = 52 * u.year**-1 * u.pc**-3,
        method_kwargs={},
        **filtering_kwargs,
    ):
        # real_obs_snapの観測時間に対するイベント数をlocal_evaporation_rateから計算
        TIME_OF_OBS_START = Time(real_obs_snap.metadata["obs_duration"][0])
        TIME_OF_OBS_STOP = Time(real_obs_snap.metadata["obs_duration"][1])
        obs_duration = 1 * u.week #(TIME_OF_OBS_STOP - TIME_OF_OBS_START).sec * u.s
        evaporation_rate_for_obs_duration = round(
            (local_evaporation_rate.to(u.s**-1*u.pc**-3) * u.pc**3 * obs_duration.to(u.s)).value
        )
        
        X, Y = np.meshgrid(distances, lifetimes)
        Z = np.zeros_like(X.value)
        rows, cols = Z.shape
        skymodels = []
        for row, col in product(range(rows), range(cols)):
            # 各グリッドの値を取得
            distance = X[row, col]
            lifetime = Y[row, col]

            # self.fake_obs_campaignを作って、self._detector下に登録する
            self.generate_fake_obs_campaign(
                real_obs_snap,
                nfakes=nfakes,
                fake_pbh_params=(distance, lifetime, evaporation_rate_for_obs_duration),
                use_fake_counts_as_npred_bkg=True
            )
            #self._detector.obs_campaign = copy.deepcopy(self.real_obs_campaign)
            #self._detector.setup_obs_snap(real_obs_snap, method_kwargs=method_kwargs)
            self._detector.obs_campaign = self.fake_obs_campaign
            
            # 条件を通過したイベントのテーブルを取得
            filtered_table = self._detector.extract_filtered_table(**filtering_kwargs)
            #logger.debug(filtered_table)
            print(filtered_table)
            
            #
            #Z[row, col] = len(filtered_table)
            Z[row, col] = evaporation_rate_for_obs_duration if len(filtered_table) > evaporation_rate_for_obs_duration else len(filtered_table)
        Z /= (evaporation_rate_for_obs_duration * nfakes)
        
        plt.pcolormesh(X.value, Y.value, Z, shading="auto", cmap="viridis")
        plt.colorbar(label="Fraction of Detected-PBHs")
        plt.xlabel(f"Distance ({X.unit})")
        plt.ylabel(f"Lifetime ({Y.unit})")
        plt.xlim(distances[0].value, distances[-1].value)
        plt.ylim(lifetimes[0].value, lifetimes[-1].value)
        plt.show()
        
    def compute_pbh_alarm_rate(
        self,
        real_obs_snap,
        fake_pbh_params=(0.01*u.pc, 1*u.week, 1),
        nsimulations=10
    ):
        total_evap_alarms_per_energy = {}

        # アラーム条件を設定
        alarm_conditions = [
            {"level": "level3", "nphotons": 10, "ts_min": 25},
            {"level": "level2", "nphotons": 8, "ts_min": 16},
            {"level": "level1", "nphotons": 5, "ts_min": 9},
        ]

        for _ in range(nsimulations):
            # 年間のシュミレート
            # TODO: nfakes should be changed depends on obs_snap duration
            self.generate_fake_obs_campaign(
                real_obs_snap,
                nfakes=52,
                fake_pbh_params=fake_pbh_params,
                use_fake_counts_as_npred_bkg=True, #trueにすることで、false alarm rateを0にできる
            )
            self._detector.obs_campaign = self.fake_obs_campaign

            # 各エネルギー範囲で pbh_alarms を計算
            energy_edges = self.fake_obs_campaign.snaps[0].map_dataset.geoms["geom"].axes["energy"].edges
            for emin in energy_edges[:-1]:
                emin = round(emin.value) * emin.unit
                emax = round(energy_edges[-1].value) * energy_edges[-1].unit
                if emin not in total_evap_alarms_per_energy:
                    total_evap_alarms_per_energy[emin] = {"level3": 0, "level2": 0, "level1": 0}
                
                for condition in alarm_conditions:
                    level = condition["level"]
                    nphotons = condition["nphotons"]
                    ts_min = condition["ts_min"]

                    # 現在のエネルギー範囲に該当するデータと条件を用いてフィルタリング
                    filtered_table = self._detector.extract_filtered_table(
                        nphotons_min=nphotons,
                        ts_min=ts_min,
                        energy_min=emin,
                        use_galactic_plane=True,
                        use_off_plane=True,
                        lat_threshold=10*u.deg
                    )
                    
                    # 条件を通過したイベントの数を記録
                    total_evap_alarms_per_energy[emin][level] += len(filtered_table)

            # 不要なメモリを解放
            import gc
            del self._detector.obs_campaign
            del self.fake_obs_campaign
            del filtered_table
            gc.collect()
            #import psutil
            # メモリ使用量を測定
            #process = psutil.Process()
            #memory = process.memory_info().rss  # 常駐メモリ使用量 (bytes)
            #print(f"Memory usage before deletion: {memory / 1024**2:.2f} MB")
            
        # 結果をエネルギー範囲ごとに出力
        print("PBH Evaporation Alarm Rates (per energy range):")
        print(f"{real_obs_snap.map_dataset.geoms['geom']}")
        print(f"Observation range: {real_obs_snap.metadata['obs_duration']}")
        for emin, evap_alarms in total_evap_alarms_per_energy.items():
            print(f"Energy range: {emin} - {emax}")
            for level, count in evap_alarms.items():
                average_rate = count / nsimulations  # 年間の平均 alarm rate
                print(f"  {level}: {count} pbh evaporation alarms in total, {average_rate:.2f} per year (on average)")
        
    def compute_false_alarm_rate(
        self,
        # wcsのbinszの細かさで結構変わる
        real_obs_snap,
        nsimulations=10
    ):
        total_false_alarms_per_energy = {}
        
        # アラーム条件の設定
        alarm_conditions = [
            {"level": "level3", "nphotons": 10, "ts_min": 25},
            {"level": "level2", "nphotons": 8, "ts_min": 16},
            {"level": "level1", "nphotons": 5, "ts_min": 9},
        ]
            
        for _ in range(nsimulations):
            # 年間のシュミレート
            self.generate_fake_obs_campaign(
                real_obs_snap,
                nfakes=52,
                fake_pbh_params=None,
                use_fake_counts_as_npred_bkg=False
            )
            self._detector.obs_campaign = self.fake_obs_campaign

            # 各エネルギー範囲で false_alarms を計算
            energy_edges = \
                self.fake_obs_campaign.snaps[0].map_dataset.geoms["geom"].axes["energy"].edges
            for emin in energy_edges[:-1]:
                emin = round(emin.value) * emin.unit
                emax = round(energy_edges[-1].value) * energy_edges[-1].unit
                
                if emin not in total_false_alarms_per_energy:
                    total_false_alarms_per_energy[emin] = {"level3": 0, "level2": 0, "level1": 0}
                
                for condition in alarm_conditions:
                    level = condition["level"]
                    nphotons = condition["nphotons"]
                    ts_min = condition["ts_min"]

                    # 現在のエネルギー範囲に該当するデータと条件を用いてフィルタリング
                    filtered_table = self._detector.extract_filtered_table(
                        nphotons_min=nphotons,
                        ts_min=ts_min,
                        energy_min=emin,
                        use_galactic_plane=True,
                        use_off_plane=True,
                        lat_threshold=10*u.deg
                    )
                    
                    # 条件を通過したイベントの数を記録
                    total_false_alarms_per_energy[emin][level] += len(filtered_table)
            
            # 不要なメモリを解放
            import gc
            del self._detector.obs_campaign
            del self.fake_obs_campaign
            del filtered_table
            gc.collect()
            #import psutil
            # メモリ使用量を測定
            #process = psutil.Process()
            #memory = process.memory_info().rss  # 常駐メモリ使用量 (bytes)
            #print(f"Memory usage before deletion: {memory / 1024**2:.2f} MB")

        # 結果をエネルギー範囲ごとに出力
        print("False Alarm Rates (per energy range):")
        print(f"{real_obs_snap.map_dataset.geoms['geom']}")
        print(f"Observation range: {real_obs_snap.metadata['obs_duration']}")
        for emin, false_alarms in total_false_alarms_per_energy.items():
            print(f"Energy range: {emin} - {emax}")
            for level, count in false_alarms.items():
                average_rate = count / nsimulations  # 年間の平均 false alarm rate
                print(f"  {level}: {count} false alarms in total, {average_rate:.2f} per year (on average)")

    def mask_from_catalog():
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

    def add_pbh_models(self, sig_models):
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
        if not self._obs_on.map_dataset.models:
            org_dataset_sig_models = Models([])
        else:
            org_dataset_sig_models = Models([])
            for sky_model in self._obs_on.map_dataset.models[:]:
                org_dataset_sig_models.append(sky_model)
        
        self._obs_on.map_dataset.models = org_dataset_sig_models + sig_models
        print(self._obs_on.map_dataset)
        
    def clear_pbh_models(self, names=None):
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
            off_region_spectra.append(
                self._obs_off.map_dataset.to_spectrum_dataset(off_region)
            )
        return on_region_spectra, off_region_spectra

    def plot_on_region(self):
        pass

    def evaluate_npred_map(self):
        pass

    def evaluate_residual_map(self):
        #self.evaluate_counts_mapとself.evaluate_npred_mapを使って、そのさをみる
        pass
        
        
    def fit_map():
        pass
    def fit_spectrum():
        pass
    def test_statistics(self, test_models_name):
        pass

