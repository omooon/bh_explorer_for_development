#!/usr/bin/env python

'''
ObsCampaign stands for a project to derive a specific result
from a series of ObsSnap.
It holds information on the target objects
and the ObsSnap series.
The IRF is assumed to be different for each ObsSnap.
ObsCampaign can be composed of multiple observation
runs toward one or more celestical regions.
'''
from datetime import datetime

from astropy import constants as c
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from astropy.table import QTable

from pathlib import Path
from gammapy.maps import MapAxis, WcsNDMap, HpxNDMap
from gammapy.modeling.models import Models
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from logging import getLogger, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)

class FermiLAT:
    def __init__(self, obs_snaps=[], loglevel=None):
        """
        ObsSnapFermiLAT holds information on a short-term observation for Fermi-LAT Telescope.
        Here "short-term" means the IRF can be considered to be constant.'
        
        Parameters:
        ----------
        obs_snaps : dict, default={}
            A dictionary of ObsSnap instances to be analysed in the campaign.
            The key must be the time starting of the ObsSnap.
        """
        # loglevel setting
        if loglevel:
            logger.setLevel(loglevel)
            for handler in logger.handlers:
                handler.setLevel(loglevel)
        
        self.snaps = obs_snaps
        self.reference_time = self.snaps[0].map_dataset.gti.time_ref
        
    def evaluate_signal_detection_per_snap(
        self,
        skymodel_list,
        use_first_snap_result=False,
        **kwargs
    ):
        for i, obs_snap in enumerate(self.snaps):
            if i == 0:
                obs_snap.evaluate_signal_detection(skymodel_list, **kwargs)
            elif i > 0:
                if use_first_snap_result:
                    obs_snap.data["npred_signal"] = self.snaps[0].data["npred_signal"]
                elif not use_first_snap_result:
                    obs_snap.evaluate_signal_detection(skymodel_list, **kwargs)
                    
    def get_detected_sources_table(self, **kwargs):
        detected_tables = []
        for obs_snap in self.snaps:
            detected_tables.append(obs_snap.get_detected_source_table(**kwargs))

        from astropy.table import vstack
        return vstack(detected_tables)
    
    def get_delta_livetimes(self):
        """
        Command to return the livetime duration of each observation.

        Returns:
            np.ndarray: the livetime duration of each observation.
        """
        delta_livetimes = np.array([0]*len(self.snaps)) * u.s
        
        for iobs, obs_snap in enumerate(self.snaps):
            delta_livetime = (obs_snap.time_stop - obs_snap.time_start).sec * u.s
            delta_livetimes[iobs] = delta_livetime
        
        return delta_livetimes

    def get_timeres_cubes(self):
        timeres_cubes = []
        for obs_snap in self.snaps.values():
            timeres_cubes.append(
                obs_snap.map_dataset.counts
            )
        return timeres_cubes
        
    def get_timeres_onoffspectra(self):
        timeres_onoffspectra = []
        for obs_snap in self.snaps.values():
            timeres_onoffspectra.append(
                obs_snap.onoff_spectrum_dataset
            )
        return timeres_onoffspectra

    def evaluate_count_statistics(
        self,
        show_mode="individual",
        **kwargs
    ):
        """
        Obs_snapごとの分布を観測時間のスライダー形式、もしくは、全部合わせてプロット
        Evaluates the counts distribution within the dataset.

        Parameters:
        -----------
        show_mode:
            "individual" or "combined"
        """
        if show_mode == 'individual':
            fig, ax = plt.subplots(figsize=(12, 8))

            # The outside frame of the ax is turned off.
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set(xticks=[], yticks=[], title="")
        
            # create space for slider
            plt.subplots_adjust(bottom=0.2)
            
            # create slider
            ax_slider = plt.axes(
                [0.25, 0.05, 0.65, 0.03],
                facecolor='lightgoldenrodyellow'
            )
            dates = list(self.snaps.keys())
            slider = Slider(
                ax=ax_slider,
                label="Date of observation",
                valmin=0,
                valmax=len(dates) - 1,
                valstep=1,
                valinit=0
            )
            slider.valtext.set_position((0, 1.05))
            
            # initial plot
            cwindow, cmasked_roi_counts, cpassed_roi_counts_dict = \
                self.snaps[dates[slider.valinit]].evaluate_count_statistic(
                    show=False,
                    **kwargs
                )
            self.__axes = self.snaps[dates[slider.valinit]]._plot_for_evaluate_count_statistic_function(
                cwindow,
                cmasked_roi_counts.data.flatten(),
                cpassed_roi_counts_dict,
                parent_ax=ax,
                tight_layout=False # Warning: if True, plt.subplots_adjust(bottom=0.2) is cancelled
            )
            slider.valtext.set_text(f"{dates[slider.valinit]} to ")
            
            def update(val):
                self.__axes["Left"].remove()
                self.__axes["Top Right"].remove()
                self.__axes["Bottom Right"].remove()
        
                # new plot
                idx = int(slider.val)
                cwindow, cmasked_roi_counts, cpassed_roi_counts_dict = \
                    self.snaps[dates[idx]].evaluate_count_statistic(
                        show=False,
                        **kwargs
                    )
                self.__axes = self.snaps[dates[idx]]._plot_for_evaluate_count_statistic_function(
                    cwindow,
                    cmasked_roi_counts.data.flatten(),
                    cpassed_roi_counts_dict,
                    parent_ax=ax,
                    tight_layout=False # Warning: if True, plt.subplots_adjust(bottom=0.2) is cancelled
                )
                slider.valtext.set_text(f"{dates[idx]} to ")
                
            slider.on_changed(update)
            plt.show()
        
        elif show_mode == 'combined':
            # Plotting with data across self.snaps.
            cmasked_roi_counts_data_list = []
            total_cpassed_roi_counts_dict = None
            for key, obs_snap in self.snaps.items():
                cwindow, cmasked_roi_counts, cpassed_roi_counts_dict = \
                    obs_snap.evaluate_count_statistic(
                        show=False,
                        **kwargs
                    )
                    
                cmasked_roi_counts_data_list.append(cmasked_roi_counts.data.flatten())
                if not total_cpassed_roi_counts_dict:
                    total_cpassed_roi_counts_dict = cpassed_roi_counts_dict.copy()
                else:
                    total_cpassed_roi_counts_dict["map"].data += cpassed_roi_counts_dict["map"].data
                    
            # Adjust the number of pixels according to the number of self.snaps.
            combined_factor = len(self.snaps)
            total_cpassed_roi_counts_dict["npixels"] = \
                total_cpassed_roi_counts_dict["npixels"] * combined_factor
            total_cpassed_roi_counts_dict["galactic plane region"]["npixels"] = \
                total_cpassed_roi_counts_dict["galactic plane region"]["npixels"] * combined_factor
            total_cpassed_roi_counts_dict["off plane region"]["npixels"] = \
                total_cpassed_roi_counts_dict["off plane region"]["npixels"] * combined_factor
        
            # Run a plot using any obs_snap instance.
            obs_snap._plot_for_evaluate_count_statistic_function(
                cwindow,
                np.concatenate(cmasked_roi_counts_data_list),
                total_cpassed_roi_counts_dict,
                parent_ax=None, # If set to None, the plot is executed internally within this function.
                tight_layout=True
            )

class CTAO:
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
        self.snaps = obs_snaps
        self.reference_time = reference_time

    def timeres_onoffspectra(self):
        timeres_onoffspectra = []
        for obs_snap in self.snaps.values():
            timeres_onoffspectra.append(
                obs_snap.onoff_spectrum_dataset
            )
        return timeres_onoffspectra

    def timeres_cubes(self):
        timeres_cubes = []
        for obs_snap in self.snaps.values():
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
        return list(self.snaps.values())

    def on_regions(self):
        """Returns a list of on regions for each observation.

        Returns:
        -------
        on_regions: list
            A list of on regions for each observation.
        """
        on_regions = []
        for obs_snap in self.snaps.values():
            on_regions.append(obs_snap.oncount_geom.region)
        return on_regions
