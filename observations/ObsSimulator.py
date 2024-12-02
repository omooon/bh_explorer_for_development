#!/usr/bin/env python
import numpy as np
from pathlib import Path, PosixPath
import numpy.ma as ma
import matplotlib as mpl
import json
from astropy import units as u
from astropy.time import Time
from gammapy.modeling.models import SkyModel
from regions import CircleSkyRegion

from logging import (
    getLogger,
    StreamHandler
)

from ObsSnap import ObsSnap

# Logger
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)


class ObsCampaign:
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
        self.outdir = outdir if isinstance(outdir, Path) else Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.obs_snaps = obs_snaps
        self.reference_time = reference_time

    def delta_times(self):
        """Command to return the livetime duration of each observation.

        Returns:
            np.ndarray: the livetime duration of each observation.
        """
        delta_times = np.ndarray(shape=(len(self.obs_snaps))) * u.s
        # delta_times = u.Quantity(
        #     np.ndarray(shape=(len(self.obs_snaps))),
        #     unit=u.s
        #     )
        for iobs, obs_snap in enumerate(self.obs_snaps.values()):
            delta_time =\
                (obs_snap.observation.tstop - obs_snap.observation.tstart)\
                .to(u.s)
            # logger.debug('delta_time: {0}'.format(delta_time))
            delta_times[iobs] = delta_time
        return delta_times

    def time_to_evaporation(self):
        time_to_evaporation = np.ndarray(shape=(len(self.obs_snaps)))
        logger.debug(self.reference_time)
        for iobs, obs_snap in enumerate(self.obs_snaps.values()):
            logger.debug(obs_snap.observation.tstart)
            time_to_evaporation[iobs]\
                = (self.reference_time - obs_snap.observation.tstart)\
                .to_value(u.s)
        logger.debug('Time to evaporation:\n{0}'.format(time_to_evaporation))
        return time_to_evaporation

    def time_to_evaporation_to_zero(self):
        return np.append(self.time_to_evaporation(), [0])

    def time_to_evaporation_log(self):
        return np.log10(self.time_to_evaporation())

    def time_to_evaporation_to_zero_log(self):
        """Query to return the log10 of the time to evaporation,
with the last value being 1/10 of the previous value
to avoid log10(0).

    Returns:
        np.ndarray: the log10 of the time to evaporation.
    """
        time_to_evaporation = self.time_to_evaporation()
        return np.log10(
            np.append(time_to_evaporation, [time_to_evaporation[-1] / 10.])
            )

    def common_energy_edges(self):
        common_energy_edges = None
        for obs_snap in self.obs_snaps.values():
            if common_energy_edges is None:
                common_energy_edges =\
                    obs_snap.energy_axis.edges
            else:
                common_energy_edges = np.intersect1d(
                    common_energy_edges,
                    obs_snap.energy_axis.edges
                    )
        return common_energy_edges

    def common_energy_edges_TeV(self):
        return self.common_energy_edges().to(u.TeV).value

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

    def dump_json(self, path_jsonfile=None, overwrite=False):
        """Writes a dictionary representation of
        this object's non-function properties to a JSON file.

         Parameters:
         ----------
         path_jsonfile : str
             The path of the JSON file to write to.
         """
        if path_jsonfile is None:
            path_jsonfile = self.outdir / 'obs_campaign.json'
        data_dict = {}
        for kmem, vmem in vars(self).items():
            #  Custom conversions
            if kmem == 'obs_snaps':
                data_dict[kmem] = {}
                for ksnap, vsnap in self.obs_snaps.items():
                    path_obssnap = vsnap.outdir / 'obs_snap.json'
                    vsnap.dump_json(
                        path_jsonfile=path_obssnap,
                        overwrite=overwrite
                        )
                    data_dict[kmem][ksnap] = str(path_obssnap)
            #  Generic conversion
            elif not callable(vmem) and not kmem.startswith("__"):
                if isinstance(vmem, Path) or isinstance(vmem, PosixPath):
                    data_dict[kmem] = str(vmem)
                else:
                    data_dict[kmem] = vmem

        with open(path_jsonfile, 'w') as json_file:
            # convert non-JSON serializable objects to string
            json.dump(data_dict, json_file, indent=2, default=str)

    def load_json(self, path_jsonfile=None):
        """Reads a dictionary representation of
        this object's non-function properties to a JSON file.

         Parameters:
         ----------
         path_jsonfile : str
             The path of the JSON file to read.
         """
        if path_jsonfile is None:
            path_jsonfile = self.outdir / 'obs_campaign.json'

        with open(path_jsonfile, 'r') as json_file:
            # convert non-JSON serializable objects to string
            data_dict = json.load(json_file)
            for kmem, vmem in data_dict.items():
                #  Custom conversions to JSON
                if kmem == 'obs_snaps':
                    obs_snaps = {}
                    for ksnap, vsnap in vmem.items():
                        path_obssnap = Path(vsnap)
                        obs_snaps[ksnap] = ObsSnap()
                        obs_snaps[ksnap].load_json(
                            path_jsonfile=path_obssnap
                            )
                    setattr(self, kmem, obs_snaps)
                elif kmem == 'reference_time':
                    setattr(self, kmem, Time(vmem))
                #  Generic conversion
                elif isinstance(vmem, str):
                    path_vmem = Path(vmem)
                    if path_vmem.exists():
                        vmem = path_vmem
                    setattr(self, kmem, vmem)

    def max_likelihoods(self, itbin_lo=0, itbin_hi=None):
        """Compute the maximum possible likelihood
for each observation in the campaign.

        Parameters
        ----------
        itbin_lo : int, optional
            The lower index of the time bins to consider, by default 0
        itbin_hi : int, optional
            The upper index of the time bins to consider,
            by default the length of
            the obs_snaps list

        Returns
        -------
        np.ndarray
            The maximum possible likelihood
            for each observation in the campaign
        """
        if itbin_hi is None:
            itbin_hi = len(self.obs_snaps)
        max_test_stats\
            = np.zeros(shape=itbin_hi-itbin_lo)
        for iobs, obs_snap in enumerate(
            list(self.obs_snaps.values())[itbin_lo:itbin_hi]
        ):
            max_test_stats[iobs] = obs_snap.max_likelihood
        return max_test_stats

    def likelihoods(self, models, itbin_lo=0, itbin_hi=None):
        if itbin_hi is None:
            itbin_hi = len(self.obs_snaps)
        test_stats = np.zeros(shape=itbin_hi-itbin_lo)
        for iobs, obs_snap in enumerate(
            list(self.obs_snaps.values())[itbin_lo:itbin_hi]
        ):
            test_stats[iobs] = obs_snap.test_statistics(test_models=models)
        return test_stats

    def fit_spectra(self, skymodel, energy_edges):
        '''Command to fit the spectrum of each ObsSnap in the self.obs_snaps
with the given skymodel.'''
        # timeres_eflux = u.Quantity(
        #     np.zeros(shape=len(self.obs_snaps)),
        #     unit=u.Unit('TeV / (s cm2)')
        #     )
        # timeres_eflux_err = u.Quantity(
        #     np.zeros(shape=len(self.obs_snaps)),
        #     unit=u.Unit('TeV / (s cm2)')
        #     )
        for obs_snap in self.obs_snaps.values():
            obs_snap.fit_spectrum(
                fit_models=skymodel,
                energy_edges=energy_edges
                )
            # timeres_eflux[iobs], timeres_eflux_err[iobs] = \
            #     obs_snap.reconstructed_flux_datasets[skymodel.name]\
            #     .models[0].spectral_model.energy_flux_error(
            #         energy_min=energy_edges[0],
            #         energy_max=energy_edges[-1]
            #         )
        # return (timeres_eflux, timeres_eflux_err)

    def get_energy_flux_curve(self, skymodel):
        """Query to get the reconstructed energy flux curve for the given sky model.

        Parameters
        ----------
        skymodel : `~gammapy.modeling.models.SkyModel`
            The sky model to get the reconstructed energy flux curve for.

        Returns
        -------
        tuple
            A tuple containing the reconstructed energy flux curve and its error.
            The units are astropy quantities with the unit 'TeV/(s cm2)'.
        """
        ENERGY_FLUX_UNIT = u.Unit('TeV / (s cm2)')
        timeres_eflux = u.Quantity(
            np.zeros(shape=len(self.obs_snaps)),
            unit=ENERGY_FLUX_UNIT
            )
        timeres_eflux_err = u.Quantity(
            np.zeros(shape=len(self.obs_snaps)),
            unit=ENERGY_FLUX_UNIT
            )
        for iobs, obs_snap in enumerate(self.obs_snaps.values()):
            reco_eflux\
                = obs_snap.reconstructed_flux[skymodel.name]['energy_flux']
            timeres_eflux[iobs]\
                = (reco_eflux['value'] * u.Unit(reco_eflux['unit']))\
                .to(ENERGY_FLUX_UNIT)
            timeres_eflux_err[iobs]\
                = (reco_eflux['error'] * u.Unit(reco_eflux['unit']))\
                .to(ENERGY_FLUX_UNIT)

        return (timeres_eflux, timeres_eflux_err)

    # def get_spectra(self, skymodel, energy_edges):
    #     '''Fit the spectrum of each ObsSnap in the self.obs_snaps
    #     with the given skymodel.
    #     Returns a tuple of the energy flux spectrum and its error values
    #     against the given energy_edges.'''
    #     timeres_eflux = u.Quantity(
    #         np.zeros(shape=len(self.obs_snaps)),
    #         unit=u.Unit('TeV / (s cm2)')
    #         )
    #     timeres_eflux_err = u.Quantity(
    #         np.zeros(shape=len(self.obs_snaps)),
    #         unit=u.Unit('TeV / (s cm2)')
    #         )
    #     for iobs, obs_snap in enumerate(self.obs_snaps.values()):
    #         # logger.debug(obs_snap.reconstructed_flux_datasets[skymodel.name])
    #         # logger.debug(obs_snap.reconstructed_flux_datasets[skymodel.name].data)
    #         # timeres_eflux[iobs] = \
    #         #     obs_snap.reconstructed_flux_datasets[skymodel.name]\
    #         #     .data.eflux
    #         # timeres_eflux_err[iobs] = \
    #         #     obs_snap.reconstructed_flux_datasets[skymodel.name]\
    #         #     .data.eflux_err
    #         timeres_eflux[iobs], timeres_eflux_err[iobs] = \
    #             obs_snap.reconstructed_flux_datasets[skymodel.name]\
    #             .data.model.spectral_model.energy_flux_error(
    #                 energy_min=energy_edges[0],
    #                 energy_max=energy_edges[-1]
    #                 )
    #     return (timeres_eflux, timeres_eflux_err)

    def livetime_integrate_onspectra(self, itbin_lo=0, itbin_hi=None):
        if itbin_hi is None:
            itbin_hi = len(self.obs_snaps)
        timeintg_onspectrum =\
            list(self.obs_snaps.values())[itbin_lo].on_spectrum_dataset.copy()
        #  For process acceleration
        timeintg_onspectrum_stack = timeintg_onspectrum.stack
        for iobs, obs_snap in enumerate(
            list(self.obs_snaps.values())[itbin_lo:itbin_hi]
        ):
            if iobs > 0:
                timeintg_onspectrum_stack(obs_snap.on_spectrum_dataset)
        return timeintg_onspectrum

    def livetime_average_fluxspectra(self, skymodel, energy_edges,
                                     itbin_lo=0, itbin_hi=None):
        if itbin_hi is None:
            itbin_hi = len(self.obs_snaps)

        weighted_flux_sum = u.Quantity(
            np.zeros(len(energy_edges)-1),
            unit=u.Unit("cm-2 s-1 TeV-1")*u.s
            )
        # exposure_sum = np.zeros_like(energy_edges\\)
        livetime_sum = 0*u.s
        for iobs, obs_snap in enumerate(
            list(self.obs_snaps.values())[itbin_lo:itbin_hi]
        ):
            reco_fluxes = np.squeeze(
                obs_snap.reconstructed_flux_datasets[skymodel.name]
                .data.dnde.quantity
                )
            masked_reco_fluxes = ma.masked_array(
                reco_fluxes,
                mask=~(reco_fluxes > 0),
                fill_value=0*u.Unit("cm-2 s-1 TeV-1")
                )
            weighted_flux_sum +=\
                obs_snap.observation.observation_live_time_duration *\
                masked_reco_fluxes.filled(
                    fill_value=0*u.Unit("cm-2 s-1 TeV-1")
                    )
            # exposure_sum += obs_snap.exposure
            livetime_sum += obs_snap.observation.observation_live_time_duration

        logger.debug("Livetime: {0}".format(livetime_sum))
        logger.debug("Weighted flux: {0}".format(weighted_flux_sum))
        return weighted_flux_sum / livetime_sum

    def draw_fluxspectra(self,
                         ax: mpl.axes,
                         skymodel: SkyModel,
                         energy_edges,
                         colors,
                         itbin_lo=0,
                         itbin_hi=None,
                         title='Time-resolved spectrum',
                         reference_label='Average model'
                         ):
        """This method is used to draw the flux spectra for the given sky model
        from the `gammapy.modeling.models` module.
        It uses matplotlib to plot the reconstructed flux datasets of
        the sky model over certain energy edges defined
        by astropy units of measure.
        Each flux data point is depicted with unique colors
        representing different segments of time resolved data.

        Parameters:
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot
            The axes upon which the data will be plotted.
        skymodel: gammapy.modeling.models.SkyModel
            The sky model whose reconstructed flux datasets will be plotted.
        energy_edges: astropy Quantity array_like
            The astropy Quantity array describing the energy bin edges over
            which the fluxes will be plotted.
        colors: array_like
            The colors to use for each ObsSnap in the scatter plot.
        itbin_lo: int, default=0
            The lower time bin index to limit a set of ObsSnaps to be plotted.
        itbin_hi: int, default=None
            The upper time bin index to limit a set of ObsSnaps to be plotted.
            If None, the length of obs_snaps will be used.
        title: str, default='Time-resolved spectrum'
            The title for the graph.

        Returns:
        -------
        None: just updates the provided axes parameter with a new scatter plot.
        """

        if itbin_hi is None:
            itbin_hi = len(self.obs_snaps)
        FLUX_UNIT = u.Unit("TeV / (s cm2)")
        flux_min = 1
        flux_max = 1e-30
        for iobs, obs_snap in enumerate(
            list(self.obs_snaps.values())[itbin_lo:itbin_hi]
        ):
            # Get the reconstructed flux from ObsSnap
            reco_fluxes = np.squeeze(
                obs_snap.reconstructed_flux_datasets[skymodel.name]
                .data.e2dnde.quantity
                )
            reco_flux_errs = np.squeeze(
                obs_snap.reconstructed_flux_datasets[skymodel.name]
                .data.e2dnde_err.quantity
                )
            logger.debug('reconstructed flux: %s', repr(reco_fluxes))
            logger.debug('reconstructed flux error: %s', repr(reco_flux_errs))

            masked_reco_fluxes = ma.masked_array(
                reco_fluxes.to_value(FLUX_UNIT),
                mask=~((reco_fluxes > 0 * FLUX_UNIT) &
                       (reco_fluxes > reco_flux_errs))
                )
            masked_reco_flux_errs = ma.masked_array(
                reco_flux_errs.to_value(FLUX_UNIT),
                mask=~(reco_flux_errs > 0*FLUX_UNIT)
                )
            logger.debug('Masked reconstructed flux: %s',
                         repr(masked_reco_fluxes))
            logger.debug('Masked reconstructed flux error: %s',
                         repr(masked_reco_flux_errs))
            energy_refs = np.sqrt(energy_edges[:-1]*energy_edges[1:])
            logger.debug('reference energy: %s', repr(energy_refs))
            if (not all(masked_reco_fluxes.mask)) and \
               (not all(masked_reco_flux_errs.mask)):
                ax.errorbar(
                    x=energy_refs,
                    y=masked_reco_fluxes,
                    xerr=(energy_refs-energy_edges[:-1],
                          energy_edges[1:]-energy_refs),
                    yerr=masked_reco_flux_errs,
                    fmt='o',
                    lw=1,
                    color=colors[iobs],
                    alpha=0.5,
                    capsize=2,
                    ms=3
                    )
                # Find appropriate minimum and maximum flux values to plot
                flux_min = min(
                    flux_min,
                    np.min(masked_reco_fluxes-masked_reco_flux_errs)
                    )
                flux_max = max(
                    flux_max,
                    np.max(masked_reco_fluxes+masked_reco_flux_errs)
                    )
        # Draw the reference model
        skymodel.spectral_model.plot(
            energy_bounds=(energy_edges[0], energy_edges[-1]),
            ax=ax,
            sed_type='e2dnde',
            color='gray',
            label=reference_label
            )
        logger.debug('Y-range: {0}-{1}'.format(flux_min, flux_max))
        ax.set_ylim(flux_min, flux_max)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel("Energy [TeV]")
        ax.set_ylabel(FLUX_UNIT.to_string())
        ax.legend()

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
