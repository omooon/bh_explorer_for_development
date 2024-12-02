#!/usr/bin/env python
import sys
import numpy as np
from pathlib import Path, PosixPath
from IPython.display import display
import json
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from gammapy.maps import WcsGeom, HpxGeom, RegionGeom, MapAxis
from gammapy.data import Observation, GTI, FixedPointingInfo
from gammapy.datasets import (
    Datasets,
    MapDataset,
    MapDatasetOnOff,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset,
    ObservationEventSampler
)
from gammapy.makers import (
    MapDatasetMaker,
    SpectrumDatasetMaker,
)
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    TemplateSpectralModel,
    PowerLawSpectralModel,
    PointSpatialModel,
    SkyModel,
    Models,
    FoVBackgroundModel,
)
from gammapy.stats import cash
from regions import CircleSkyRegion
from gammapy.estimators import FluxPointsEstimator
from gammapy.irf.io import load_irf_dict_from_file

from logging import getLogger, StreamHandler

# Logger
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)


class ObsSnap:
    def __init__(
            self,
            observation: Observation = None,
            # https://docs.gammapy.org/dev/api/gammapy.data.Observation.html#gammapy.data.Observation
            energy_axis: MapAxis = MapAxis.from_energy_bounds(
                0.01, 100.0, 20,
                unit="TeV", name="energy"
                ),
            energy_axis_true: MapAxis = MapAxis.from_energy_bounds(
                0.005, 200, 50,
                unit="TeV", name="energy_true"),
            binsz: float = 0.02,
            width: tuple = (2, 2),
            center_coordinates: SkyCoord = SkyCoord(
                0*u.deg, 0*u.deg, frame="icrs"
                ),
            on_region_radius: Angle = Angle("0.11 deg"),
            hpx_nside=None,
            map_dataset_maker=MapDatasetMaker(
                selection=["background", "edisp", "psf", "exposure"]
                ),
            spectrum_dataset_maker=SpectrumDatasetMaker(
                selection=["exposure", "edisp", "background"]
                ),
            outdir=Path('.')
            ):
        '''ObsSnap holds information on a short-term observation.
        Here "short-term" means the IRF can be considered to be constant.'''
        self.observation = observation
        self.energy_axis = energy_axis
        self.energy_axis_true = energy_axis_true
        self.binsz = binsz
        self.width = width
        self.frame = "icrs"
        self.proj = "CAR"
        self.center_coordinates = center_coordinates
        self.on_region_radius = on_region_radius
        self.map_dataset_maker = map_dataset_maker
        self.spectrum_dataset_maker = spectrum_dataset_maker
        self.sim_models = None
        self.spectral_fit_result = {}
        self.reconstructed_flux_datasets = {}
        #  Key: Models, Value: FluxPointsDataset
        self.reconstructed_flux = {}
        self.max_likelihood = None
        self.outdir = outdir
        if not self.outdir.is_dir():
            self.outdir.mkdir(parents=True, exist_ok=True)

        logger.debug(self.observation)
        if self.observation is None:
            self.geom = None
            self.oncount_geom = None
            self.on_spectrum_dataset = None
            self.onoff_spectrum_dataset = None
            self.map_dataset = None
        else:
            if hpx_nside is not None:
                logger.debug(self.observation.target_radec)
                logger.debug('DISK({lon},{lat},{rad})'.format(
                    lon=self.observation.target_radec.ra.degree,
                    lat=self.observation.target_radec.dec.degree,
                    rad=self.width[0]/np.sqrt(2)))
                self.geom = HpxGeom.create(
                    nside=hpx_nside,
                    nest=True,
                    region='DISK({lon},{lat},{rad})'.format(
                        lon=self.observation.target_radec.ra.degree,
                        lat=self.observation.target_radec.dec.degree,
                        rad=self.width[0]/np.sqrt(2)
                        ),
                    skydir=(self.observation.target_radec.ra.degree,
                            self.observation.target_radec.dec.degree),
                    binsz=self.binsz,
                    width=self.width,
                    frame=self.frame,
                    axes=[self.energy_axis],
                )
                logger.debug('Target HealPix: {0}'.format(
                    self.geom.to_image()
                    .coord_to_pix(coords=self.observation.target_radec)
                    ))
            else:
                self.geom = WcsGeom.create(
                    skydir=self.center_coordinates,
                    # skydir=(self.observation.target_radec.ra.degree,
                    #        self.observation.target_radec.dec.degree),
                    binsz=self.binsz,
                    width=self.width,
                    frame=self.frame,
                    proj=self.proj,
                    axes=[self.energy_axis],
                    )

            self.oncount_geom = RegionGeom.create(
                region=CircleSkyRegion(
                    center=self.center_coordinates,
                    # center=self.observation.target_radec,
                    radius=self.on_region_radius
                    ),
                axes=[self.energy_axis],
                )
            empty_map = MapDataset.create(
                geom=self.geom,
                energy_axis_true=self.energy_axis_true,
                name="empty"
                )
            empty_spectrum = SpectrumDataset.create(
                geom=self.oncount_geom,
                energy_axis_true=self.energy_axis_true,
                name="empty"
                )
            self.on_spectrum_dataset = self.spectrum_dataset_maker.run(
                empty_spectrum,
                self.observation
                )
            self.map_dataset = \
                self.map_dataset_maker.run(empty_map, self.observation)

    def set_sim_models(
            self,
            photon_distributions=[],
            delta_times=[],
            emin=1*u.GeV
            ):
        models = [
            FoVBackgroundModel(dataset_name="dataset-mcmc")
            ]
        if len(photon_distributions) != len(delta_times):
            logger.error('''Length of photon_distributions
                        and delta_times does not match!!''')
            return 1
        for photon_distribution, dt in zip(photon_distributions, delta_times):
            if photon_distribution.spec_hist:
                logger.debug('Spectrum is given as a histogram.')
                arrival_spectrum = \
                    photon_distribution.spectrum[0]\
                    / (4.*np.pi*(photon_distribution.position["distance"]
                                 .to(u.cm)**2))\
                    / dt /\
                    (photon_distribution.spectrum[1][1:]
                     - photon_distribution.spectrum[1][:-1])
                energies = np.sqrt(
                    photon_distribution.spectrum[1][:-1]
                    * photon_distribution.spectrum[1][1:])
            else:
                logger.debug('Spectrum is given as a graph.')
                arrival_spectrum = \
                    photon_distribution.spectrum[0]\
                    / (4.*np.pi*(photon_distribution.position["distance"]
                       .to(u.cm)**2))\
                    / dt
                energies = photon_distribution.spectrum[1]
            models.append(
                SkyModel(
                    spatial_model=PointSpatialModel(
                        lon_0=photon_distribution.position["RA"],
                        lat_0=photon_distribution.position["DEC"],
                        frame="icrs"
                        ),
                    spectral_model=TemplateSpectralModel(
                        energy=energies[energies >= emin],
                        values=arrival_spectrum[energies >= emin],
                        interp_kwargs={'fill_value': 0}
                        ),
                    # https://docs.gammapy.org/1.1/user-guide/model-gallery/spectral/plot_template_spectral.html#template-spectral-model
                    name=photon_distribution.name
                )
            )
        self.sim_models = Models(models)

    def simulate(self):
        obssampler = ObservationEventSampler(
            dataset_kwargs={
                'energy_axis_true': self.energy_axis_true,
                'energy_axis': self.energy_axis,
                'spatial_width': self.width[0] * u.deg,
                'spatial_bin_size': self.binsz * u.deg,
                }
        )

        self.observation = obssampler.run(
            observation=self.observation,
            models=self.sim_models,
            dataset_name='dataset-mcmc'
            )
        self.map_dataset = \
            self.map_dataset_maker.run(
                self.map_dataset,
                self.observation
                )
        self.on_spectrum_dataset = self.spectrum_dataset_maker.run(
            self.on_spectrum_dataset,
            self.observation
            )
        self.onoff_spectrum_dataset = \
            SpectrumDatasetOnOff.from_spectrum_dataset(
                dataset=self.on_spectrum_dataset,
                acceptance=1,
                acceptance_off=3
                )
        self.onoff_spectrum_dataset.fake(
            npred_background=self.on_spectrum_dataset.npred_background()
            )

        # Derive the possible maximum likelihood
        self.max_likelihood = cash(
            n_on=self.on_spectrum_dataset.counts,
            mu_on=self.on_spectrum_dataset.counts
            ).sum()
        # https://docs.gammapy.org/1.1/api/gammapy.stats.cash.html?highlight=gammapy%20stats%20cash#gammapy.stats.cash

    def test_simulate(self):
        # Define simulation parameters
        model_simu = PowerLawSpectralModel(
            index=3.0,
            amplitude=2.5e-12 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1 * u.TeV,
        )
        model = SkyModel(
            spatial_model=PointSpatialModel(
                lon_0=0 * u.deg,
                lat_0=0.4 * u.deg,
                frame="icrs"
                ),
            spectral_model=model_simu,
            name="source"
            )

        # Set simulation models
        self.sim_models = Models([
            model,
            FoVBackgroundModel(dataset_name="dataset-mcmc")
            ])

        # Simulate observation
        self.simulate()

        # Check that the simulation was successful
        assert self.map_dataset is not None
        assert self.on_spectrum_dataset is not None
        assert self.onoff_spectrum_dataset is not None

    def simulate_spectrum(self):
        '''Simulate to make a count spectrum in a certain sky region following
        (https://docs.gammapy.org/dev/tutorials/analysis-1d/
        spectrum_simulation.html)'''
        # ON region
        self.on_spectrum_dataset\
            = self.map_dataset.to_spectrum_dataset(
                on_region=self.oncount_geom.region
            )

        # OFF region
        self.onoff_spectrum_dataset\
            = self.onoff_map_dataset.to_spectrum_dataset(
                on_region=self.oncount_geom.region
            )
        # self.onoff_spectrum_dataset = \
        #     SpectrumDatasetOnOff.from_map_dataset(
        #         dataset=self.map_dataset,
        #         acceptance=1,
        #         acceptance_off=3
        #         )
        # self.onoff_spectrum_dataset = \
        #     SpectrumDatasetOnOff.from_spectrum_dataset(
        #         dataset=self.on_spectrum_dataset,
        #         acceptance=1,
        #         acceptance_off=3
        #         )
        # self.onoff_spectrum_dataset.fake(
        #     npred_background=self.on_spectrum_dataset.npred_background()
        #     )
        #  self.exposure = self.on_spectrum_dataset.exposure

        # Derive the possible maximum likelihood
        self.max_likelihood = cash(
            n_on=self.on_spectrum_dataset.counts,
            mu_on=self.on_spectrum_dataset.counts
            ).sum()
        # https://docs.gammapy.org/1.1/api/gammapy.stats.cash.html?highlight=gammapy%20stats%20cash#gammapy.stats.cash

    def map(self):
        '''Simulate to make maps following
        (https://gammapy.github.io/gammapy-recipes/_build/html/notebooks/
        mcmc-sampling-emcee/mcmc_sampling.html#Simulate-an-observation).
'''
        self.map_dataset.models = self.sim_models
        self.map_dataset.fake()
        self.onoff_map_dataset\
            = MapDatasetOnOff.from_map_dataset(
                dataset=self.map_dataset,
                acceptance=1,
                acceptance_off=3
                )

    def fit_spectrum(self, fit_models, energy_edges):
        on_spectrum = self.on_spectrum_dataset.copy()
        on_datasets = Datasets()
        on_datasets.append(on_spectrum)
        on_datasets.models = fit_models

        # Fitting
        fit = Fit()
        self.spectral_fit_result[fit_models.name] = fit.run(on_datasets)
        logger.info(self.spectral_fit_result[fit_models.name])

        fpe = FluxPointsEstimator(
            energy_edges=energy_edges,
            source="blackhole",
            selection_optional="all"
            )
        flux_points = fpe.run(datasets=on_datasets)
        display(flux_points.to_table(sed_type="dnde", formatted=True))
        self.reconstructed_flux_datasets[fit_models.name] = FluxPointsDataset(
            data=flux_points,
            models=on_datasets.models
            )

        COUNT_FLUX_UNIT = u.Unit('1 / (s cm2)')
        ENERGY_FLUX_UNIT = u.Unit('TeV / (s cm2)')
        flux_val, flux_err\
            = self.reconstructed_flux_datasets[fit_models.name].models[0]\
            .spectral_model.integral_error(
                energy_min=energy_edges[0],
                energy_max=energy_edges[-1]
                )
        eflux_val, eflux_err\
            = self.reconstructed_flux_datasets[fit_models.name].models[0]\
            .spectral_model.energy_flux_error(
                energy_min=energy_edges[0],
                energy_max=energy_edges[-1]
                )
        self.reconstructed_flux[fit_models.name] = {
                'count_flux': {
                    'unit': COUNT_FLUX_UNIT.to_string(),
                    'value': flux_val.to_value(COUNT_FLUX_UNIT),
                    'error': flux_err.to_value(COUNT_FLUX_UNIT)
                },
                'energy_flux': {
                    'unit': ENERGY_FLUX_UNIT.to_string(),
                    'value': eflux_val.to_value(ENERGY_FLUX_UNIT),
                    'error': eflux_err.to_value(ENERGY_FLUX_UNIT)
                }
            }

    def test_statistics(self, test_models):
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

    def dump_json(self, path_jsonfile=None, overwrite=False):
        """Writes a dictionary representation of
this object's non-function properties to a JSON file.

Parameters:
----------
path_jsonfile : str
    The path of the JSON file to write to.
"""
        if path_jsonfile is None:
            path_jsonfile = self.outdir / 'obs_snap.json'
        data_dict = {}
        for kmem, vmem in vars(self).items():
            #  Custom conversions
            if kmem == 'observation':
                path_obs = self.outdir / 'observation.gdf'
                if path_obs.exists() and not overwrite:
                    logger.error(
                        'Observation file {0} already exists!! Skipping.'
                        .format(path_obs)
                        )
                vmem.write(
                    path=path_obs,
                    format='gadf',
                    include_irfs=True,
                    overwrite=overwrite
                    )
                data_dict[kmem] = str(path_obs)
                data_dict['pointing']\
                    = vmem.get_pointing_icrs(vmem.tmid).to_string('dms')
            elif kmem in ('energy_axis', 'energy_axis_true'):
                data_dict[kmem] = str(vmem)
            elif kmem in (
                'map_dataset',
                'onoff_map_dataset',
                'on_spectrum_dataset',
                'onoff_spectrum_dataset'
            ):
                if vmem is not None:
                    path_dataset = self.outdir / '{0}.gdf'.format(kmem)
                    vmem.write(path_dataset, overwrite=overwrite)
                    data_dict[kmem] = str(path_dataset)
                else:
                    data_dict[kmem] = None
            elif kmem in ('center_coordinates', ):
                data_dict[kmem] = vmem.to_string('dms')
            elif kmem in ('on_region_radius'):
                data_dict[kmem] = vmem.to_string()
            elif kmem == 'sim_models':
                data_dict[kmem] = vmem.to_dict()
            elif kmem == 'spectral_fit_result':
                data_dict[kmem] = {}
                for kresult, vresult in vmem.items():
                    data_dict[kmem][kresult] = str(vresult)
            elif kmem == 'reconstructed_flux_datasets':
                data_dict[kmem] = {}
                for kflux, vflux in vmem.items():
                    path_fluxdata = self.outdir / '{0}_{1}.fits'.format(
                        kmem,
                        kflux.replace(' ', '_')
                        )
                    vflux.write(
                        filename=path_fluxdata,
                        overwrite=overwrite,
                        format='fits'
                        )
                    data_dict[kmem][kflux] = str(path_fluxdata)
            elif kmem in ('geom', 'oncount_geom'):
                data_dict[kmem] = str(vmem)
            elif kmem in (
                'map_dataset_maker',
                'spectrum_dataset_maker',
            ):
                data_dict[kmem] = {'selection': vmem.available_selection}

            #  Generic conversion
            elif not callable(vmem) and not kmem.startswith("__"):
                if isinstance(vmem, Path) or isinstance(vmem, PosixPath):
                    data_dict[kmem] = str(vmem)
                else:
                    data_dict[kmem] = vmem
        logger.debug(data_dict)
        with open(path_jsonfile, 'w') as json_file:
            json.dump(data_dict, json_file, indent=2)

    def load_json(self, path_jsonfile=None):
        """Loads a dictionary representation of
this object's non-function properties from a JSON file.

    Parameters
    ----------
    path_jsonfile : str
        The path of the JSON file to load from.

    Returns
    -------
    None

"""
        if path_jsonfile is None:
            path_jsonfile = self.outdir / 'obs_snap.json'
        with open(path_jsonfile, 'r') as json_file:
            data_dict = json.load(json_file)
        for kmem, vmem in vars(self).items():
            if kmem == 'observation':
                try:
                    self.observation = Observation.read(data_dict[kmem])
                except KeyError:
                    irfs = load_irf_dict_from_file(data_dict[kmem])
                    gti = GTI.read(data_dict[kmem])
                    pointing = FixedPointingInfo(
                        fixed_icrs=SkyCoord(data_dict['pointing'])
                    )
                    self.observation = Observation.create(
                        tstart=gti.time_start,
                        tstop=gti.time_stop,
                        livetime=gti.time_sum,
                        pointing=pointing,
                        irfs=irfs
                        )
            elif kmem in ('energy_axis', 'energy_axis_true'):
                pass
            elif kmem in (
                'map_dataset',
            ):
                setattr(self, kmem, MapDataset.read(data_dict[kmem]))
                self.geom = getattr(self, kmem).geoms['geom']
                self.energy_axis\
                    = getattr(self, kmem).geoms['geom'].axes[0]
                self.energy_axis_true\
                    = getattr(self, kmem).geoms['geom_exposure'].axes[0]
            elif kmem in (
                'on_spectrum_dataset',
            ):
                setattr(self, kmem, SpectrumDataset.read(data_dict[kmem]))
                self.oncount_geom = getattr(self, kmem).geoms['geom']
            elif kmem in (
                'onoff_spectrum_dataset',
            ):
                setattr(self, kmem, SpectrumDatasetOnOff.read(data_dict[kmem]))
            elif kmem in ('center_coordinates', ):
                setattr(self, kmem, SkyCoord(data_dict[kmem]))
            elif kmem in ('on_region_radius'):
                setattr(self, kmem, Angle(data_dict[kmem]))
            elif kmem == 'sim_models':
                setattr(self, kmem, Models.from_dict(data_dict[kmem]))
            elif kmem == 'spectral_fit_result':
                setattr(self, kmem, {})
                for kresult, vresult in data_dict[kmem].items():
                    getattr(self, kmem)[kresult] = vresult
            elif kmem == 'reconstructed_flux_datasets':
                setattr(self, kmem, {})
                for kflux, vflux in data_dict[kmem].items():
                    getattr(self, kmem)[kflux] = FluxPointsDataset.read(vflux)
                logger.debug(self.reconstructed_flux_datasets)
                if self.reconstructed_flux_datasets['blackhole'] is None:
                    sys.exit(1)
            elif kmem in ('geom', 'oncount_geom'):
                pass
            elif kmem in (
                'map_dataset_maker',
            ):
                setattr(
                    self,
                    kmem,
                    MapDatasetMaker(
                        selection=["background", "edisp", "psf", "exposure"]
                    )
                )
            elif kmem in (
                'spectrum_dataset_maker',
            ):
                setattr(
                    self,
                    kmem,
                    SpectrumDatasetMaker(
                        selection=["exposure", "edisp", "background"]
                    )
                )

            #  Generic conversion
            elif not callable(vmem) and not kmem.startswith("__"):
                if isinstance(data_dict[kmem], str):
                    path_vmem = Path(data_dict[kmem])
                    if path_vmem.exists():
                        data_dict[kmem] = path_vmem
                setattr(self, kmem, data_dict[kmem])
