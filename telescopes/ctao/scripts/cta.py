import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pathlib import Path
import os
import sys

from gammapy.data import EventList
from gammapy.maps import Map
from gammapy.irf import EDispKernelMap, PSFMap

import subprocess

from gammapy.modeling.models import (
    Models,
    PowerLawNormSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)
from gammapy.datasets import MapDataset
import gt_apps
from GtApp import GtApp
import subprocess
from pathlib import Path
import copy


from .config import FermiConfig
class FermiIRF:
    def __init__(
        self, 
        loglevel='INFO',
        **fov_params
    ):
        # irf configuration
        fov_params.setdefault("fov_lon", 0)
        fov_params.setdefault("fov_lat", 0)
        fov_params.setdefault("fov_width", 5)
        fov_params.setdefault("fov_binsz", 0.1)
        fov_params.setdefault("frame", "galactic")
        self.config = FermiConfig(**fov_params)
        
        # logger
        self.logger = self.config.getLogger(loglevel)
        self.IRF_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/fermi"

        # IRF
        self.EVCLASS = 128
        self.EVTYPE = 3
        self.IRFS = "P8R3_SOURCE_V3"
        self.ZMAX = 90
        self.IRF_ENERGY_MIN = self.config.ENERGY_MIN_TRUE
        self.IRF_ENERGY_MAX = self.config.ENERGY_MAX_TRUE
        self.IRF_ENUMBINS = int(np.ceil(np.log10(self.config.ENERGY_MAX_TRUE / self.config.ENERGY_MIN_TRUE) * self.config.ENERGY_BINS_PER_DEC_TRUE))
        self.IRF_LON_WIDTH = 5 * self.config.SPACE_LON_WIDTH
        self.IRF_LAT_WIDTH = 5 * self.config.SPACE_LAT_WIDTH
        self.IRF_PIXEL_PER_DEGREE = self.config.SPACE_BINSZ

        self.log_parameters()




class CTAIRF:
    pass
