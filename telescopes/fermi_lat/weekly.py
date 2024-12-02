"all-sky用"
import os
import re
from pathlib import Path
from pbh_explorer.utils import PathChecker as PathC

from gammapy.data import GTI
import glob
import subprocess
from .recommended_parameter import get_config
from GtApp import GtApp
import gt_apps
import os
import shutil
from pathlib import Path
import json


from gammapy.datasets import MapDataset
from gammapy.maps import Map, MapAxis, WcsGeom, HpxGeom
from gammapy.irf import PSFMap, EDispKernelMap


def download_datafiles():
    pass

def combine_datafiles(lat_photon_weekly_files):
    # combine the weekly files into a single file
    #!ls lat_photon_weekly* > filelist.txt
    subprocess.run(["punlearn", "gtselect"])
    
    # Combining the files takes 10-15 minutes for the full dataset.
    gt_apps.filter["evclass"] = "INDEF"
    gt_apps.filter["evtype"]  = "INDEF"
    gt_apps.filter["infile"]  = "@filelist.txt"
    gt_apps.filter["outfile"] = "lat_alldata.fits"
    gt_apps.filter["ra"]      = 0
    gt_apps.filter["dec"]     = 0
    gt_apps.filter["rad"]     = 180
    gt_apps.filter["emin"]    = 30
    gt_apps.filter["emax"]    = 1000000
    gt_apps.filter["tmin"]    = "INDEF"
    gt_apps.filter["tmax"]    = "INDEF"
    gt_apps.filter["zmax"]    = 180
    gt_apps.filter["chatter"] = 4

script_dirpath = os.path.dirname(__file__)

class WeeklyDataAnalysis:
    def __init__(
        self,
        photon_filename=os.path.join(script_dirpath, "weekly/lat_photon_weekly_w843_p305_v001.fits"),
        spacecraft_filename=os.path.join(script_dirpath, "weekly/lat_spacecraft_weekly_w843_p310_v001.fits"),
        diffuse_bkg_filename=os.path.join(script_dirpath, "xml/diffuse_bkg.xml"),
        use_scratch_dir=False,
    ):
        self._config = get_config("All-Sky Point Source")
        
        self._ft1_filename = str( PathC.confirm_file_path(Path(photon_filename), "ft1") )
        self._ft2_filename = str( PathC.confirm_file_path(Path(spacecraft_filename), "ft2") )
        self._bkg_filename = str( PathC.confirm_file_path(Path(diffuse_bkg_filename), "diffuse background") )
        
        if use_scratch_dir:
            self._outdirname = "scratch"
        if not use_scratch_dir:
            # Extract numbers following ‘w’ in regular expressions
            match1 = re.search(r'w\d+', os.path.basename(photon_filename))
            match2 = re.search(r'w\d+', os.path.basename(spacecraft_filename))
            if not match1.group() == match2.group():
                return -1
            self._outdirname = match1.group()

    def run_fermi_pipeline(self, config=None, algorithm="ccube"):
        '''
        https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/LAT_weekly_allsky.html
        '''
        if not config:
            config = self._config
        
        try:
            print("\n----- Running fermi pipeline for weekly data -----")
            #-------------------------#
            # Photon events filtering #
            #-------------------------#
            #
            subprocess.run(["punlearn", "gtselect"])
            gt_apps.filter["infile"]  = self._ft1_filename
            gt_apps.filter["outfile"] = "gtselect.fits"
            gt_apps.filter["ra"]      = config["space"]["center"]["ra"]
            gt_apps.filter["dec"]     = config["space"]["center"]["dec"]
            gt_apps.filter["rad"]     = config["space"]["radius"]
            gt_apps.filter["zmax"]    = config["event"]["zmax"]
            gt_apps.filter["emin"]    = config["energy"]["emin"]
            gt_apps.filter["emax"]    = config["energy"]["emax"]
            gt_apps.filter["tmin"]    = config["time"]["tmin"]
            gt_apps.filter["tmax"]    = config["time"]["tmax"]
            gt_apps.filter["evclass"] = config["event"]["evclass"]
            gt_apps.filter["evtype"]  = config["event"]["evtype"]
            gt_apps.filter["chatter"]  = 4
            print(f"\n    Running gtselect...")
            gt_apps.filter.run()
            
            subprocess.run(["punlearn", "gtmktime"])
            gt_apps.maketime["evfile"]  = "gtselect.fits"
            gt_apps.maketime["outfile"] = "gtmktime.fits"
            gt_apps.maketime["scfile"]  = self._ft2_filename
            gt_apps.maketime["filter"]  = config["event"]["filter"]
            gt_apps.maketime["roicut"]  = config["event"]["roicut"]
            gt_apps.maketime["chatter"]  = 4
            print(f"\n    Running gtmktime...")
            gt_apps.maketime.run()
            
            subprocess.run(["punlearn", "gtbin"])
            gt_apps.evtbin["evfile"]    = "gtmktime.fits"
            gt_apps.evtbin["outfile"]   = f"gtmktime_{algorithm}.fits"
            gt_apps.evtbin["scfile"]    = self._ft2_filename
            gt_apps.evtbin["algorithm"] = algorithm.upper()
        
            gt_apps.evtbin["ebinalg"]  = "LOG"
            gt_apps.evtbin["emin"]     = config["energy"]["emin"]
            gt_apps.evtbin["emax"]     = config["energy"]["emax"]
            gt_apps.evtbin["enumbins"] = config["energy"]["enumbins"]
        
            gt_apps.evtbin["coordsys"] = "GAL"
            if algorithm == "ccube":
                gt_apps.evtbin["nxpix"]     = config["space"]["npix"]["x"]
                gt_apps.evtbin["nypix"]     = config["space"]["npix"]["y"]
                gt_apps.evtbin["binsz"]     = config["space"]["binsz"]
                gt_apps.evtbin["xref"]      = 0
                gt_apps.evtbin["yref"]      = 0
                gt_apps.evtbin["axisrot"]   = 0
                gt_apps.evtbin["proj"]      = config["space"]["proj"]
            if algorithm == "healpix":
                gt_apps.evtbin["hpx_ordering_scheme"] = config["space"]["hpx_ordering_scheme"]
                gt_apps.evtbin["hpx_order"]           = config["space"]["hpx_order"]
                gt_apps.evtbin["hpx_ebin"]            = True
                gt_apps.evtbin["hpx_region"]          = ""
            
            gt_apps.evtbin["chatter"] = 4
            print(f"\n    Running gtbin ({algorithm})...")
            gt_apps.evtbin.run()
        


            #------------------------------#
            # Instrument Response Function #
            #------------------------------#
            # Calculating the instrument livetime for the entire sky
            subprocess.run(["punlearn", "gtltcube"])
            gt_apps.expCube["evfile"]    = "gtmktime.fits"
            gt_apps.expCube["outfile"]   = "gtltcube.fits"
            gt_apps.expCube["scfile"]    = self._ft2_filename
            gt_apps.expCube["zmax"]      = config["event"]["zmax"]
            gt_apps.expCube["dcostheta"] = 0.025
            gt_apps.expCube["binsz"]     = config["space"]["binsz"]
            gt_apps.expCube["chatter"]   = 4
            print(f"\n    Running gtltcube...")
            gt_apps.expCube.run()

            # Convolving the livetime with the IRF
            subprocess.run(["punlearn", "gtexpcube2"])
            gtexpCube2 = GtApp("gtexpcube2", "Likelihood")
            gtexpCube2["infile"]    = "gtltcube.fits"
            gtexpCube2["outfile"]   = f"gtexpcube2_{algorithm}.fits"
            gtexpCube2["cmap"]      = f"gtmktime_{algorithm}.fits"
            gtexpCube2["irfs"]      = config["event"]["irf_name"]
            gtexpCube2["evtype"]    = config["event"]["evtype"]
            gtexpCube2["bincalc"]   = "EDGE"
            gtexpCube2["chatter"]   = 4
            print(f"\n    Running gtexpcube2...")
            gtexpCube2.run()
            
            #
            subprocess.run(["punlearn", "gtpsf"])
            gtpsf = GtApp("gtpsf", "Likelihood")
            gtpsf["expcube"]   = "gtltcube.fits"
            gtpsf["outfile"]   = "gtpsf.fits"
            gtpsf["irfs"]      = config["event"]["irf_name"]
            gtpsf["evtype"]    = config["event"]["evtype"]
            gtpsf["ra"]        = config["space"]["center"]["ra"]
            gtpsf["dec"]       = config["space"]["center"]["dec"]
            gtpsf["emin"]      = config["energy"]["emin"]
            gtpsf["emax"]      = config["energy"]["emax"]
            gtpsf["nenergies"] = config["energy"]["enumbins"]
            gtpsf["thetamax"]  = 30  # 固定値
            gtpsf["ntheta"]    = 300   # 固定値
            print(f"\n    Running gtpsf...")
            gtpsf.run()
        
        
            # diffuse background npred
            subprocess.run(["punlearn", "gtsrcmaps"])
            gtsrcmaps = GtApp("gtsrcmaps", "Likelihood")
            gtsrcmaps["scfile"]  = self._ft2_filename
            gtsrcmaps["expcube"] = "gtltcube.fits"
            gtsrcmaps["cmap"]    = f"gtmktime_{algorithm}.fits"
            gtsrcmaps["srcmdl"]  = self._bkg_filename
            gtsrcmaps["bexpmap"] = f"gtexpcube2_{algorithm}.fits"
            gtsrcmaps["outfile"] = f"gtsrcmaps_diffuse_{algorithm}.fits"
            gtsrcmaps["irfs"]    = config["event"]["irf_name"]
            gtsrcmaps["evtype"]  = config["event"]["evtype"]
            print(f"\n    Running gtsrcmaps...")
            gtsrcmaps.run()

            print("\n----- fermi pipeline for weekly-data are passed succesfully -----\n")
            
        except Exception as e:
            print(f"Error: {e}")

        else:
            # ディレクトリが存在しない場合は作成
            weekly_dir = os.path.join(script_dirpath, "weekly")
            outdir = os.path.join(weekly_dir, self._outdirname)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
                print(f"Directory created: {outdir}")
            else:
                print(f"Directory already exists: {outdir}")
                
            # 正常終了時にfitsファイルを保存用ディレクトリに移動
            temp_fits_files = glob.glob("*.fits")
            for temp_file in temp_fits_files:
                saved_file_path = os.path.join(outdir, temp_file)
                shutil.move(temp_file, saved_file_path)
                print(f"Created File have been moved: {temp_file} -> {saved_file_path}")
            
            # 正常終了時にft1,ft2,bkgファイルのエイリアスを作成して保存用ディレクトリに移動
            subprocess.run(["ln", "-s", self._ft1_filename], check=True)
            symlink_filename = os.path.basename(self._ft1_filename)
            saved_file_path = os.path.join(outdir, symlink_filename)
            shutil.move(symlink_filename, saved_file_path)
            print(f"Synbolic File have been moved: {symlink_filename} -> {saved_file_path}")
            subprocess.run(["ln", "-s", self._ft2_filename], check=True)
            symlink_filename = os.path.basename(self._ft2_filename)
            saved_file_path = os.path.join(outdir, symlink_filename)
            shutil.move(symlink_filename, saved_file_path)
            print(f"Synboric File have been moved: {symlink_filename} -> {saved_file_path}")
            
            subprocess.run(["ln", "-s", self._bkg_filename], check=True)
            symlink_filename = os.path.basename(self._bkg_filename)
            saved_file_path = os.path.join(outdir, symlink_filename)
            shutil.move(symlink_filename, saved_file_path)
            print(f"Synboric File have been moved: {symlink_filename} -> {saved_file_path}")
            
            # 正常終了時にconfig変数からテキストファイルを作成して保存用ディレクトリに移動
            temp_file = "config.txt"
            with open(temp_file, "w") as file:
                json.dump(config, file, indent=4, default=str)
            print(f"Configuration saved to {temp_file}")
            saved_file_path = os.path.join(outdir, temp_file)
            shutil.move(temp_file, saved_file_path)
            print(f"Created File have been moved:\n {temp_file} -> {saved_file_path}")

        finally:
            # 一時ファイルが残っていれば現在のディレクトリにある .fits ファイルを検索して削除
            # 削除していいかどうかストップしてきく
            fits_files = glob.glob("*.fits")
            print("\n")
            for fits_file in fits_files:
                os.remove(fits_file)
                print(f"Deleted: {fits_file}")

    def get_map_dataset(
        self,
        energy_axis=MapAxis.from_energy_bounds(
                        "100 MeV", "1 TeV", nbin=1, per_decade=True,
                        unit="MeV", name="energy"
                    ),
        map_type="wcs",
    ):
        weekly_dir = os.path.join(script_dirpath, "weekly")
        outdir = os.path.join(weekly_dir, self._outdirname)
        if not os.path.exists(outdir):
            print(f"Doesn't exist {outdir}")
            print("please run 'run_fermi_pipeline()' function before execute this function")
            return -1
        
        if map_type == "wcs":
            algorithm = "ccube"
            counts_filename = os.path.join(outdir, f"gtmktime_{algorithm}.fits")
            counts = Map.read(counts_filename)
            # エネルギー軸以外は一緒のgeomを作成
            new_energy_axis_geom = WcsGeom(
                counts.geom.wcs,
                npix=counts.geom.npix,
                cdelt=counts.geom._cdelt,
                crpix=counts.geom._crpix,
                axes=[energy_axis]
            )
        elif map_type == "hpx":
            algorithm = "healpix"
            counts_filename = os.path.join(outdir, f"gtmktime_{algorithm}.fits")
            counts = Map.read(counts_filename)
            # エネルギー軸以外は一緒のgeomを作成
            new_energy_axis_geom = HpxGeom(
                nside=counts.geom.nside,
                nest=counts.geom.nest,
                frame=counts.geom.frame,
                region=counts.geom.region,
                axes=[energy_axis]
            )
        else:
            print('sssssssssssssssss')
            return -1
        counts = counts.interp_to_geom(new_energy_axis_geom)
        
        background_filename = os.path.join(outdir, f"gtsrcmaps_diffuse_{algorithm}.fits")
        background = Map.read(background_filename)
        background = background.interp_to_geom(new_energy_axis_geom)

        exposure_filename = os.path.join(outdir, f"gtexpcube2_{algorithm}.fits")
        exposure = Map.read(exposure_filename)
        # for some reason the WCS definitions are not aligned...
        #exposure.geom._wcs = counts.geom.wcs

        psf_filename = os.path.join(outdir, "gtpsf.fits")
        psf = PSFMap.read(psf_filename, format="gtpsf")
        # reduce size of the PSF
        #psf = psf.slice_by_idx(slices={"rad": slice(0, 130)})

        edisp = EDispKernelMap.from_diagonal_response(
            energy_axis=counts.geom.axes["energy"],
            energy_axis_true=exposure.geom.axes["energy_true"],
        )

        #mask_safe = counts.geom.boundary_mask(width="0.2 deg")
        mask_safe = None

        gti = GTI.read(self._ft1_filename)
        
        return MapDataset(
            models=None, #ここをゼロにすると、backgroundで渡したmapのカウントがそのままnpredになる
            counts=counts,
            background=background,
            exposure=exposure,
            psf=psf,
            edisp=edisp,
            mask_safe=mask_safe,
            gti=gti
        )

    def create_xml_from_models(self):
        pass
        
    def unbinned_likelihood(self):
        pass
        
    def binned_likelihood(self):
        pass


if __name__ == "__main__":
    from fermilat_recommended_parameter import get_config
    from pbh_explorer.utils.path_checker import  confirm_file_path

    ft1_filename = "/Users/omooon/pbh-search/fermi/data/weekly/lat_photon_weekly_w843_p305_v001.fits"
    ft2_filename = "/Users/omooon/pbh-search/fermi/data/weekly/lat_spacecraft_weekly_w843_p310_v001.fits"
    bkg_filename = "/Users/omooon/pbh-search/fermi/data/background/diffuse.xml"

    save_dirname = get_weekly_date_info(ft1_filename)
    print(save_dirname)


    config = get_config("All-Sky Point Source", binsz=1)
    print(config)

    '''
    run_fermi_pipeline(
        ft1_filename,
        ft2_filename,
        bkg_filename,
        config,
        save_dirname,
        algorithm="healpix"
    )
    '''
    dataset = get_dataset(save_dirname, map_type="hpx")
    print(dataset)

    data = []
    
    dataset.fake()
    print(dataset)
    data.append(dataset.counts.slice_by_idx({"energy": slice(2, 5)}).data)

    dataset.fake()
    print(dataset)
    data.append(dataset.counts.slice_by_idx({"energy": slice(2, 5)}).data)
    
    dataset.fake()
    print(dataset)
    data.append(dataset.counts.slice_by_idx({"energy": slice(2, 5)}).data)
    
    dataset.fake()
    print(dataset)
    data.append(dataset.counts.slice_by_idx({"energy": slice(2, 5)}).data)
    
    plot_counts_histogram(data, log_scale="log")
    
    '''
    

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    # %matplotlib inline
    import matplotlib.pyplot as plt
    from IPython.display import display
    from gammapy.data import EventList
    from gammapy.datasets import Datasets, MapDataset
    from gammapy.irf import EDispKernelMap, PSFMap
    from gammapy.maps import Map, MapAxis, WcsGeom
    from gammapy.modeling import Fit
    from gammapy.modeling.models import (
        Models,
        PointSpatialModel,
        PowerLawNormSpectralModel,
        PowerLawSpectralModel,
        SkyModel,
        TemplateSpatialModel,
        create_fermi_isotropic_diffuse_model,
    )

    # Spatial model (point source at Galactic coordinates l=0, b=0)
    spatial_model = PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic")

    # Spectral model (Power-law with specific parameters)
    spectral_model = PowerLawSpectralModel(
        index=2.7,
        amplitude="5.8e-10 cm-2 s-1 TeV-1",
        reference="100 GeV"
    )

    # Define the sky model
    source = SkyModel(
        spectral_model=spectral_model,
        spatial_model=spatial_model,
        name="source-gc"
    )

    # Create the Models object
    models = Models([source])

    # Create the MapDataset with the defined models and other dataset properties
    dataset = MapDataset(
        models=models,
        counts=dataset.counts,
        background=dataset.background,
        exposure=dataset.exposure,
        psf=dataset.psf,
        edisp=dataset.edisp,
        name="fermi-dataset"
    )
    print(dataset)

    dataset.fake()
    print(dataset)

    
    '''
