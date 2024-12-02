import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u

# logger setting
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel("INFO")
logger.addHandler(handler)
logger.setLevel("INFO")

def get_config(
    template_name,
    binsz=1,
    above_galactic_plane=True,
    glon=0,
    glat=45,
    gwidth=10,
):
    """
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data_Exploration/Data_preparation.html
    で推奨される解析方法に従う、これによって保存されるファイルの命名規則もここで与える

    シミュレーションIRFファイルの命名規則を設定するクラス

    Parameters
    ----------
    template_name : Optional[str], default=None
        時間範囲または観測年（例: "2024"や"2020-2024"）

        
    """
    recommendations = {
        "event":{},
        "space":{},
        "energy":{},
        "time":{}
    }
    recommendations['event']['evtype']   = 3
    recommendations['event']['zmax']     = 90
    recommendations['event']['evclass']  = 128
    recommendations['event']['irf_name'] = "P8R3_SOURCE_V3"
    recommendations['event']['roicut']   = "no"
    recommendations['event']['filter']   = "(DATA_QUAL>0)&&(LAT_CONFIG==1)"

    recommendations["space"]["unit"] = u.degree
    recommendations["energy"]["unit"] = u.MeV
    recommendations["time"]["unit"] = u.second


    # 銀河座標系 (Galactic Coordinates) に基づく領域の分割
    #
    #                           銀河緯度 (b)
    #                               +90°
    #                                ↑
    #                                |
    #         銀河面上領域 (Above Galactic Plane)
    #           10° <= b <= 90° (中心: b = 50°)
    #         ================================
    #                                |
    #     銀河面領域 (Galactic Plane Region)
    #          -10° <= b <= +10° (中心: b = 0°)
    #         ================================
    #                                |
    #         銀河面下領域 (Below Galactic Plane)
    #         -90° <= b <= -10° (中心: b = -50°)
    #                                |
    #                               -90°
    #
    #   銀河経度 (l): 0° から 360° の全範囲をカバー
    #   - 各領域は銀河緯度 (b) によって分割される
    #   - 銀河面 (Galactic Plane): |b| <= 10°
    #   - 銀河面上 (Above Galactic Plane): 10° < b <= 90°
    #   - 銀河面下 (Below Galactic Plane): -90° <= b < -10°

    if template_name == "Galactic Point Source":
        cen_gal = SkyCoord(l=0 * u.degree, b=0 * u.degree, frame='galactic')
        cen_icrs = cen_gal.transform_to('icrs')
        GLON = cen_gal.l.deg
        GLAT = cen_gal.b.deg
        RA  = cen_icrs.ra.deg
        DEC = cen_icrs.dec.deg

        GLON_WIDTH = 360
        GLAT_WIDTH = 20
        RADIUS = 180
        RESOLUTION = binsz

    elif template_name == "Off-plane Point Source":
        if above_galactic_plane:
            cen_gal = SkyCoord(l=0 * u.degree, b=50 * u.degree, frame='galactic')
        elif not above_galactic_plane:
            cen_gal = SkyCoord(l=0 * u.degree, b=-50 * u.degree, frame='galactic')     
        cen_icrs = cen_gal.transform_to('icrs')
        GLON = cen_gal.l.deg
        GLAT = cen_gal.b.deg
        RA  = cen_icrs.ra.deg
        DEC = cen_icrs.dec.deg

        GLON_WIDTH = 360
        GLAT_WIDTH = 80
        RADIUS = 180
        RESOLUTION = binsz
    
    elif template_name == "All-Sky Point Source":
        cen_gal = SkyCoord(l=0 * u.degree, b=0 * u.degree, frame='galactic')
        cen_icrs = cen_gal.transform_to('icrs')
        GLON = cen_gal.l.deg
        GLAT = cen_gal.b.deg
        RA  = cen_icrs.ra.deg
        DEC = cen_icrs.dec.deg

        GLON_WIDTH = 360
        GLAT_WIDTH = 180
        RADIUS = 180
        RESOLUTION = binsz

    elif template_name == "Region Point Source":
        cen_gal = SkyCoord(l=glon * u.degree, b=glat * u.degree, frame='galactic')
        cen_icrs = cen_gal.transform_to('icrs')
        GLON = cen_gal.l.deg
        GLAT = cen_gal.b.deg
        RA  = cen_icrs.ra.deg
        DEC = cen_icrs.dec.deg
        
        GLON_WIDTH = gwidth
        GLAT_WIDTH = gwidth
        RADIUS = gwidth / np.sqrt(2)
        RESOLUTION = binsz 

    else:
        return -1

    recommendations['space']['center'] = \
        {
            'l':GLON,
            'b':GLAT,
            'ra':RA,
            'dec':DEC
        }
    recommendations['space']['width'] = \
        {
            'lon':GLON_WIDTH,
            'lat':GLAT_WIDTH
        }
    recommendations['space']['radius'] = RADIUS
    recommendations['space']['rotation'] = 0
    recommendations['space']['binsz'] = RESOLUTION
    recommendations['space']['npix'] = \
        {
            'x': int(recommendations['space']['width']['lon']\
                            /recommendations['space']['binsz']),
            'y': int(recommendations['space']['width']['lat']\
                            /recommendations['space']['binsz'])
        }
    recommendations['space']['proj']  = "CAR"
    
    recommendations['space']['hpx_ordering_scheme'] = "RING"
    nside_values = [2**i for i in range(13)] # NSIDEのリスト（2^0 から 2^12まで）
    resolutions = [hp.nside2resol(nside, arcmin=True) / 60 for nside in nside_values] # 各NSIDEに対応する解像度を計算
    idx = np.absolute(np.array(resolutions) - recommendations['space']['binsz']).argmin()
    recommendations['space']['hpx_nside'] = nside_values[idx]
    recommendations['space']['hpx_order'] = hp.nside2order(nside_values[idx])

    recommendations['energy']['emin'] = 100
    recommendations['energy']['emax'] = 1000000
    recommendations['energy']['bins_per_dec'] = 2
    recommendations['energy']['enumbins'] = \
        int(
            np.ceil(
                np.log10(recommendations['energy']['emax']) \
                / np.log10(recommendations['energy']['emin']) \
                * recommendations['energy']['bins_per_dec']
            )
        )
    
    recommendations['time']['tmin'] = "INDEF"
    recommendations['time']['tmax'] = "INDEF"

    return recommendations
