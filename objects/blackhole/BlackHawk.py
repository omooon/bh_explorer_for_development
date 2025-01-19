#!/usr/bin/env python3

import os
import sys
import numpy as np
import numpy.ma as ma
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from astropy import constants as c
from astropy.table import QTable, Column
from astropy import units as u
from collections import defaultdict
from pbh_explorer.utils.PathChecker import confirm_directory_path, confirm_file_path

# Logger
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
loglevel = 'INFO'
handler.setLevel(loglevel)
logger.setLevel(loglevel)
logger.addHandler(handler)

class BlackHawk:
    def __init__(
        self, 
        top_path=Path(os.getenv('HOME'))/'blackhawk', 
        parameter_path=None
    ):
        # check Top_PATH
        self.TOP_PATH = confirm_directory_path(
            directory_path=top_path,
            message='BlackHawk-installed'
        )

        # set, check, and read parameter_path
        parameter_path\
            = self.TOP_PATH / 'parameters.txt' if parameter_path is None\
            else parameter_path
        self.PARAMETER_PATH\
            = confirm_file_path(
                file_path=parameter_path,
                message='BlackHawk parameter'
            )
        self._parameters = {}
        self._parameter_comments = {}
        self.read_parameter_file()

        logger.info('BlackHawk parameters:')
        for key, value in self._parameters.items():
            logger.info('  {0}: {1}'.format(key, value))
        self._radiation_spectra = {}
        self.COLUMNS = {'t': 'time', 'M': 'mass', 'a': 'reduced spin'}
        self.UNITS = {
            't': u.s,
            'M': u.g,
            'a': u.Unit(1),
            }
        
        # NOTE: BlackHawk_inst.x の実行には対応していない
        self.EXE_TOT_PATH = self.TOP_PATH / 'BlackHawk_tot.x'
        self.EXE_INST_PATH = self.TOP_PATH / 'BlackHawk_inst.x'
    
    @property
    def elapsed_time(self):
        return self._dts["t"]
    
    @property
    def delta_time(self):
        return self._dts["dt"]

    @property
    def time_to_evaporation(self):
        return self._dts["rt"]

    @property
    def mass_to_evaporation(self):
        return self._life_evolutions["M"]

    @property
    def spin_to_evaporation(self):
        return self._life_evolutions["a"]

    @property
    def charge_to_evaporation(self):
        # あるはずだけど見当たらない。。。
        # とりあえず、self._life_evolutions["a"]と同じなので代用
        return self._life_evolutions["a"]

    @property
    def diff_spectra(self):
        return self._radiation_spectra
        
    @property
    def integ_spectra(self):
        return self._radiation_spectra_tintegral

    def read_parameter_file(self, param_path=None):
        '''parameters.txt infomation reader function'''
        if param_path is None:
            param_path = self.PARAMETER_PATH
        with open(str(param_path), 'r') as param_file:
            lines = param_file.readlines()
            for line in lines:
                if len(line.replace(' ', '').replace('\n', '')) <= 0:
                    continue
                line_wo_comment, comment = line.split('#')[:2]
                # Removing comments
                param_name, param_val = line_wo_comment.split('=')
                self._parameters[
                    param_name
                    .replace(' ', '').replace('\n', '').replace('\t', '')]\
                    = param_val\
                    .replace(' ', '').replace('\n', '').replace('\t', '')
                self._parameter_comments[param_name] = comment

    def launch_tot(self):
        dir_prev = os.getcwd()
        os.chdir(self.TOP_PATH)
        cmd = [str(self.EXE_TOT_PATH), str(self.PARAMETER_PATH)]
        proc = subprocess.Popen(cmd)
        os.chdir(dir_prev)
        return proc.wait()

    def read_results(
        self, 
        result_path=None, 
        particles =['photon_primary', 'photon_secondary']
    ):
        if result_path is None:
            result_path\
                = self.TOP_PATH\
                / 'results'\
                / self._parameters['destination_folder']
            
        self._read_result_dts(result_path=result_path/'dts.txt')
        self._read_result_life_evolutions(
            result_path=result_path/'life_evolutions.txt'
            )
        for particle in particles:
            self._read_result_particle(
                result_path=result_path/f'{particle}_spectrum.txt',
                particle=particle
                )
        
        self._remove_secondary_anomalies( self._radiation_spectra )
        self._time_integrate_spectra()
        
    def _read_result_dts(self, result_path=None):
        if result_path is None:
            result_path\
                = self.TOP_PATH / 'results'\
                / self._parameters['destination_folder'] / 'dts.txt'
        if not result_path.is_file():
            logger.error(
                'File {0} does not exist!!'.format(result_path)
                )
        with open(result_path, 'r') as result_file:
            lines = result_file.readlines()
            for hline, line in enumerate(lines):
                if hline == 0:
                    logger.info(line)
                else:
                    line_split = [
                        s.replace(' ', '').replace('\t', '').replace('\n', '')
                        for s in line.split(' ') if len(s) > 0
                        ]
                    logger.debug(line_split)
                    if hline == 1:
                        self._dts = QTable(
                            names=[key for key in line_split],
                            units=[u.s, u.s]
                            )
                    elif len(line_split) > 0:
                        values = [float(s)*u.s for s in line_split]
                        self._dts.add_row(values)
        self._dts.add_column(
            np.flip(
                np.flip(self._dts['dt']).cumsum()
                ),
            name='rt'
        )
        logger.info(self._dts)

    def _read_result_life_evolutions(self, result_path=None):
        #  UNITS = {'t': u.s, 'M': u.g, 'a': u.Unit(1)}
        if result_path is None:
            result_path\
                = self.TOP_PATH / 'results'\
                / self._parameters['destination_folder']\
                / 'life_evolutions.txt'
        if not result_path.is_file():
            logger.error('File {0} does not exist!!'.format(result_path))
        n_iterations = None
        with open(result_path, 'r') as result_file:
            lines = result_file.readlines()
            for hline, line in enumerate(lines):
                if hline == 0:
                    logger.info(line)
                elif 'Total number of time iterations' in line:
                    n_iterations = int(line.split(':')[1])
                    break
            for iline, line in enumerate(lines[hline+1:]):
                if n_iterations is not None:
                    line_split = [
                        s.replace(' ', '').replace('\t', '').replace('\n', '')
                        for s in line.split(' ') if len(s) > 0
                        ]
                    print(line_split)
                    if 't' in line_split\
                       and 'M' in line_split\
                       and 'a' in line_split:  # Header line
                        self._life_evolutions = QTable(
                            [Column(
                                np.full(n_iterations, np.nan),
                                unit=self.UNITS[key]
                                ) for key in line_split],
                            names=[key for key in line_split],
                            )
                        break
            for jline, line in enumerate(lines[hline+iline+2:]):
                if n_iterations is not None\
                   and len(
                       line
                       .replace(' ', '').replace('\t', '').replace('\n', '')
                       ) > 0:
                    line_split = [
                        s.replace(' ', '').replace('\t', '').replace('\n', '')
                        for s in line.split(' ') if len(s) > 0
                        ]
                    for key, value in zip(
                        self._life_evolutions.keys(),
                        line_split
                    ):
                        self._life_evolutions[key][jline]\
                            = float(value) * self.UNITS[key]
        logger.info(self._life_evolutions)

    def _read_result_particle(self, result_path=None, particle=None):
        if result_path is None:
            result_path = self.TOP_PATH\
                / 'results'\
                / self._parameters['destination_folder']\
                / '{p}_spectrum.txt'.format(p=particle)
        if not result_path.is_file():
            logger.error('File {0} does not exist'.format(result_path))
            return 1

        with open(result_path, 'r') as result_file:
            lines = result_file.readlines()
            for iline, line in enumerate(lines):
                line_split = [
                    s.replace(' ', '').replace('\t', '').replace('\n', '')
                    for s in line.split(' ') if len(s) > 0
                    ]
                if iline == 0:
                    logger.info(line)
                elif iline == 1:
                    logger.debug(line_split)
                    columns = []
                    units = []
                    for js, s in enumerate(line_split):
                        columns.append(float(s)*u.GeV if js > 0 else s)
                        units.append(
                            (u.GeV*u.s*u.cm**3)**(-1) if js > 0 else u.s
                            )
                    self._radiation_spectra[particle] = QTable(
                        names=columns,
                        units=units
                        )
                elif len(line_split) > 0:
                    values = [float(s)*units[ks]
                              for ks, s in enumerate(line_split)]
                    self._radiation_spectra[particle].add_row(values)
        logger.info(self._radiation_spectra[particle])

    def _remove_secondary_anomalies(self, all_particles_spectra_table):
        '''
        BlackHawk2.3以前（2.3しか確認はしていないけど、、、）は、secondary成分と言っておきながらprimary成分も足されたtotalスペクトルなので、それを修正
        ついでに、primaryとsecondaryのエネルギー軸も揃えるようにした
        '''
        def get_energy_axis(spectrum_table):
            energy_unit = u.Quantity(spectrum_table.colnames[1:][1]).unit
            return np.array([
                                u.Quantity(colname).to(energy_unit).value
                                        for colname in spectrum_table.colnames[1:]

                            ]) * energy_unit
            
        def get_spectrum_axis(spectrum_table, irow):
            spectrum_data = np.array([])
            spectrum_unit = spectrum_table[irow][1].unit
            for jcol in range(1, len(spectrum_table.colnames)):
                spectrum_data = np.append(spectrum_data, spectrum_table[irow][jcol].value)
            return spectrum_data * spectrum_unit

        # particle ごとにキーをグループ化して、各particleごとに処理
        grouped_keys = defaultdict(list)
        for key in all_particles_spectra_table.keys():
            particle, _ = key.split("_", 1)
            grouped_keys[particle].append(key)

        for particle, keys in grouped_keys.items():
            print(f"Processing secondary {particle} spectra anomalies:")
            
            # スペクトルテーブルを成分ごとに取得
            for key in keys:
                if key == f"{particle}_secondary":
                    total_spectra_table = all_particles_spectra_table[key]
                elif key == f"{particle}_primary":
                    primary_spectra_table = all_particles_spectra_table[key]
                    
            # 共通のエネルギー軸を作成（例えば最小値から最大値までの範囲でサンプリングを統一）
            common_energy_axis = np.union1d(
                                    get_energy_axis(total_spectra_table),
                                    get_energy_axis(primary_spectra_table)
                                )
                                
            # 補正テーブルデータを作成
            interped_time_table = QTable([total_spectra_table.columns[0]])
            for colname in common_energy_axis:
                interped_time_table[str(colname)] = None
                interped_time_table[str(colname)] = interped_time_table[str(colname)].astype(float)
                interped_time_table[str(colname)].unit = total_spectra_table[0][1].unit
            
            interped_primary_spectra_table = interped_time_table.copy()
            interped_secondary_spectra_table = interped_time_table.copy()
        
            # 各テーブルのフラックスデータをcommon_energy_axisに揃えるように各テーブル行ごとに値を補間する
            #for colname in enumerate(interped_time_table.colnames):
            for irow, colname in enumerate(interped_time_table.columns[0]):
                
                # 新しいエネルギー軸に対応するtotalスペクトルおよびprimaryスペクトルを作成
                interped_total_spectrum = np.interp(
                    common_energy_axis, # 新しいエネルギー軸
                    get_energy_axis(total_spectra_table), # 古いエネルギー軸
                    get_spectrum_axis(total_spectra_table, irow), # 古いエネルギー軸に対応するスペクトル
                )
                interped_primary_spectrum = np.interp(
                    common_energy_axis,
                    get_energy_axis(primary_spectra_table),
                    get_spectrum_axis(primary_spectra_table, irow)
                )
                
                # それぞれ、新しいエネルギー軸に対して元のエネルギー軸の範囲外の部分を 0 に設定
                interped_total_spectrum = np.where(
                    (common_energy_axis < get_energy_axis(total_spectra_table)[0]) | (common_energy_axis > get_energy_axis(total_spectra_table)[-1]),
                    0,
                    interped_total_spectrum
                )
                interped_primary_spectrum = np.where(
                    (common_energy_axis < get_energy_axis(primary_spectra_table)[0]) | (common_energy_axis > get_energy_axis(primary_spectra_table)[-1]),
                    0,
                    interped_primary_spectrum
                )
                
                # secondary成分の抽出
                secondary_spectrum = interped_total_spectrum - interped_primary_spectrum
                
                # 値をテーブルに詰めていく
                for j, colname in enumerate(common_energy_axis):
                    interped_primary_spectra_table[str(colname)][irow] = interped_primary_spectrum[j]
                    interped_secondary_spectra_table[str(colname)][irow] = secondary_spectrum[j]
                
            # 上記で作ったinterped_tableを、入力した元のキーの対応する粒子の各成分のテーブルに代入して更新する
            for key in keys:
                if key == f"{particle}_secondary":
                    all_particles_spectra_table[key] = interped_secondary_spectra_table
                elif key == f"{particle}_primary":
                    all_particles_spectra_table[key] = interped_primary_spectra_table
        
        return all_particles_spectra_table

    def _time_integrate_spectra(self):
        logger.info('Integrate spectra by dt...')
        self._radiation_spectra_tintegral = {}
        for particle, spectrum_table in self._radiation_spectra.items():
            logger.info(particle)
            # Copy the time column
            time_table = QTable([spectrum_table.columns[0]])
            for colname in spectrum_table.colnames:
                if 'time/energy' not in colname:  # Exclude the time column
                    time_table.add_column(
                        spectrum_table[colname] * self._dts['dt'],
                        name=colname
                        )
                    tintg_table = time_table
                    
            self._radiation_spectra_tintegral[particle] = tintg_table
            logger.info(self._radiation_spectra_tintegral[particle])
            
    def time_reduce_spectra(
        self,
        min_duration=0.1*u.s,
        min_deltamass=0.1
    ):
        logger.info('Spectrum table is to be reduced by grouping times...')
        reduced_spectra = {}
        # Make row groups for reduction of the spectrum table
        grouping_rtimes = np.full(len(self._life_evolutions), np.nan)
        dt_cum_temp = 0
        delta_frac_m = 0
        m_temp = self._life_evolutions['M'][-1]
        rt_temp = self._dts['rt'][-1]
        # Iterates over all rows from bottom
        for irow in reversed(range(len(self._life_evolutions))):
            dt_row = self._dts[irow]
            evolution_row = self._life_evolutions[irow]
            # Continue to group rows when the time duration < 0.1 s
            # and the mass change < 10%
            if dt_cum_temp < min_duration or delta_frac_m < min_deltamass:
                # rt_temp and m_temp is left as they are
                dt_cum_temp += dt_row['dt']
            else:  # Stop grouping and reset
                dt_cum_temp = 0
                rt_temp = dt_row['rt']  # Update to the current value
                m_temp = evolution_row['M']
            grouping_rtimes[irow] = rt_temp.value
            delta_frac_m = evolution_row['M']/m_temp - 1.
            
        # update
        dts_grouped = self._dts.group_by(grouping_rtimes)
        logger.debug(self._dts)
        logger.debug(dts_grouped)
        dts_rebinned_sum = dts_grouped.groups.aggregate(np.sum)
        dts_rebinned_min = dts_grouped.groups.aggregate(min)
        dts_rebinned_max = dts_grouped.groups.aggregate(max)
        logger.debug(dts_rebinned_sum)
        logger.debug(dts_rebinned_min)
        logger.debug(dts_rebinned_max)
        self._dts = dts_rebinned_sum[::-1]

        life_evolutions_grouped = self._life_evolutions.group_by(grouping_rtimes)
        life_evolutions_rebinned_sum = life_evolutions_grouped.groups.aggregate(np.sum)
        self._life_evolutions = life_evolutions_rebinned_sum[::-1]

        logger.info('Time grouping: \n{0}'.format(grouping_rtimes))
        for (particle, diff_spectrum_table), (_, integ_spectrum_table)\
            in zip(self._radiation_spectra.items(), self._radiation_spectra_tintegral.items()):
            logger.info(particle)

            diff_spectrum_grouped = diff_spectrum_table.group_by(grouping_rtimes)
            diff_spectrum_rebinned = diff_spectrum_grouped.groups.aggregate(np.sum)

            integ_spectrum_grouped = integ_spectrum_table.group_by(grouping_rtimes)
            integ_spectrum_rebinned = integ_spectrum_grouped.groups.aggregate(np.sum)
            
            # updata
            self._radiation_spectra[particle] = diff_spectrum_rebinned[::-1]
            self._radiation_spectra_tintegral[particle] = integ_spectrum_rebinned[::-1]

    def plot_evolution(self, ax_evol=None, fig_evol=None):
        # Check axes to draw the evolution
        if ax_evol is None:
            ax_evol = {}
            if fig_evol is None:
                logger.error('''Neither matplotlib.pyplot.figure or matplotlib.axes is provided for the evolution profile!!''')
                return 1
            else:
                ax_evol['mass'] = plt.subplot2grid(
                    (6, 2), (0, 0),
                    rowspan=2,
                    fig=fig_evol
                    )
                ax_evol['temperature'] = plt.subplot2grid(
                    (6, 2), (0, 1),
                    rowspan=2,
                    fig=fig_evol
                    )

        UNITS_PLOT = {
            'time': u.s,
            'mass': u.g,
            'temperature': u.GeV
            }
        SCALES = {
            'time': 'log',
            'mass': 'log',
            'temperature': 'log'
            }

        # Configure the axes
        for q, a in ax_evol.items():
            a.set_xlabel(
                'time to evaporation [{u}]'.format(u=UNITS_PLOT['time'])
                )
            a.set_ylabel(
                '{q} [{u}]'.format(q=q, u=UNITS_PLOT[q])
                )
            a.set_xscale(SCALES['time'])
            a.set_yscale(SCALES[q])

        # Quantity profiles
        profiles = {
            'time': self.time_to_evaporation,
            'mass': self.mass_to_evaporation,
            'temperature': self.temp_to_evaporation,
        }
        # Time series
        for q, a in ax_evol.items():
            a.plot(
                ma.array(
                    profiles['time'].to(UNITS_PLOT['time']),
                    mask=profiles['time'] < 1e-3*u.s
                    ),
                profiles[q].to(UNITS_PLOT[q]),
                )
