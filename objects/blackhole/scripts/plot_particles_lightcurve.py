class BlackHawkPlotter(BlackHawk):
    def __init__(
        self,
        top_path=Path(os.getenv('HOME'))/'blackhawk',
        parameter_path=None,
        time_window=100*u.s,
        initial_lifetimes=[1e+10, 1e+7, 1e+5, 1e+3]*u.s
    ):
        super().__init__(top_path, parameter_path)
        self.read_results(particles =['photon_primary', 'photon_secondary'])
        radiation_spectra_table, time_evolutions_table = self.time_window_spectra(time_window=time_window, initial_lifetimes=initial_lifetimes)


    def profile_particle_distributions(self, blackhole, spec_tables=None):
        if spec_tables is None:
            spec_tables = self.time_reduce_spectra()
        profiles = {}
        for particle_group, spec_table in spec_tables.items():
            logger.info(particle_group)
            energy_bins = np.array([
                u.Quantity(colname).to(u.GeV).value
                for colname in spec_table.colnames[1:]
                ]) * u.GeV
            for colname in spec_table.colnames[1:]:
                spectra_list.append((spec_table[colname] * u.cm**3).to(u.GeV**(-1)).value)  # Convert from 1/(GeV cm3) to count number/GeV
            spectra_array = np.array(spectra_list).T
            spectra_list = []
            for irow in reversed(range(len(spec_table))):
                spectrum_values = spectra_array[irow] * u.GeV**(-1)
                particle_graph = (spectrum_values, energy_bins)
                spectra_list.append(
                    ParticleDistribution(
                        particle=particle,
                        position=blackhole.position,
                        spectrum=particle_graph,  # particle_hist,
                        name=particle_group,
                        spec_hist=False,
                    )
                )
            profiles['{0}_spectrum'.format(particle_group)].append(
                ParticleDistributionProfile(
                    spectra=spectra_list,
                    energy_bins=energy_bins,
                    )
                )
        return profiles
    

    def plot_initial_mass_distribution(self):
        '''
        Plots the BHs mass funciton, that is to say the comoving density as a function of mass.
        The mass bins are represented by extended horizontal bars (this may fail for a Dirac distribution).
        '''
        pass


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

    def get_energy_axis(self, spectrum_table):
        energy_unit = u.Quantity(spectrum_table.colnames[1:][1]).unit
        return np.array([
                            u.Quantity(colname).to(energy_unit).value
                                    for colname in spectrum_table.colnames[1:]

                        ]) * energy_unit
    
    def get_spectrum_axis(self, spectrum_table, irow):
        spectrum_data = np.array([])
        spectrum_unit = spectrum_table[irow][1].unit

        for jcol in range(1, len(spectrum_table.colnames)):
            spectrum_data = np.append(spectrum_data, spectrum_table[irow][jcol].value)
        
        return spectrum_data * spectrum_unit

    def photon_energy_spectrum(
            self,
            radiation_spectra_table,
            time_evolutions_table,
            particle="photon",
        ):
        """
        与えられたテーブルを使って、photon_total_spectrumをプロットする関数
        (particlename)_secondaryのスペクトルは実際はトータルなのでそれを使ってプロット
        TODOテーブル数が多すぎたらエラー吐くようにする？
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cmap_time = plt.cm.get_cmap('rainbow')
        norm_time = mpl.colors.LogNorm(
            vmax=time_evolutions_table['time_to_evaporation'][0].to(u.s).value,
            vmin=time_evolutions_table['time_to_evaporation'][-1].to(u.s).value
        )
    
        total_spectrum_table = radiation_spectra_table[f'{particle}_secondary']
        for idx in range(len(total_spectrum_table)):
            # 光子のプロット
            lifetime_str = "{:.1e}".format(time_evolutions_table['time_to_evaporation'][idx].to(u.s).value)
            mass_str = "{:.1e}".format(time_evolutions_table['mass_to_evaporation'][idx].to(u.g).value)
            temperature_str = "{:.1e}".format(time_evolutions_table['temp_to_evaporation'][idx].to(u.GeV).value)
            show_color = cmap_time(norm_time(time_evolutions_table['time_to_evaporation'][idx].to(u.s).value))

            energy = self.get_energy_axis(total_spectrum_table)
            spectra = self.get_spectrum_axis(total_spectrum_table, idx)
            ax.loglog(energy, spectra, \
                    ls='-', lw=1.5, alpha=1, color=show_color, label=f"τ={lifetime_str} s, m={mass_str} g, T={temperature_str} GeV" )
            ax.set_xlabel("$E{\\rm\,\, (GeV)}$")
            ax.set_ylabel("${\\rm d}^2n/{\\rm d}t{\\rm d}E\,\, ({\\rm GeV}^{-1}\cdot{\\rm s}^{-1}\cdot{\\rm cm}^{-3})$")

        # titleの表示
        if self._parameters["hadronization_choice"] == "0":
            ax.set_title(f"Intrinsic Energy Spectrum (BBN Epoch for PYTHIA)\n")
        elif self._parameters["hadronization_choice"] == "1":
            ax.set_title(f"Intrinsic Energy Spectrum (BBN Epoch for HERWIG)\n")
        elif self._parameters["hadronization_choice"] == "2":
            ax.set_title(f"Intrinsic Energy Spectrum (Present Epoch for PYTHIA)\n")
        elif self._parameters["hadronization_choice"] == "3":
            ax.set_title(f"Intrinsic Energy Spectrum (Present Epoch for HAZMA)\n")
        elif self._parameters["hadronization_choice"] == "4":
            ax.set_title(f"Intrinsic Energy Spectrum (Present Epoch for HDMSpectra)\n")

        fig.canvas.draw_idle()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_xlim(1e-6, 1e+9)
        ax.set_ylim(1e+15, 1e+31)
        plt.grid()
        plt.tight_layout()
        plt.show()


    def plot_perticle_light_curve(
            self,
            radiation_spectra_table=None,
            time_evolutions_table=None,
        ):
        """
        与えられたテーブルを使って、photon_total_light_curveをプロットする関数
        """
        #########################################################
        # Plotting primary and secondary spectra (fixed energy) #
        #########################################################
        '''this block plots the desired primary particles spectra at a fixed energy,
        that is to say their time-dependant emission.'''

        # NOTE: primaryとsecondaryでenegy範囲が違うから、energy_idxだけだと場所がズレる

        if fixed == "energy":
            fig= plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(111)

            for i, particle in enumerate(particles_primary):
                if particles_primary[particle]:
                    ax.loglog(data_primary[i][1:,0], data_primary[i][1:,1+energy_idx], label=particle, linewidth=1, linestyle='--', alpha=0.5)
            for i, particle in enumerate(particles_secondary):
                if particle == "photon_secondary":
                    # これはphoton_secondaryと言いながらトータルスペクトルになっているのに注意
                    ax.loglog(data_secondary[i][1:,0], data_secondary[i][1:,1+energy_idx], label="photon_total", linewidth=2, linestyle='-', alpha=1)
                    # data_primary[i][1+time_idx,1:]のエネルギー範囲をsecondaryに合わせてからじゃないと引けない
                    #only_photon_secondary =  data_secondary[i][1+time_idx,1:] - data_primary[i][1+time_idx,1:]
                    #ax.loglog(data_secondary[i][0,1:], only_photon_secondary[i][1+time_idx,1:], label=particle, linewidth=1, linestyle='--', alpha=0.5)
                elif particles_secondary[particle] and particle != "photon_secondary":
                    ax.loglog(data_secondary[i][1:,0], data_secondary[i][1:,1+energy_idx], label=particle, linewidth=1, linestyle='--', alpha=0.5)

            ax.set_xlabel('$t{\\rm \,\, (s)}$')
            ax.set_ylabel('${\\rm d}^2n/{\\rm d}t{\\rm d}E\,\, ({\\rm GeV}^{-1}\cdot{\\rm s}^{-1}\cdot{\\rm cm}^{-3})$')
            if self._parameters["hadronization_choice"] == 0:
                ax.set_title("Hadronization Choice: 0 - BBN Epoch for PYTHIA\n")
            elif self._parameters["hadronization_choice"] == 1:
                ax.set_title("Hadronization Choice: 1 - BBN Epoch for HERWIG\n")
            elif self._parameters["hadronization_choice"] == 2:
                ax.set_title("Hadronization Choice: 2 - Present Epoch for PYTHIA\n")
            elif self._parameters["hadronization_choice"] == 3:
                ax.set_title("Hadronization Choice: 3 - Present Epoch for HAZMA\n")
            elif self._parameters["hadronization_choice"] == 4:
                ax.set_title("Hadronization Choice: 4 - Present Epoch for HDMSpectra\n")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.grid()
            plt.tight_layout()

            fig.canvas.draw_idle()
            plt.show()



    def ddplot_particle_energy_spectrum(self, temperature=None, particles_primary=None, particles_secondary=None):
        '''
        particles_primary, particles_secondaryは、順番変える。particles_primary, particles_secondary

        self.TOP_PATH/scripts/visualization_scripts/ にある plot_tot.py と plot_inst を元に作成

        temperature, mass, lifetimeのどれか指定
        '''
        self.read_result_particle(
            result_path=result_path/'photon_primary_spectrum.txt',
            particle='photon_primary'
            )
        self.read_result_particle(
            result_path=result_path/'photon_secondary_spectrum.txt',
            particle='photon_secondary'
            )
        self.time_integrate_spectra()

        #####################
        # Folder definition #
        #####################
        '''this block defines the paths toward the computed data and the localization of your favorite figure folder.'''
        result_folder = f"{self.TOP_PATH}/results/"
        destination_folder = result_folder + self._parameters["destination_folder"] + "/"


        ###################################
        # Plotting options (primary data) #
        ###################################
        '''this block defines which primary particle spectra will be plotted.'''

        # Primaryパラメータのデフォルト値
        default_particles_primary = {
            "photon_primary": 1,
            "gluons_primary": 0,
            "higgs_primary": 0,
            "W_primary": 0,
            "Z_primary": 0,
            "neutrinos_primary": 0,
            "electron_primary": 0,
            "muon_primary": 0,
            "tau_primary": 0,
            "up_primary": 0,
            "down_primary": 0,
            "charm_primary": 0,
            "strange_primary": 0,
            "top_primary": 0,
            "bottom_primary": 0,
            "graviton_primary": 0
        }

        # particles_primaryで指定されていないパラメータに対してデフォルト値を更新 （default_particles_primaryの順番を崩さないよう注意）
        if particles_primary is None:
            particles_primary = {**default_particles_primary, **(particles_primary or {})}
        #elif "photon_primary" not in particles_primary:
        #    particles_primary = {**default_particles_primary, **(particles_primary or {})}
        #    particles_primary["photon_primary"] =  0
        else:
            particles_primary = {**default_particles_primary, **(particles_primary or {})}

        # プロットする粒子のデータを取得
        data_primary = []
        for particle in particles_primary:
            if particles_primary[particle]:
                data_primary.append(np.genfromtxt(f"{destination_folder}{particle}_spectrum.txt", skip_header = 1))
            else:
                data_primary.append([])


        #####################################
        # Plotting options (secondary data) #
        #####################################
        '''this block defines which secondary particle spectra will be plotted.'''

        # Secondaryパラメータのデフォルト値
        epoch = self._parameters["hadronization_choice"]
        if epoch == 0 or 1: # spectrum for BBN epoch
            default_particles_secondary = {
                "photon_secondary": 1,
                "electron_secondary": 0,
                "muon_secondary": 0,
                "nu_e_secondary": 0,
                "nu_mu_secondary": 0,
                "nu_tau_secondary": 0,
                "pipm_secondary": 0,
                "K0L_secondary": 0,
                "Kpm_secondary": 0,
                "proton_secondary": 0,
                "neutron_secondary": 0
            }
        elif epoch == 2 or 3 or 4: # spectrum for present epoch
            default_particles_secondary = {
                "photon_secondary": 1,
                "electron_secondary": 0,
                "nu_e_secondary": 0,
                "nu_mu_secondary": 0,
                "nu_tau_secondary": 0,
                "proton_secondary": 0,
            }

        # particles_secondaryで指定されていないパラメータに対してデフォルト値を更新
        if particles_secondary is None:
            particles_secondary = {**default_particles_secondary, **(particles_secondary or {})}
        #elif "photon_secondary" not in particles_secondary:
        #    particles_secondary = {**default_particles_secondary, **(particles_secondary or {})}
        #    particles_secondary["photon_secondary"] =  0
        else:
            particles_secondary = {**default_particles_secondary, **(particles_secondary or {})}

        # プロットする粒子のデータを取得
        data_secondary = []
        for particle in particles_secondary:
            if particles_secondary[particle]:
                data_secondary.append(np.genfromtxt(f"{destination_folder}{particle}_spectrum.txt", skip_header = 1))
            else:
                data_secondary.append([])


        #######################################################
        # Plotting primary and secondary spectra (fixed time) #
        #######################################################
        '''this block plots the desired primary particles spectra at a fixed time,
        that is to say their instantaneous emission as a function of energy.'''

        fig= plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)

        if temperature: # GeV scale
            temperature_Kelvin = temperature.to(u.J) / c.k_B # conversion from GeV to Kelvins
            mass = ( (c.hbar * c.c**3) / (8 * np.pi * c.G * temperature_Kelvin * c.k_B) )
            self.read_result_life_evolutions()
            time_idx = np.abs(self._life_evolutions['M'].to(u.g).value - mass.to(u.g).value).argmin()
            lifetime = 400 * ( mass.to(u.g) / (10**10 * u.g) ) * u.s # 大体の寿命
        else:
            logger.error("Temperature parameter was not entered.")

        # map color 設定
        cmap_time = plt.cm.get_cmap('rainbow')
        norm_time = mpl.colors.LogNorm(
            vmax=10**10,
            vmin=10**0
        )
        plot_color = cmap_time(norm_time(lifetime.value))

        # 光子のプロット
        if particles_secondary["photon_secondary"]:
            # これはphoton_secondaryと言いながらトータルスペクトルになっているのに注意
            ax.loglog(data_secondary[0][0,1:], data_secondary[0][1+time_idx,1:], label="$\gamma_{total}$", ls='-', lw=1.5, alpha=1, color=plot_color )
        if particles_primary["photon_primary"]:
            ax.loglog(data_primary[0][0,1:], data_primary[0][1+time_idx,1:], label="$\gamma_{primary}$", ls='--', lw=1, marker='.', ms=5, alpha=0.5, color=plot_color )
        # data_primary[i][1+time_idx,1:]のエネルギー範囲をsecondaryに合わせてからじゃないと引けない
        #only_photon_secondary =  data_secondary[i][1+time_idx,1:] - data_primary[i][1+time_idx,1:]
        #ax.loglog(data_secondary[i][0,1:], only_photon_secondary[i][1+time_idx,1:], label=particle, linewidth=1, linestyle='--', alpha=0.5)
        #label="$\gamma_{secondary}$", ls='-.', lw=1, marker='.', ms=5, alpha=0.5

        # 残りの粒子のプロット
        for i, particle in enumerate(particles_primary):
            if particles_primary[particle] and particle != "photon_primary":
                ax.loglog(data_primary[i][0,1:], data_primary[i][1+time_idx,1:], label=particle, linewidth=1, linestyle=':', alpha=0.5, color=plot_color )
        for i, particle in enumerate(particles_secondary):
            if particles_secondary[particle] and particle != "photon_secondary":
                ax.loglog(data_secondary[i][0,1:], data_secondary[i][1+time_idx,1:], label=particle, linewidth=1, linestyle=':', alpha=0.5, color=plot_color )

        fig.canvas.draw_idle()
        
        # 軸ラベルの表示
        ax.set_xlabel("$E{\\rm\,\, (GeV)}$")
        ax.set_ylabel("${\\rm d}^2n/{\\rm d}t{\\rm d}E\,\, ({\\rm GeV}^{-1}\cdot{\\rm s}^{-1}\cdot{\\rm cm}^{-3})$")

        # titleの表示
        lifetime_str = "{:.1e}".format(lifetime.to(u.s).value)
        mass_str = "{:.1e}".format(mass.to(u.g).value)
        temperature_str = temperature.to(u.GeV).value
        if self._parameters["hadronization_choice"] == "0":
            ax.set_title(f"Intrinsic Energy Spectrum (BBN Epoch for PYTHIA)\nlifetime: {lifetime_str} s, mass: {mass_str} g, temperature: {temperature_str} GeV\n")
        elif self._parameters["hadronization_choice"] == "1":
            ax.set_title(f"Intrinsic Energy Spectrum (BBN Epoch for HERWIG)\nlifetime: {lifetime_str} s, mass: {mass_str} g, temperature: {temperature_str} GeV\n")
        elif self._parameters["hadronization_choice"] == "2":
            ax.set_title(f"Intrinsic Energy Spectrum (Present Epoch for PYTHIA)\nlifetime: {lifetime_str} s, mass: {mass_str} g, temperature: {temperature_str} GeV\n")
        elif self._parameters["hadronization_choice"] == "3":
            ax.set_title(f"Intrinsic Energy Spectrum (Present Epoch for HAZMA)\nlifetime: {lifetime_str} s, mass: {mass_str} g, temperature: {temperature_str} GeV\n")
        elif self._parameters["hadronization_choice"] == "4":
            ax.set_title(f"Intrinsic Energy Spectrum (Present Epoch for HDMSpectra)\nlifetime: {lifetime_str} s, mass: {mass_str} g, temperature: {temperature_str} GeV\n")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_xlim(1e-3, 1e+3)
        ax.set_ylim(1e+17, 1e+27)
        plt.grid()
        plt.tight_layout()
        plt.show()


        '''
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons
        import numpy as np

        # データの準備
        x = np.linspace(1, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.tan(x) / 10  # 値が大きくならないようにスケールダウン

        # 図の作成
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y1, label='Particle 1')
        line2, = ax.plot(x, y2, label='Particle 2')
        line3, = ax.plot(x, y3, label='Particle 3')
        plt.legend()

        # CheckButtonsの配置
        rax = plt.axes([0.8, 0.4, 0.15, 0.2])
        check = CheckButtons(rax, ['Particle 1', 'Particle 2', 'Particle 3'], [True, True, True])

        # チェックボタンのイベントハンドラ
        def toggle_visibility(label):
            if label == 'Particle 1':
                line1.set_visible(not line1.get_visible())
            elif label == 'Particle 2':
                line2.set_visible(not line2.get_visible())
            elif label == 'Particle 3':
                line3.set_visible(not line3.get_visible())
            plt.draw()

        check.on_clicked(toggle_visibility)

        plt.show()
        
        '''




        '''
        In [8]: In [113]: import numpy as np
        ...:      ...: import matplotlib.pyplot as plt
        ...:      ...:
        ...:      ...: x = np.arange(0, 50, 0.5)
        ...:      ...:
        ...:      ...: plt.plot(x, x, label="$\gamma_{total}$", ls='-', lw=1.5, marker=',', ms=5, alpha=1)
        ...:      ...: plt.plot(x+1, x, label="$\gamma_{primary}$", ls='--', lw=1, marker='.', ms=5, alpha=0.5)
        ...:      ...: plt.plot(x+2, x, label="$\gamma_{secondary}$", ls=':', lw=1, marker='.', ms=3, alpha=0.5)
        ...:      ...:
        ...:      ...: plt.plot(x+4, x, "^", ms=5, label="$g_{primary}$")
        ...:      ...: plt.plot(x+5, x, "^", ms=3, label="$g_{secondary}$")
        ...:      ...:
        ...:      ...: plt.plot(x+6, x, "<", ms=3, label="$h_{primary}$")
        ...:      ...: plt.plot(x+7, x, ">", ms=3, label="$h_{secondary}$")
        ...:      ...:
        ...:      ...: plt.plot(x+8, x, "s", ms=3, label="$e^\pm$")
        ...:      ...: plt.plot(x+9, x, "+", ms=4, label="$\mu^\pm$")
        ...:      ...: plt.plot(x+10, x, "x", ms=3, label="$\\tau^\pm$")
        ...:      ...: plt.plot(x+11, x, "D", ms=3, label="$u,\overline{u}$")
        ...:      ...: plt.plot(x+12, x, "d", ms=3, label="$d,\overline{d}$")
        ...:      ...: plt.plot(x+13, x, "1", ms=5, label="$c,\overline{c}$")
        ...:      ...: plt.plot(x+14, x, "2", ms=5, label="$s,\overline{s}$")
        ...:      ...: plt.plot(x+15, x, "3", ms=5, label="$t,\overline{t}$")
        ...:      ...: plt.plot(x+16, x, "4", ms=5, label="$b,\overline{b}$")
        ...:      ...: plt.plot(x+17, x, "h", ms=3, label="${\\rm G}$")
        ...:      ...: plt.plot(x+18, x, "H", ms=3, label="6")
        ...:      ...: plt.plot(x+19, x, "p", ms=3, label="p")
        ...:      ...: plt.plot(x+20, x, "|", ms=4, label="|")
        ...:      ...: plt.plot(x+21, x, "_", ms=4, label="dd")
        ...:      ...:
        ...:      ...: plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
        ...:      ...:
        ...:      ...: plt.xlim(0, 30)
        ...:      ...: plt.ylim(0, 10)
        Out[8]: (0.0, 10.0)

        In [9]: plt.show()

        '''



    def plot_bhawk(self):
        # スライダーでプロット、evaporationの図の対応する場所にポイント、左にsecondaryのチェックボックス、右にprimaryのチェックボックス
    
        fig = plt.figure(figsize=(12, 8))
        self.bhawk.plot_evolution(fig_evol=fig)
            
        # time to evaporation spectrum
        ax_spec = plt.subplot2grid(
            (6, 2), (2, 0),
            rowspan=3, colspan=2,
            fig=fig
            )
        
        # time to evaporation slider
        ax_evaporation = fig.add_axes(
            [0.3, 0.08, 0.4, 0.02],
            facecolor='lightsteelblue'
            )
        evaporation_slider = Slider(
            ax_evaporation,
            'log10(Time to evaporation [s])',
            self._time_to_evaporation_log[-1],
            self._time_to_evaporation_log[0],
            valstep=np.flip(self._time_to_evaporation_log),
            valinit=self._time_to_evaporation[0],
            initcolor='hotpink'
            )

        # time to evaporation colorbar
        cmap_time = plt.cm.get_cmap('rainbow')
        norm_time = mpl.colors.LogNorm(
            vmax=10**self._time_to_evaporation_to_zero_log[0],
            vmin=10**self._time_to_evaporation_to_zero_log[-1]
            )
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm_time, cmap=cmap_time),
            ax=ax_spec,
            label='Time to evaporation [s]'
        )
            
        # reset button
        ax_flux_reset = fig.add_axes([0.8, 0.08, 0.1, 0.02])
        button_flux_reset = Button(
            ax_flux_reset,
            'Reset',
            color='g', hovercolor='r'
            )

        def show_spectrum(val):
            '''Draws primary/secondary photon spectra of the blackhole (for 4pi sr).'''
            
            # 現在のスライダーに対応するインデックスの情報をそれぞれ取得
            ival = np.absolute(self._time_to_evaporation_log - val).argmin()
            ph_radiation1 = self._radiation_profiles['photon_primary'].spectra[ival]
            ph_radiation2 = self._radiation_profiles['photon_secondary'].spectra[ival]
            dt = self._delta_times[ival]
            t_to_evap = self._time_to_evaporation[ival]
            
            # スペクトルのプロット
            signal_models = self.create_short_term_signal_skymodels(ph_radiation1, ph_radiation2, dt, t_to_evap)
            signal_models.spectral_model.plot(
                ax=ax_spec,
                energy_bounds=(1e-6, 10000) * u.TeV,
                sed_type='dnde',
                ls='-', lw=1, marker='o', ms=2, alpha=1.0,
                color=cmap_time(norm_time(t_to_evap)),
                label='photon primary at {0:1.2E} s'.format(t_to_evap),
                )
            #signal_models.spectral_model.model2.plot(
            #    ax=ax_spec,
            #    energy_bounds=(1e-6, 10000) * u.TeV,
            #    sed_type='dnde',
            #    ls='-', lw=1, marker=',', ms=0, alpha=0.5,
            #    color=cmap_time(norm_time(t_to_evap)),
            #    label='photon_secondary at {0:1.2E} s'.format(t_to_evap),
            #    )
            # Add vertical dashed lines
            energy_axis = self._getEnergyAxis()
            emin = energy_axis.bounds.to(u.TeV).min().value
            emax = energy_axis.bounds.to(u.TeV).max().value
            ax_spec.axvline(x=emin,
                            color='gray',
                            linestyle='--',
                            )
            ax_spec.axvline(x=emax,
                            color='gray',
                            linestyle='--'
                            )
            ax_spec.legend()
            evaporation_slider.poly.set(facecolor=cmap_time(norm_time(t_to_evap)))
            fig.canvas.draw_idle()
            
        def reset_flux(event):
            evaporation_slider.reset()
            ax_spec.clear()

        evaporation_slider.on_changed(show_spectrum)
        button_flux_reset.on_clicked(reset_flux)
        
        plt.tight_layout()
        plt.show()
