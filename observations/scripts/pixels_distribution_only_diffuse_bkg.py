from pbh_explorer.telescopes.fermi_lat.weekly import WeeklyDataAnalysis
from pbh_explorer.observations.ObsSnapFermiLAT import ObsSnapFermiLAT

weekly = WeeklyDataAnalysis()
dataset = weekly.get_map_dataset(map_type="hpx")

fermi = ObsSnapFermiLAT(obs_map_dataset=dataset)
fermi.evaluate_counts_distribution(nfakes=100, show_dist="pixels")
