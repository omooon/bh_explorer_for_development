from pbh_explorer.fermi_lat.weekly import WeeklyDataAnalysis

weekly = WeeklyDataAnalysis(use_scratch_dir=True)
dataset = weekly.get_map_dataset()

print(dataset)
print(dataset.gti)
