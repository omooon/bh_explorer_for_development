import os
import sys
from pbh_explorer.fermi_lat.weekly import WeeklyDataAnalysis

weekly = WeeklyDataAnalysis(use_scratch_dir=True)
weekly.run_fermi_pipeline()
