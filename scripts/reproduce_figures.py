"""Regenerate experimental plots in generated/, leaving archived figures intact."""
from pathlib import Path
import importlib.util,shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated/figures'
OUT.mkdir(parents=True,exist_ok=True)
EQ=ROOT/'analysis/guard15_equal_real30'
SCAN=ROOT/'analysis/guard15_equal_real30_n2_scan120k'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

vertical=module('release_vertical',EQ/'plot_equilibrium_errorbars_vertical.py')
vertical.OUT=OUT
shutil.copy2(ROOT/'source-data/equilibrium_rmse_ci95.csv',OUT/'rmse_ci95.csv')
vertical.main()
scan=module('release_scan_plot',SCAN/'plot_scan_energy_bars.py')
scan.OUT=OUT
shutil.copy2(ROOT/'source-data/scan_reference_energies.json',OUT/'scan_exact_reference_energies.json')
shutil.copy2(ROOT/'source-data/scan_signed_mean_ci95.csv',OUT/'scan_signed_errors_mean_ci95.csv')
scan.main()
ablation=module('release_ablation',EQ/'plot_ablation_energy_decomposition.py')
ablation.EQ=OUT
ablation.main()
# The supplementary plot is a short script; redirect its destination by changing
# only the MS root while retaining the published input CSV.
detail=ROOT/'scripts/plot_energy_detail.py'
source=detail.read_text().replace("MS=Path(__file__).resolve().parents[1]", "MS=Path(__file__).resolve().parents[1]/'generated'")
data=ROOT/'generated/source-data';data.mkdir(exist_ok=True)
shutil.copy2(ROOT/'source-data/equilibrium_rmse_ci95.csv',data/'equilibrium_rmse_ci95.csv')
exec(compile(source,str(detail),'exec'),{'__file__':str(detail),'__name__':'__main__'})
print('Regenerated five experimental figures and Supplementary Fig. S1 in',OUT)

