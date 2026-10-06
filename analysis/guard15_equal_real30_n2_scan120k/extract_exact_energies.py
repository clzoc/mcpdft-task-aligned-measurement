#!/usr/bin/env python3
"""Extract exact CAS reference energies (H and ftPBE F, Eh) for the 11 scan
geometries, using the same builder chain the campaign used
(physical_frame_design/experiment.py, kind='n2', cc-pVDZ, CAS(10e,8o),
FtPBEEnergyObjective grid_level=1). One geometry at a time; freed between
geometries to keep memory bounded."""
import json
import sys
from pathlib import Path
import gc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'outputs'/'physical_frame_design'))
import experiment as ex

BONDS=(.80,.90,1.00,1.10,1.25,1.45,1.60,1.80,2.00,2.20,2.50)
OUT=HERE/'figures'/'signed_errors'

def main():
    table={}
    for bond in BONDS:
        tag=f'n2_108_r{int(round(bond*100)):03d}'
        ex.SYSTEMS[tag]=dict(kind='n2',bond_length=bond,basis='cc-pvdz',
                             active_electrons=10,active_orbitals=8)
        selection,exact=ex.build_system(tag)
        objective=ex.j.FtPBEEnergyObjective(selection,grid_level=1)
        href,fref=ex.j._energy_values(selection,objective,exact.exact_d2,exact.exact_gamma)
        table[f'{bond:.2f}']=dict(bond_angstrom=bond,exact_h_eh=float(href),exact_f_eh=float(fref))
        print('EXACT',tag,f'H {href:.10f} F {fref:.10f}',flush=True)
        del selection,exact,objective
        gc.collect()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'scan_exact_reference_energies.json').write_text(json.dumps(dict(
        note='Exact CAS reference energies; H = Hamiltonian, F = nonlinear ftPBE (MC-PDFT), Eh; '
             'same builder chain as the scan campaign (CAS(10e,8o), cc-pVDZ, ftPBE grid_level=1)',
        energies=table),indent=2)+'\n')
    print('saved',OUT/'scan_exact_reference_energies.json',flush=True)

if __name__=='__main__':main()
