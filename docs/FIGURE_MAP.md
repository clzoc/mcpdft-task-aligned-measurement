# Figure-to-data map

| Manuscript figure | Supplied PDF | Source and regeneration |
| --- | --- | --- |
| Fig. 1 | `guard15_method_schematic.pdf` | `analysis/method_schematic/`; diagnostic in `analysis/outputs/n2_10e8o_dqg_ftpbe_1pct_range/`; panel d uses equilibrium and scan source tables |
| Fig. 2 | `n2_rmse_errorbars_vertical.pdf` | Equilibrium `results/n2/`; `plot_equilibrium_errorbars.py` computes bootstrap statistics and `plot_equilibrium_errorbars_vertical.py` draws panels |
| Fig. 3 | `scan_energy_and_signed_error.pdf` | Scan signed mean/CIs and classical references; `plot_scan_energy_bars.py` |
| Fig. 4 | `scan_hamiltonian_energy_and_signed_error.pdf` | Same scan records, using the Hamiltonian energy |
| Fig. 5 | `co_rmse_errorbars_vertical.pdf` | Equilibrium `results/co_eq/`; same plotting scripts as N₂ |
| Fig. 6 | `ablation_60k_energy_error_decomposition.pdf` | Five arms at 60,000 shots; `plot_ablation_energy_decomposition.py` |
| Fig. S1 | `energy_high_budget_detail.pdf` | Existing equilibrium RMSE/CI table; `scripts/plot_energy_detail.py` |

`scripts/reproduce_figures.py` writes regenerated experimental figures to `generated/figures/` and leaves all supplied PDFs intact. Rendering can vary with installed fonts; the underlying numerical values are checked independently by `scripts/verify_data.py`.

