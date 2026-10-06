"""Build manuscript tables and validate them against the frozen result records."""
from pathlib import Path
import csv, hashlib, json, shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MS=ROOT/'generated'
EQ=ROOT/'analysis/guard15_equal_real30'
SCAN=ROOT/'analysis/guard15_equal_real30_n2_scan120k'
OUT=ROOT/'source-data'
OUT.mkdir(exist_ok=True)
MS.mkdir(exist_ok=True)
BUDGETS=(30000,60000,120000,240000)
ARMS=('uniform','guard15_equal_mu0','guard15_equal','exclusive_guard15_mu0','exclusive_guard15_mu2')
LABEL={'uniform':'Uniform30','guard15_equal_mu0':r'Select15, $\mu=0$',
 'guard15_equal':r'Select15, $\mu=2$','exclusive_guard15_mu0':r'Complement15, $\mu=0$',
 'exclusive_guard15_mu2':r'Complement15, $\mu=2$'}
RECORDS={}
for system in ('n2','co_eq'):
 for b in BUDGETS:
  for arm in ARMS:
   vals=[]
   for r in range(8):
    path=EQ/'results'/system/arm/f'b{b}_r{r}.json'
    d=json.loads(path.read_text())
    assert d['status']=='optimal' and sum(d['counts'])==b
    assert d['lambda_radius']==1 and d['mu_ftpbe']==(2 if arm in ('guard15_equal','exclusive_guard15_mu2') else 0)
    assert d['identity_error_meh']<1e-7 and d['contraction_error']<1e-6
    assert d['equality_residual']<1e-6 and d['minimum_dqg_eigenvalue']>-2e-6
    if arm!='uniform': assert sum(d['fit_counts'])==b-7500
    vals.append(d)
   RECORDS[system,b,arm]=vals
  for left,right in [('guard15_equal','guard15_equal_mu0'),('exclusive_guard15_mu2','exclusive_guard15_mu0')]:
   for r in range(8):
    p=EQ/'results'/system
    with np.load(p/left/f'b{b}_r{r}.npz') as a,np.load(p/right/f'b{b}_r{r}.npz') as z:
     for key in ('values','indices','counts'): np.testing.assert_array_equal(a[key],z[key])

def rms(ds,key): return float(np.sqrt(np.mean([d[key]**2 for d in ds])))
def mean(ds,key): return float(np.mean([d[key] for d in ds]))

# Validate the figures' numerical source, including the RMS rather than mean norm.
ci=list(csv.DictReader((EQ/'figures/equilibrium_errorbars/rmse_ci95.csv').open()))
for row in ci:
 ds=RECORDS[row['system'],int(row['budget']),row['arm']]
 assert abs(rms(ds,row['metric'])-float(row['rmse']))<1e-10

copies={
 'equilibrium_rmse_ci95.csv':EQ/'figures/equilibrium_errorbars/rmse_ci95.csv',
 'equilibrium_stream_errors.csv':EQ/'figures/equilibrium_errorbars/stream_errors.csv',
 'equilibrium_bootstrap_method.json':EQ/'figures/equilibrium_errorbars/method.json',
 'equilibrium_protocol.json':EQ/'protocol.json',
 'paired_ablation_contrasts.csv':EQ/'ablation_contrasts.csv',
 'scan_summary.csv':SCAN/'summary.csv','scan_stream_errors.csv':SCAN/'per_stream.csv',
 'scan_protocol.json':SCAN/'protocol.json',
 'scan_signed_mean_ci95.csv':SCAN/'figures/signed_errors/scan_signed_errors_mean_ci95.csv',
 'scan_reference_energies.json':SCAN/'figures/signed_errors/scan_exact_reference_energies.json',
 'ablation_scaled_components.csv':EQ/'figures/equilibrium_errorbars/ablation_60k_energy_error_decomposition.csv',
}
for name,source in copies.items(): shutil.copy2(source,OUT/name)
for source in (SCAN/'figures/signed_errors').glob('*method*.json'):
 shutil.copy2(source,OUT/'scan_bootstrap_method.json')
source_dir=OUT/'source-snapshots'; source_dir.mkdir(exist_ok=True)
for prefix,folder,names in [('equilibrium',EQ,['experiment.py','plot_equilibrium_errorbars.py','plot_equilibrium_errorbars_vertical.py','plot_ablation_energy_decomposition.py']),('scan',SCAN,['scan.py','engine.py','plot_scan_energy_bars.py'])]:
 for name in names: shutil.copy2(folder/name,source_dir/(prefix+'_'+name))
for rel in json.loads((EQ/'protocol.json').read_text())['source_sha256']:
 source=ROOT/'analysis'/rel
 dest=source_dir/rel; dest.parent.mkdir(exist_ok=True,parents=True)
 shutil.copy2(source,dest)

# Small component records address the 1-RDM question without repeating an SDP.
components=[]
for (system,b,arm),ds in RECORDS.items():
 row=dict(system=system,budget=b,arm=arm,n=8)
 for key in ['gamma_error','h_gamma_meh','h_d2_meh','f_non_ontop_meh','f_on_top_meh']:
  row[key+'_rms']=rms(ds,key)
  row[key+'_mean']=mean(ds,key)
 for energy,ka,kb in [('H','h_gamma_meh','h_d2_meh'),('F','f_non_ontop_meh','f_on_top_meh')]:
  a=np.array([d[ka] for d in ds]); z=np.array([d[kb] for d in ds])
  row[energy+'_cross_second_moment']=float(2*np.mean(a*z))
  assert abs(np.mean(a*a)+np.mean(z*z)+row[energy+'_cross_second_moment']-rms(ds,'h_error_meh' if energy=='H' else 'f_error_meh')**2)<1e-5
 components.append(row)
with (OUT/'rdm_and_energy_components.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(components[0]));w.writeheader();w.writerows(components)

def table(caption,label,cols,heading,rows):
 return '\n'.join([r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\small',r'\begin{tabular}{'+cols+'}',r'\toprule',heading+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\end{table}',''])

si=r'''\documentclass[pdflatex,sn-nature]{sn-jnl}
\usepackage{amsmath,amssymb,booktabs,graphicx,placeins,xr-hyper}
\externaldocument{sn-article}
\renewcommand{\thepage}{S\arabic{page}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\thefigure}{S\arabic{figure}}
\raggedbottom
\begin{document}
\title[Supplementary Information]{Supplementary Information for Reducing Measurement Costs in Quantum--Classical Multiconfiguration Pair-Density Functional Theory via Task-Aligned Derandomized Shadow}
\author{\fnm{Zhanou} \sur{Liu}}
\author{\fnm{Yuhao} \sur{Chen}}
\author{\fnm{Yingjin} \sur{Ma}}
\author{\fnm{Xiao} \sur{He}}
\author{\fnm{Yuxin} \sur{Deng}}
\maketitle
\renewcommand{\thepage}{S\arabic{page}}

\section*{S1. Implementation details and shot accounting}\label{sec:si-settings}
All calculations use singlet CAS(10e,8o) states, cc-pVDZ and two frozen-core orbitals. The pair matrix has dimension 120 and trace 45; the active 1-RDM has dimension 16 and trace 10. Neither RDM error is divided by its trace. The same 30 real orbital rotations act on both spin sectors, with five active electrons of each spin. The pool is generated with seed 20260716 and is fixed across molecules, geometries, budgets and repetitions. The measured subspace has dimension 171 for equilibrium N$_2$ and 305 for CO. Ties in the backward deletion score are resolved by frame index.

For pilot outcomes $\mathbf y_s$, the regularized covariance used in selection is
\begin{align}
 \widetilde{\mathbf m}&=\frac{\sum_{s=1}^{500}\mathbf y_s+50\mathbf m_0}{550},\\
 \widetilde M&=\frac{\sum_{s=1}^{500}\mathbf y_s\mathbf y_s^T+50M_0}{550},\\
 \widehat\Sigma&=\widetilde M-\widetilde{\mathbf m}\widetilde{\mathbf m}^T+\epsilon I.
\end{align}
Here $(\mathbf m_0,M_0)$ are the exact first and second moments of the uniform determinant distribution with $(N_\alpha,N_\beta)=(5,5)$ in eight spatial orbitals, and $\epsilon$ is $10^{-8}$ times the largest diagonal entry before regularization. For example, the prior mean of a same-spin pair occupation is $5\times4/(8\times7)$ and that of an opposite-spin pair is $25/64$. These prior moments depend on the known particle sector, not on the CASCI wave function.

The independent affine coordinates satisfy the linear particle-number, spin and spatial-symmetry constraints. Off-diagonal pair-matrix entries carry a factor $\sqrt{2}$ in the Frobenius metric. An orthonormal basis for the row space of the stacked full-pool design is obtained by singular-value decomposition, retaining singular values above $10^{-10}$ times the largest. Selection is performed in this measured subspace. The backward deletion guard uses the eigenvalue threshold stated in Methods.

In the measurement-band SDP, both positive semidefinite error matrices have zero entries between different spin or spatial-symmetry sectors. For each error matrix $E$, the singlet relations are $E_{\alpha\alpha}=E_{\beta\beta}$, $U_A^TE_{\alpha\beta}U_A=E_{\alpha\alpha}$ and $U_A^TE_{\alpha\beta}U_S=0$, where $U_A$ and $U_S$ transform opposite-spin orbital pairs to antisymmetric and symmetric spatial combinations. These error matrices do not acquire the particle-number trace and contraction constraints of the physical RDM. All 120 occupation-pair rows per frame enter the fit: 1,800 rows for a subset and 3,600 for Uniform30. Pilot covariances are used in selection, while the measurement bands use the raw means.

The Select15 and complementary protocols retain only their own 15 pilot prefixes. Their fitted count is $B-7{,}500$, although all $B$ acquired shots are charged. Table~\ref{tab:shots} gives the integer allocations. Uniform30 uses all $B$ observations. Thus $60{,}000$ in the ablation figure denotes the total budget, not the post-pilot budget.
'''
si+=table(r'\textbf{Shot accounting.} Select15 and its complement each measure a 30-frame pilot of 500 shots per frame. Counts below apply to either 15-frame subset; the last column gives the no-pilot Uniform30 allocation.', 'tab:shots','rrrrr',r'Total $B$ & New/frame & Fitted/frame & Total fitted & Uniform/frame', [f'{b:,} & {(b-15000)//15:,} & {500+(b-15000)//15:,} & {b-7500:,} & {b//30:,}'+r' \\' for b in BUDGETS])
si+=r'''
\FloatBarrier
\section*{S2. Stream-level statistics and the 1-RDM}\label{sec:si-statistics}
Each independent repetition includes a fresh pilot, its resulting selection and the additional shots. Within a repetition, the per-frame random seed is $202610018100+104729(200+r)$ for $r=0,\ldots,7$; shared prefixes pair methods and nest budgets. Consequently, different budgets and geometries do not add independent repetitions. The eight complete repetitions are the resampling units for the plotted confidence intervals.

The bootstrap uses 100,000 resamples and seed 20261006 (20261007 for the separate CO equilibrium resampling). A common list of resampled stream identifiers is applied to the paired configurations. The figure-source tables contain the RMSE or signed mean and the 2.5th and 97.5th percentile bounds. There are no hypothesis tests or significance markers in the figures.

Table~\ref{tab:rdm-components} reports the error of the contracted 1-RDM and its effective one-electron contribution to the energy at 60,000 shots. The latter is $\langle h^{\mathrm{eff}},\widehat\gamma-\gamma_\star\rangle$. This effective Hamiltonian contribution includes the interaction with the frozen core. The MC-PDFT non-on-top term contains the molecular one-electron energy and the classical Coulomb energy evaluated from the total 1-RDM. The two additive decompositions of the total error are
\begin{align}
 \Delta E_H&=\langle h^{\mathrm{eff}},\Delta\gamma\rangle+\langle V,\Delta D\rangle,\\
 \Delta E_F&=\Delta E_{\mathrm C}+\Delta E_{\mathrm{ot}}.
\end{align}
For either decomposition $\Delta E=a+b$, the squared error obeys
\begin{equation}
 \overline{(\Delta E)^2}=\overline{a^2}+\overline{b^2}+2\overline{ab}.
\end{equation}
The component RMS values therefore need not add to the total RMSE. The accompanying component table records all three terms for every equilibrium configuration. To obtain the molecular one-electron errors, the two classical references were regenerated with the recorded orbital conventions and contracted with the saved fitted 1-RDMs. The effective one-electron errors reproduced the stored values to within $4\times10^{-11}$~m$E_{\mathrm h}$; no measurement or SDP fit was repeated.
'''
one_e=list(csv.DictReader((OUT/'molecular_one_electron_errors.csv').open()))
rows=[]
for s in ['n2','co_eq']:
 for a in ARMS:
  ds=RECORDS[s,60000,a]
  bare_rms=float(np.sqrt(np.mean([float(r['molecular_one_electron_error_mHa'])**2 for r in one_e if r['system']==s and r['arm']==a and int(r['budget'])==60000])))
  rows.append(('N$_2$' if s=='n2' else 'CO')+' & '+LABEL[a]+' & '+f'{rms(ds,"gamma_error"):.5f} & {rms(ds,"h_gamma_meh"):.2f} & {bare_rms:.2f} & {rms(ds,"f_non_ontop_meh"):.2f} & {rms(ds,"f_on_top_meh"):.2f}'+r' \\')
si+=table(r'\textbf{1-RDM and energy-component errors at 60,000 total shots.} N$_2$ uses $R=1.10$~\AA\ and CO uses $R=1.128$~\AA; both use cc-pVDZ and CAS(10e,8o). Entries are RMS errors across eight complete streams. $e_\gamma$ is an unnormalized Frobenius error; all other columns are in m$E_{\mathrm h}$. $H_{1}$ is the effective active-space one-electron term in the Hamiltonian, including its frozen-core interaction. $E_{1}$ is the molecular one-electron contribution to MC-PDFT, $\langle h,\gamma^{\mathrm{tot}}\rangle$. Complement15 is the set labelled Except-Select15 in the main figure.','tab:rdm-components','llrrrrr',r'Molecule & Protocol & $e_\gamma$ & $H_1$ & $E_1$ & $E_{\mathrm C}$ & $E_{\mathrm{ot}}$',rows)
si+=r'''
\FloatBarrier
\section*{S3. Bond-scan and matched-subset results}\label{sec:si-scan}
The scan uses 120,000 shots at each of 11 bond lengths: 0.80, 0.90, 1.00, 1.10, 1.25, 1.45, 1.60, 1.80, 2.00, 2.20 and 2.50~\AA. The same frame pool and stream-seed convention are used across geometries; the 1.10-\AA\ point reuses the corresponding equilibrium results. Resampled stream identifiers are shared across geometries when constructing the pointwise bootstrap intervals. The reference energy labelled Exact in the figures is the CASCI Hamiltonian energy for $H$, and the nonlinear ftPBE energy evaluated on the exact CASCI RDMs for $F$. It is not an exact solution of the full-basis electronic problem. Table~\ref{tab:scan-errors} gives the per-geometry RMSEs, complementing the signed mean errors in the main figures. Equal weighting over the 11 geometries and eight streams gives the scan-wide RMSEs quoted in Results. Cubic splines in the figures interpolate only the reference energies; no fitted spectroscopic parameters are inferred from them.
'''
scan=json.loads((SCAN/'summary.json').read_text())
rows=[]
for d in scan:
 rows.append(f'{d["bond_angstrom"]:.2f} & '+ ' & '.join(f'{d[k]:.3f}' for k in ['uniform_h_rmse_mHa','guard15_equal_h_rmse_mHa','uniform_f_rmse_mHa','guard15_equal_f_rmse_mHa'])+r' \\')
si+=table(r'\textbf{N$_2$ bond-scan energy RMSEs.} cc-pVDZ, CAS(10e,8o), 120,000 total shots per geometry and eight complete streams. Errors are in m$E_{\mathrm h}$ relative to the CASCI Hamiltonian energy or the ftPBE energy of its exact RDMs. Select15 uses $\mu=2$; Uniform30 uses $\mu=0$; both use $\lambda=1$.','tab:scan-errors','rrrrr',r'$R$ (\AA) & Uniform $H$ & Select15 $H$ & Uniform $F$ & Select15 $F$',rows)
rows=[]
for s in ['n2','co_eq']:
 for b in BUDGETS:
  for mu,a,z in [(0,'guard15_equal_mu0','exclusive_guard15_mu0'),(2,'guard15_equal','exclusive_guard15_mu2')]:
   left,right=RECORDS[s,b,a],RECORDS[s,b,z]
   rows.append(('N$_2$' if s=='n2' else 'CO')+f' & {b//1000} & {mu} & '+ ' & '.join(f'{v:.2f}' for v in [rms(left,'h_error_meh'),rms(right,'h_error_meh'),rms(left,'f_error_meh'),rms(right,'f_error_meh')])+r' \\')
si+=table(r'\textbf{Selected versus complementary frames at matched reconstruction weights.} N$_2$ at 1.10~\AA\ and CO at 1.128~\AA, cc-pVDZ, CAS(10e,8o). The total budget $B$ is in thousands of shots. Both subsets have 15 frames and identical pilot and fit-shot counts. Values are eight-stream energy RMSEs in m$E_{\mathrm h}$; $\lambda=1$ throughout.','tab:subset-comparison','lrrrrrr',r'Molecule & $B/10^3$ & $\mu$ & Select $H$ & Complement $H$ & Select $F$ & Complement $F$',rows)
si+=r'''
\FloatBarrier
\section*{S4. Noise-free recovery and the selection model}\label{sec:si-model}
As a reference check, full-pool reconstructions with exact occupation means and $(\lambda,\mu)=(1,2)$ give the following signed errors: N$_2$, $\Delta E_H=-4.63\times10^{-5}$ and $\Delta E_F=-1.65\times10^{-5}$~m$E_{\mathrm h}$; CO, $\Delta E_H=-9.20\times10^{-4}$ and $\Delta E_F=-2.53\times10^{-4}$~m$E_{\mathrm h}$. The unnormalized 2-RDM errors are $2.17\times10^{-4}$ and $7.50\times10^{-5}$, respectively. These tests use the same equilibrium geometries, basis, active space, real frame pool and energy functionals as the main calculations. They quantify the noise-free reference error of these full-pool fits; subset comparisons are evaluated separately through the finite-shot experiments.

The selection calculation models the covariance of a linear least-squares estimator in the measured subspace. The energy variance ratios retained in each pilot plan are $(n_0/n_S)\mathbf h^TC_S\mathbf h/(\mathbf h^TC_{S_0}\mathbf h)$ and the analogous expression for $\mathbf f$. Table~\ref{tab:proxy} compares their averages across the eight pilots with the observed squared-RMSE ratios. Both $\mu=0$ and $\mu=2$ reconstructions share the same selection model and observations. The comparison indicates how the final error changes when the measurement model is followed by the constrained fit; the local ratios are not fitted to the scoring references.
'''
rows=[]; proxy=[]
for s in ['n2','co_eq']:
 for b in BUDGETS:
  ds=RECORDS[s,b,'guard15_equal']; base=RECORDS[s,b,'uniform']; mu0=RECORDS[s,b,'guard15_equal_mu0']
  ph=np.mean([d['guard_proxy']['h'] for d in ds]);pf=np.mean([d['guard_proxy']['f'] for d in ds])
  obs=[(rms(x,k)/rms(base,k))**2 for x in [mu0,ds] for k in ['h_error_meh','f_error_meh']]
  rows.append(('N$_2$' if s=='n2' else 'CO')+f' & {b//1000} & '+' & '.join(f'{v:.3f}' for v in [ph,pf,*obs])+r' \\')
  proxy.append(dict(system=s,budget=b,proxy_h=float(ph),proxy_f=float(pf),mu0_h_mse_ratio=obs[0],mu0_f_mse_ratio=obs[1],mu2_h_mse_ratio=obs[2],mu2_f_mse_ratio=obs[3]))
si+=table(r'\textbf{Local variance ratios and observed MSE ratios relative to Uniform30.} N$_2$ at 1.10~\AA\ and CO at 1.128~\AA, cc-pVDZ, CAS(10e,8o); eight complete streams. Pilot-model ratios are averaged over streams. Observed ratios divide the squared RMSE of Select15 by that of Uniform30 at the same total budget.','tab:proxy','lrrrrrrr',r'Molecule & $B/10^3$ & Model $H$ & Model $F$ & $\mu{=}0$, $H$ & $\mu{=}0$, $F$ & $\mu{=}2$, $H$ & $\mu{=}2$, $F$',rows)
with (OUT/'proxy_vs_observed.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(proxy[0]));w.writeheader();w.writerows(proxy)
si+=r'''
\FloatBarrier
\section*{S5. Reading the energy-component visualization}\label{sec:si-components}
In the 60,000-shot ablation figure, the total opaque height is $\mathrm{RMSE}_X$. The common and unique parts are signed mean components, rescaled separately for each protocol and energy by $s_X=\mathrm{RMSE}_X/\overline{\Delta E_X}$. Hence the displayed heights satisfy
\begin{equation}
 s_X\overline{\Delta E_{\mathrm C}}+
 s_X\overline{\Delta E_{X,\mathrm{unique}}}=\mathrm{RMSE}_X.
\end{equation}
For $X=F$ the unique component is the on-top error; for $X=H$ it is the error in $E_H-E_{\mathrm C}$. Opposite-sign contributions are translucent behind the opaque total, so their extensions may exceed the total RMSE. The rescaling preserves the proportions and relative signs of the mean components but is not an additive decomposition of RMSE or variance. The unscaled means, component second moments and cross terms are available in the accompanying source data.

\section*{S6. Source-data organization}\label{sec:si-source}
The equilibrium and scan protocol files record the molecular settings, frame-pool seed, shot counts, solver settings and source hashes. Separate tables provide individual-stream errors, figure confidence intervals, paired ablation contrasts, contracted 1-RDM errors, energy components and the model-to-observed comparison. The public repository identified in Data availability contains these tables together with the original pilot records, fitted RDMs, computational modules and reproduction instructions. The six main-text figure files are copied without alteration from the supplied publication materials; the supplementary expanded view uses the same archived statistics. The 1.10-\AA\ scan records reuse the equilibrium records; the remaining 160 scan records correspond to separate fits.

\section*{S7. Energy-sensitivity diagnostic in the workflow figure}\label{sec:si-diagnostic}
Panel a of Fig.~\ref{fig:workflow} uses a separate deterministic RDM-perturbation calculation for singlet N$_2$ at 1.10~\AA, cc-pVDZ and CAS(10e,8o), with two doubly occupied core orbitals. Its reference is a CASSCF ground state; the orbitals are held fixed while the RDM is varied. The measurement benchmarks in panel d and the Results instead use CASCI in canonical restricted Hartree--Fock orbitals. Both calculations evaluate the ftPBE functional on a level-1 integration grid.

Starting from the CASSCF RDM, the diagnostic seeks lower and higher total MC-PDFT energies under $DQG$, particle-number, contraction and singlet constraints, with
\begin{align}
 \|D-D_\star\|_F&\leq0.0101\|D_\star\|_F,\\
 |E_H[D]-E_H[D_\star]|&\leq0.001\ E_{\mathrm h},\\
 |E_{\mathrm C}[D]-E_{\mathrm C}[D_\star]|&\leq0.005\ E_{\mathrm h}.
\end{align}
The reference norm is $\|D_\star\|_F=6.53462689$, so the absolute RDM radius is 0.06599973. The percentage printed in panel a is this relative Frobenius norm; the measurement-error plots use the unnormalized norm defined in Methods. This diagnostic uses no measurement shots and imposes no frame-observation constraints.

The numerical search uses iterative linear minimization with a bounded line search on the full nonlinear ftPBE energy. The upper bound on the common energy is imposed through its quadratic dependence on the 1-RDM; supporting tangent inequalities enforce the lower bound. The two reported feasible candidates have $(\Delta E_H,\Delta E_{\mathrm C},\Delta E_{\mathrm{ot}},\Delta E_F)=(-1.000,-5.000,-8.637,-13.637)$ and $(1.000,5.000,8.578,13.578)$~m$E_{\mathrm h}$, respectively. They illustrate the different energy sensitivities within the stated bounds; the search does not certify global extrema. The archived records contain the two RDMs, constraint diagnostics, reference energies and optimization histories.

The variance profile in panel b illustrates the backward selection procedure; the retained size $K=15$ is prescribed. The ellipse and marked positions in panel c illustrate the two-energy reconstruction geometry rather than a numerically traced feasible-set boundary. $F_{\mathrm{lin}}$ is the tangent of the total MC-PDFT energy, and the exact CAS point is used only to define scoring errors. Panel d uses the same equilibrium and scan statistics as the main Results.

\FloatBarrier
\section*{S8. Expanded energy-error views}\label{sec:si-detail}
Figure~\ref{fig:energy-detail} displays the high-budget energy data separately, with axis limits set by the confidence intervals in each panel. The numerical values and bootstrap intervals are identical to those in the main budget figures.
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/energy_high_budget_detail.pdf}
\caption{\textbf{Expanded views of the energy errors at 120,000 and 240,000 shots.} \textbf{a}, N$_2$ Hamiltonian RMSE at $R=1.10$~\AA. \textbf{b}, Total ftPBE MC-PDFT RMSE for the same N$_2$ calculations. \textbf{c}, CO Hamiltonian RMSE at $R=1.128$~\AA. \textbf{d}, Total ftPBE MC-PDFT RMSE for the same CO calculations. Both molecules use cc-pVDZ and CAS(10e,8o); errors are relative to the corresponding exact CASCI-based energies. Points summarize eight complete shot streams, with pointwise 95\% percentile bootstrap intervals from 100,000 resamples. Uniform30 samples all 30 frames equally with $\mu=0$. Select15 uses 500 pilot shots per frame, selects 15 frames and distributes the remaining budget equally, reusing the retained pilot outcomes. Both Select15 objectives use the same observations; $\mu$ weights the linearized total MC-PDFT term. All fits use $\lambda=1$, and both budgets include all 15,000 pilot shots. Horizontal offsets separate protocols at the same budget. Dashed lines mark 1.6~mHa.}
\label{fig:energy-detail}
\end{figure}
\end{document}
'''
(MS/'si.tex').write_text(si)
for s in ['n2','co_eq']:
 shutil.copy2(ROOT/'analysis/guard15_real30_dqg_lineality/references'/f'{s}_noiseless.json',OUT/f'{s}_fullpool_noiseless.json')
all_ds=[d for ds in RECORDS.values() for d in ds]
checks=dict(equilibrium_records=len(all_ds),ci_rows_verified=len(ci),matched_observations='identical for both objective weights',
 minimum_dqg_eigenvalue=min(d['minimum_dqg_eigenvalue'] for d in all_ds),max_equality_residual=max(d['equality_residual'] for d in all_ds),max_contraction_error=max(d['contraction_error'] for d in all_ds),max_energy_identity_error_mHa=max(d['identity_error_meh'] for d in all_ds))
(OUT/'numerical-validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
