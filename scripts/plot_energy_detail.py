"""Expanded high-budget views, using the published confidence intervals unchanged."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MS=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((MS/'source-data/equilibrium_rmse_ci95.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(7,5.1))
arms=[('uniform','Uniform30','#5477a6','o'),
      ('guard15_equal_mu0',r'Select15, $\mu=0$','#039b98','^'),
      ('guard15_equal',r'Select15, $\mu=2$','#e55b4c','s')]
for ax,(system,metric,letter,title) in zip(axes.flat,[
    ('n2','h_error_meh','a',r'N$_2$, Hamiltonian'),
    ('n2','f_error_meh','b',r'N$_2$, MC-PDFT'),
    ('co_eq','h_error_meh','c','CO, Hamiltonian'),
    ('co_eq','f_error_meh','d','CO, MC-PDFT')]):
    ax.set_title(title,fontsize=10)
    ax.text(-.19,1.04,letter,transform=ax.transAxes,weight='bold',fontsize=11)
    high=0
    for j,(arm,label,color,marker) in enumerate(arms):
        rs=[next(r for r in rows if r['system']==system and r['metric']==metric
                 and r['arm']==arm and int(r['budget'])==b) for b in (120000,240000)]
        y=np.array([float(r['rmse']) for r in rs]);lo=np.array([float(r['ci95_low']) for r in rs]);hi=np.array([float(r['ci95_high']) for r in rs])
        high=max(high,max(hi))
        ax.errorbar(np.arange(2)+(j-1)*.12,y,yerr=[y-lo,hi-y],label=label,
                    color=color,marker=marker,markersize=4,capsize=3,lw=1,
                    markerfacecolor='white',linestyle='none')
    ax.axhline(1.6,color='.45',lw=.8,ls='--')
    ax.set_ylim(0,high*1.08)
    ax.set_xlim(-.35,1.35)
    ax.set_xticks([0,1],['120,000','240,000'])
    ax.set_xlabel('Total measurement shots')
    ax.set_ylabel('Energy RMSE (mHa)')
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=3,
           bbox_to_anchor=(.5,1.005),frameon=False)
fig.tight_layout(rect=[0,0,1,.94],h_pad=1.6,w_pad=1.5)
fig.savefig(MS/'figures/energy_high_budget_detail.pdf',bbox_inches='tight')
plt.close(fig)
