#!/usr/bin/env python3
"""60k-shot ablation figure: energy-error decomposition for five arms on N2 and CO.

Two stacked panels (a: N2, b: CO). X categories: Hamiltonian and MC-PDFT energy
error; each category carries five bars (Uniform30, Except-Select15 mu=0,
Select15 mu=0, Except-Select15 mu=2, Select15 mu=2). Bar heights are the
8-stream RMSEs from figures/equilibrium_errorbars/rmse_ci95.csv (budget 60000);
no error bars. Each bar splits into the part common to the H and F errors
(non-on-top / E_C-driven component, f_non_ontop_meh) and the part unique to the
metric (H: remainder h_error - f_non_ontop; F: on-top f_on_top_meh), with the
signed stream means rescaled so the parts sum to the RMSE. Same-sign parts are
stacked as opaque bars; opposite-sign parts are drawn faint in opposite
directions with the opaque net bar on top.

Memory: only per-stream result JSONs and a small CSV are read; negligible."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ablation_decomp_mpl')
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE=Path(__file__).resolve().parent
EQ=HERE/'figures'/'equilibrium_errorbars'
BUDGET=60000
SYSTEMS=(('n2',r'N$_2$'),('co_eq','CO'))
ARMS=(('uniform','Uniform30'),
      ('exclusive_guard15_mu0',r'Except-Select15 $\mu=0$'),
      ('guard15_equal_mu0',r'Select15 $\mu=0$'),
      ('exclusive_guard15_mu2',r'Except-Select15 $\mu=2$'),
      ('guard15_equal',r'Select15 $\mu=2$'))
COLORS={'uniform':'#3C5488','exclusive_guard15_mu0':'#91D1C2','guard15_equal_mu0':'#00A087',
        'exclusive_guard15_mu2':'#F39B7F','guard15_equal':'#E64B35'}
METRICS=(('h','Hamiltonian','h_error_meh'),('f','MC-PDFT','f_error_meh'))

def load():
    rmse={}
    with (EQ/'rmse_ci95.csv').open() as f:
        for row in csv.DictReader(f):
            if int(row['budget'])!=BUDGET:continue
            rmse[row['system'],row['arm'],row['metric']]=float(row['rmse'])
    comp={}
    for sysname,_ in SYSTEMS:
        for arm,_ in ARMS:
            hh,ff,fn,fo=[],[],[],[]
            for r in range(8):
                d=json.loads((HERE/'results'/sysname/arm/f'b{BUDGET}_r{r}.json').read_text())
                hh.append(d['h_error_meh']);ff.append(d['f_error_meh'])
                fn.append(d['f_non_ontop_meh']);fo.append(d['f_on_top_meh'])
            hh,ff,fn,fo=map(np.array,(hh,ff,fn,fo))
            assert abs(np.sqrt((hh**2).mean())-rmse[sysname,arm,'h_error_meh'])<1e-6
            assert abs(np.sqrt((ff**2).mean())-rmse[sysname,arm,'f_error_meh'])<1e-6
            comp[sysname,arm]=dict(mean_h=float(hh.mean()),mean_f=float(ff.mean()),
                                   common=float(fn.mean()),
                                   u_h=float((hh-fn).mean()),u_f=float(fo.mean()))
    return rmse,comp

def main():
    rmse,comp=load()
    plt.rcParams.update({'font.family':'sans-serif',
                         'font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
                         'mathtext.fontset':'dejavusans',
                         'font.size':18,'axes.labelsize':16.5,'axes.titlesize':16.5,
                         'axes.linewidth':1.3,'xtick.labelsize':13.5,'ytick.labelsize':13.5,
                         'xtick.major.size':5.2,'ytick.major.size':5.2,
                         'xtick.major.width':1.05,'ytick.major.width':1.05,
                         'legend.fontsize':12,'lines.linewidth':1.8,
                         'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
                         'savefig.facecolor':'white'})
    fig,(axa,axb)=plt.subplots(2,1,figsize=(4.68,7.6))
    fig.subplots_adjust(left=.22,right=.97,bottom=.075,top=.855,hspace=.16)

    width=.13
    offsets=(np.arange(5)-2)*width*1.18
    for ax,(sysname,tag),letter in ((axa,SYSTEMS[0],'a'),(axb,SYSTEMS[1],'b')):
        for g,(m,mlabel,mkey) in enumerate(METRICS):
            for k,(arm,alabel) in enumerate(ARMS):
                x=g+offsets[k]
                total=rmse[sysname,arm,mkey]
                mean=comp[sysname,arm][f'mean_{m}']
                s=total/mean
                c=comp[sysname,arm]['common']*s
                u=comp[sysname,arm][f'u_{m}']*s
                color=COLORS[arm]
                if c*u>=0:  # same direction: opaque stack
                    ax.bar(x,c,width,color=color,edgecolor='0.1',linewidth=.8,zorder=3)
                    ax.bar(x,u,width,bottom=c,color=color,edgecolor='0.1',linewidth=.8,
                           hatch='//',zorder=3)
                else:  # opposite: faint components + opaque net
                    ax.bar(x,c,width,color=color,edgecolor='0.1',linewidth=.8,alpha=.35,zorder=3)
                    ax.bar(x,u,width,color=color,edgecolor='0.1',linewidth=.8,
                           hatch='//',alpha=.35,zorder=3)
                    ax.bar(x,total,width,color=color,edgecolor='0.1',linewidth=.8,zorder=4)
        ax.axhline(0.,color='0.0',linewidth=.9,linestyle=':',zorder=2)
        ax.axhline(1.6,color='0.35',linewidth=.9,linestyle='--',zorder=2)
        ax.set_ylabel('Energy Error (mHa)')
        ax.set_ylim(-1.5,26.)
        ax.set_yticks([0,1.6,5,10,15,20,25],
                      labels=['0','1.6','5','10','15','20','25'])
        for lbl in ax.get_yticklabels():
            if lbl.get_text()=='1.6':lbl.set_color('0.35')
        ax.set_xlim(-.5,1.55)
        ax.set_xticks([0,1],labels=['Hamiltonian','MC-PDFT'])
        ax.text(.03,.97,tag,transform=ax.transAxes,fontsize=15,ha='left',va='top')
        ax.text(-.26,1.02,letter,transform=ax.transAxes,fontsize=18,
                fontweight='bold',ha='left',va='bottom')
        ax.yaxis.set_label_coords(-.16,.5)

    handles=[Patch(facecolor=COLORS[a],edgecolor='0.1',linewidth=.8,label=l) for a,l in ARMS]
    handles+=[Patch(facecolor='0.75',edgecolor='0.1',linewidth=.8,label='Common part'),
              Patch(facecolor='0.75',edgecolor='0.1',linewidth=.8,hatch='//',label='Unique part')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.998),ncol=2,
               frameon=False,handlelength=1.4,handletextpad=.6,labelspacing=.4,
               columnspacing=1.0)

    stem=EQ/'ablation_60k_energy_error_decomposition'
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)

    with (EQ/'ablation_60k_energy_error_decomposition.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['system','metric','arm','rmse_total_mHa','common_part_mHa',
                    'unique_part_mHa','same_direction'])
        for sysname,_ in SYSTEMS:
            for m,_,mkey in METRICS:
                for arm,_ in ARMS:
                    total=rmse[sysname,arm,mkey]
                    s=total/comp[sysname,arm][f'mean_{m}']
                    c=comp[sysname,arm]['common']*s
                    u=comp[sysname,arm][f'u_{m}']*s
                    w.writerow([sysname,m,arm,f'{total:.6f}',f'{c:.6f}',f'{u:.6f}',
                                int(c*u>=0)])
    (EQ/'ablation_60k_energy_error_decomposition_caption.txt').write_text(
        'Ablation at 60000 total shots, equilibrium geometries: N2 (R=1.10 Angstrom) and\n'
        'CO (R=1.128 Angstrom), cc-pVDZ, CAS(10e,8o); 8 matched shot streams, fixed Haar real30 pool.\n'
        'Bar heights are 8-stream RMSEs of the signed energy error (no error bars). Each bar splits\n'
        'into the component common to the H and F errors (non-on-top / E_C-driven part) and the part\n'
        'unique to the metric (Hamiltonian: H error minus common; MC-PDFT: on-top part); signed\n'
        'stream-mean components are rescaled so the parts sum to the RMSE. Same-sign parts are\n'
        'stacked opaque; opposite-sign parts are drawn faint in opposite directions with the opaque\n'
        'net bar on top. Arms: Uniform30; Except-Select15 (complementary 15 frames) and Select15\n'
        '(guard15) at mu=0 and mu=2. The dashed gray line marks 1.6 mHa.\n')
    print('saved',stem.with_suffix('.png'),flush=True)

if __name__=='__main__':main()
