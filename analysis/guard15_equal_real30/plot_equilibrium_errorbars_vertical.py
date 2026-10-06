#!/usr/bin/env python3
"""Per-molecule vertical errorbar figures (2-RDM, H, F) in the n2-pec-energy-curves
panel-b grayscale style. Reads the precomputed rmse_ci95.csv only (no bootstrap,
no raw JSON/npz) so memory use stays negligible."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/equilibrium_errorbars_vertical_mpl')
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator, ScalarFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'/'equilibrium_errorbars'
BUDGETS=(30000,60000,120000,240000)
ARMS=('uniform','guard15_equal_mu0','guard15_equal')
METRICS=('d2_error','h_error_meh','f_error_meh')
SYSTEMS={
    'n2':dict(file='n2',note=r'N$_2$, $R=1.10$ Å'),
    'co_eq':dict(file='co',note=r'CO, $R=1.128$ Å'),
}
STYLES={
    'uniform':dict(color='#3C5488',marker='o',ls='-',filled=True,label='Uniform30'),
    'guard15_equal_mu0':dict(color='#00A087',marker='^',ls='-',filled=False,label=r'Select15 $\mu=0$'),
    'guard15_equal':dict(color='#E64B35',marker='s',ls='-',filled=True,label=r'Select15 $\mu=2$'),
}
PANELS=[
    dict(metric='d2_error',tag='a',ylabel='2-RDM Frobenius Error'),
    dict(metric='h_error_meh',tag='b',ylabel='Hamiltonian Error (mHa)',threshold=1.6,
         yticks=[1.6,10,20,30,40],yticklabels=['1.6','10','20','30','40']),
    dict(metric='f_error_meh',tag='c',ylabel='MC-PDFT Error (mHa)',threshold=1.6,
         yticks=[1.6,5,10,15,20,25,30],yticklabels=['1.6','5','10','15','20','25','30']),
]

def load():
    table={}
    with (OUT/'rmse_ci95.csv').open() as f:
        for row in csv.DictReader(f):
            key=(row['system'],int(row['budget']),row['arm'],row['metric'])
            table[key]=dict(rmse=float(row['rmse']),lo=float(row['ci95_low']),hi=float(row['ci95_high']))
    return table

def plot(system,info,table):
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
    fig,axes=plt.subplots(3,1,figsize=(4.68,9.4),sharex=True)
    fig.subplots_adjust(left=.214,right=.983,bottom=.082,top=.958,hspace=.115)
    x=np.array(BUDGETS)/1000
    for ax,panel in zip(axes,PANELS):
        for arm in ARMS:
            st=STYLES[arm]
            data=[table[system,b,arm,panel['metric']] for b in BUDGETS]
            y=np.array([d['rmse'] for d in data])
            lo=np.array([d['lo'] for d in data]);hi=np.array([d['hi'] for d in data])
            ax.errorbar(x,y,yerr=np.vstack((y-lo,hi-y)),color=st['color'],ls=st['ls'],
                        marker=st['marker'],markersize=8.3,mfc='white',mec=st['color'],mew=1.5,
                        elinewidth=1.2,capsize=3.0,capthick=1.2,
                        label=st['label'],zorder=3 if st['filled'] else 2)
        ax.set_xscale('log',base=2);ax.set_xlim(26,278)
        ax.set_xticks(x,labels=['30k','60k','120k','240k']);ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(bottom=0);ax.set_ylabel(panel['ylabel'])
        ax.yaxis.set_label_coords(-.15,.5)
        ax.text(-.22,1.02,panel['tag'],transform=ax.transAxes,fontsize=18,
                fontweight='bold',ha='left',va='bottom')
        if 'yticks' in panel:
            ax.set_yticks(panel['yticks'],labels=panel['yticklabels'])
        if 'threshold' in panel:
            ax.axhline(panel['threshold'],color='0.35',ls=(0,(4,3)),linewidth=1.2,zorder=1)
            for tick,label in zip(ax.get_yticks(),ax.get_yticklabels()):
                if np.isclose(tick,panel['threshold']):
                    label.set_color('0.35')
        if panel['metric']=='d2_error':
            ax.yaxis.set_major_locator(MultipleLocator(.05))
            ax.yaxis.set_major_formatter(ScalarFormatter())
    axes[0].text(.03,.06,info['note'],transform=axes[0].transAxes,fontsize=13.5,
                 fontweight='semibold',ha='left',va='bottom')
    axes[0].legend(loc='upper right',frameon=False,handlelength=1.8,
                   handletextpad=.6,labelspacing=.45,borderaxespad=.2)
    axes[-1].set_xlabel('Total measurement shots')
    stem=OUT/f"{info['file']}_rmse_errorbars_vertical"
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    print('saved',stem.with_suffix('.png'),flush=True)

def main():
    table=load()
    for system,info in SYSTEMS.items():
        plot(system,info,table)

if __name__=='__main__':main()
