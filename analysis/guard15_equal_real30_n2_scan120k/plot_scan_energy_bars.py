#!/usr/bin/env python3
"""N2 bond-scan figures in the equilibrium-vertical-figure style, for both the
MC-PDFT (F) energy and the Hamiltonian (H) energy.

Panel a: exact classical reference energies (markers, no error bars) with a
cubic-spline curve through them, plus measured energies (Uniform30 and
Select15 mu=2) as unconnected markers with 95% CI error bars; an inset zooms
the PEC minimum. Panel b: signed mean energy error as single-width overlapped
hatched bars per bond length (shorter bar on top), with +-1.6 mHa dashed guides.

Data: figures/signed_errors/scan_exact_reference_energies.json (exact energies)
and scan_signed_errors_mean_ci95.csv (signed error means and CIs, mHa).
Only small tables are read; memory use is negligible."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/scan_energy_bars_mpl')
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'/'signed_errors'
BONDS=(.80,.90,1.00,1.10,1.25,1.45,1.60,1.80,2.00,2.20,2.50)
ARMS=('uniform','guard15_equal')
STYLES={
    'exact':dict(color='#00A087',marker='^',label='Exact'),
    'uniform':dict(color='#3C5488',marker='o',label='Uniform30'),
    'guard15_equal':dict(color='#E64B35',marker='s',label=r'Select15 $\mu=2$'),
}
QUANTITIES={
    'F':dict(name='MC-PDFT',stem='scan_energy_and_signed_error',
             yticks=[-1.6,0,1.6,5,7.5,10,12.5,15],
             yticklabels=['-1.6','0','1.6','5','7.5','10','12.5','15']),
    'H':dict(name='Hamiltonian',stem='scan_hamiltonian_energy_and_signed_error',
             yticks=[-1.6,0,1.6,5,10,15],
             yticklabels=['-1.6','0','1.6','5','10','15']),
}

def load():
    energies=json.loads((OUT/'scan_exact_reference_energies.json').read_text())['energies']
    exact={q:np.array([energies[f'{b:.2f}'][f'exact_{q.lower()}_eh'] for b in BONDS])
           for q in QUANTITIES}
    stats={}
    with (OUT/'scan_signed_errors_mean_ci95.csv').open() as f:
        for row in csv.DictReader(f):
            bond=float(row['panel'].split('=')[1])
            stats[row['metric'],row['group'],bond]=dict(mean=float(row['mean_error_mHa']),
                                          lo=float(row['ci95_low_mHa']),hi=float(row['ci95_high_mHa']))
    return exact,stats

def make(q,eref,stats):
    cfg=QUANTITIES[q];name=cfg['name']
    bonds=np.array(BONDS)
    fig,(axa,axb)=plt.subplots(2,1,figsize=(4.68,7.3),sharex=True)
    fig.subplots_adjust(left=.235,right=.975,bottom=.083,top=.957,hspace=.10)

    # ---- panel a: absolute energies ---------------------------------------
    spline=CubicSpline(bonds,eref)
    dense=np.linspace(bonds[0],bonds[-1],400)
    st=STYLES['exact']
    axa.plot(dense,spline(dense),color=st['color'],ls='-',linewidth=1.8,zorder=2)
    axa.plot(bonds,eref,color=st['color'],ls='none',marker=st['marker'],markersize=8.3,
             mfc='white',mec=st['color'],mew=1.5,label=st['label'],zorder=3)
    for arm in ARMS:
        st=STYLES[arm]
        center=np.array([eref[i]+stats[q,arm,b]['mean']/1000 for i,b in enumerate(BONDS)])
        lo=np.array([stats[q,arm,b]['lo'] for i,b in enumerate(BONDS)])/1000
        hi=np.array([stats[q,arm,b]['hi'] for i,b in enumerate(BONDS)])/1000
        mean=np.array([stats[q,arm,b]['mean'] for i,b in enumerate(BONDS)])/1000
        axa.errorbar(bonds,center,yerr=np.vstack((mean-lo,hi-mean)),color=st['color'],ls='none',
                     marker=st['marker'],markersize=8.3,mfc='white',mec=st['color'],mew=1.5,
                     elinewidth=1.2,capsize=3.0,capthick=1.2,label=st['label'],zorder=3)
    axa.set_ylabel(f'{name} Energy (Ha)')
    axa.legend(loc='lower right',frameon=False,handlelength=1.8,
               handletextpad=.6,labelspacing=.45,borderaxespad=.2)

    # ---- panel a inset: zoom around the PEC minimum ------------------------
    zmask=(bonds>=.97)&(bonds<=1.28)
    zlo=min(eref[zmask])-.004
    zhi=max(eref[i]+stats[q,a,b]['hi']/1000
            for i,b in enumerate(BONDS) if zmask[i] for a in ARMS)+.004
    axin=axa.inset_axes([.21,.65,.47,.33])
    zdense=np.linspace(.97,1.28,200)
    axin.plot(zdense,spline(zdense),color=STYLES['exact']['color'],ls='-',linewidth=1.4,zorder=2)
    axin.plot(bonds[zmask],eref[zmask],color=STYLES['exact']['color'],ls='none',
              marker=STYLES['exact']['marker'],markersize=5.5,mfc='white',
              mec=STYLES['exact']['color'],mew=1.1,zorder=3)
    for arm in ARMS:
        st=STYLES[arm]
        zb=bonds[zmask]
        zc=np.array([eref[i]+stats[q,arm,b]['mean']/1000 for i,b in enumerate(BONDS) if zmask[i]])
        zlo_a=np.array([stats[q,arm,b]['lo'] for i,b in enumerate(BONDS) if zmask[i]])/1000
        zhi_a=np.array([stats[q,arm,b]['hi'] for i,b in enumerate(BONDS) if zmask[i]])/1000
        zm=np.array([stats[q,arm,b]['mean'] for i,b in enumerate(BONDS) if zmask[i]])/1000
        axin.errorbar(zb,zc,yerr=np.vstack((zm-zlo_a,zhi_a-zm)),color=st['color'],ls='none',
                      marker=st['marker'],markersize=5.5,mfc='white',mec=st['color'],mew=1.1,
                      elinewidth=1.0,capsize=2.0,capthick=1.0,zorder=3)
    axin.set_xlim(.97,1.28);axin.set_ylim(zlo,zhi)
    axin.set_xticks([1.00,1.10,1.25],labels=['1.00','1.10','1.25'])
    yt=np.arange(np.ceil(zlo*50)/50,zhi,.02)
    axin.set_yticks(yt,labels=[f'{v:.2f}' for v in yt])
    axin.tick_params(labelsize=9,length=2.6,width=.7)
    axa.indicate_inset_zoom(axin,edgecolor='0.35',lw=.9,alpha=.9,zorder=1)

    # ---- panel b: signed error bars (single width, shorter on top) ---------
    width=.055
    bar_styles={'uniform':dict(color='#3C5488',hatch='//'),
                'guard15_equal':dict(color='#E64B35',hatch='\\\\')}
    values={arm:np.array([stats[q,arm,b]['mean'] for b in BONDS]) for arm in ARMS}
    for i,b in enumerate(BONDS):
        order=sorted(ARMS,key=lambda a:-abs(values[a][i]))  # longest first
        for arm in order:
            bs=bar_styles[arm]
            axb.bar(b,values[arm][i],width=width,color=bs['color'],edgecolor='0.1',
                    linewidth=.8,hatch=bs['hatch'],zorder=3,
                    label=STYLES[arm]['label'] if i==0 else None)
    axb.axhline(0.,color='0.0',linewidth=.9,linestyle=':',zorder=2)
    for v in (-1.6,1.6):
        axb.axhline(v,color='0.35',linewidth=.9,linestyle='--',zorder=2)
    axb.set_ylabel(f'{name} Energy Error (mHa)')
    axb.set_xlabel('Bond length (Å)')
    axb.set_yticks(cfg['yticks'],labels=cfg['yticklabels'])
    for lbl in axb.get_yticklabels():
        if lbl.get_text() in ('-1.6','1.6'):lbl.set_color('0.35')
    axb.legend(loc='upper right',frameon=False,handlelength=1.4,
               handletextpad=.6,labelspacing=.45,borderaxespad=.2)

    for ax,tag in ((axa,'a'),(axb,'b')):
        ax.text(-.26,1.02,tag,transform=ax.transAxes,fontsize=18,
                fontweight='bold',ha='left',va='bottom')
        ax.yaxis.set_label_coords(-.24,.5)
    axb.set_xlim(.72,2.58)
    axb.set_xticks(bonds,labels=['0.8','','','1.1','','1.45',
                                 '','1.8','2.0','2.2','2.5'])

    stem=OUT/cfg['stem']
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    (OUT/(cfg['stem']+'_caption.txt')).write_text(
        f'N2 bond-length scan (cc-pVDZ, CAS(10e,8o), 120K shots, 8 matched streams), {name} energy.\n'
        'Panel a: exact classical reference energies (triangles) with a cubic-spline curve through\n'
        'them; measured energies (mean over 8 streams) for Uniform30 and Select15 mu=2 with pointwise\n'
        '95% bootstrap CI error bars, unconnected. Panel b: signed mean energy error (measured - exact,\n'
        'mHa) per bond length; single-width overlapped bars, bar sign follows the signed mean error.\n')
    print('saved',stem.with_suffix('.png'),flush=True)

def main():
    exact,stats=load()
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
    for q in ('F','H'):
        make(q,exact[q],stats)

if __name__=='__main__':main()
