#!/usr/bin/env python3
"""N2 bond-scan figures for the gate-noise package (g1.0 = full calibrated gate
noise, g0.2 = 1/5 gate noise), in the equilibrium-vertical-figure style of
guard15_equal_real30_n2_scan120k/plot_scan_energy_bars.py, adapted to
single-stream data (no error bars, markers instead of bars). One figure per
gate-noise scale x quantity (F = MC-PDFT, H = Hamiltonian).

Panel a: exact classical reference energies (triangles) with a cubic-spline
curve through them, plus five unconnected marker series: Noise-less (clean),
raw (Uniform30 / Select15 mu=2), rem_lin_post (Uniform30 / Select15 mu=2);
an inset zooms the PEC minimum (raw series is clipped there by design).
Panel b: signed energy error (measured - exact, mHa) as markers for the same
five series. Panel c: signed error for the two rem_lin_post series only, with
a +1.6 mHa dashed guide (-1.6 is added when it fits the adaptive y-range).

Data: analysis/<scale>/summary.csv (single-stream signed errors, mHa) and the
exact reference energies from the sibling scan campaign (verified to match the
package's own context_meta.json exact energies to <1e-9 Ha). Only small tables
are read; memory use is negligible."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/scan_energy_g02_mpl')
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
OUT=HERE.parent/'generated/figures/noise'
OUT.mkdir(parents=True,exist_ok=True)
EXACT_JSON=HERE/'analysis/scan_exact_reference_energies.json'
META_GLOB='code/mindquantum_poc/results*/n2_r*/context_meta.json'
SCALES=('g0.2','g1.0')  # analysis subdirectories / gate-noise scale tags

# The clean (noise-less) run is arm-specific; the figure shows the Select15
# arm's clean as the single "Noise-less" series. Flip to 'uniform' if wanted.
CLEAN_ARM='guard15_equal'

ARMS=('uniform','guard15_equal')
# (variant, arm, label, color, marker, filled)
SERIES=(
    ('clean',       None,            'Noise-less',                        '0.30',   'd', False),
    ('raw',         'uniform',       'Uniform30 raw',                     '#3C5488','o', False),
    ('raw',         'guard15_equal', r'Select15 $\mu=2$ raw',             '#E64B35','s', False),
    ('rem_lin_post','uniform',       'Uniform30 REM-lin + post',          '#4DBBD5','v', False),
    ('rem_lin_post','guard15_equal', r'Select15 $\mu=2$ REM-lin + post',  '#F39B7F','h', False),
)
EXACT_STYLE=dict(color='#00A087',marker='^',label='Exact')
QUANTITIES={
    'F':dict(name='MC-PDFT',prefix='scan_energy_and_signed_error',
             col='f_error_meh',ref='exact_f_eh'),
    'H':dict(name='Hamiltonian',prefix='scan_hamiltonian_energy_and_signed_error',
             col='h_error_meh',ref='exact_h_eh'),
}

def load(scale):
    """Return (bonds, exact[q][i], signed_errors[(variant,arm)][q][i]) in mHa."""
    rows=list(csv.DictReader((HERE/'analysis'/scale/'summary.csv').open()))
    bonds=sorted({float(r['bond_angstrom']) for r in rows})
    err={}
    for r in rows:
        i=bonds.index(float(r['bond_angstrom']))
        for q,cfg in QUANTITIES.items():
            err.setdefault((r['variant'],r['arm']),{}).setdefault(q,{})[i]=float(r[cfg['col']])
    ref=json.loads(EXACT_JSON.read_text())['energies']
    exact={q:np.array([ref[f'{b:.2f}'][cfg['ref']] for b in bonds])
           for q,cfg in QUANTITIES.items()}
    # cross-check: package-internal exact H energies must match the reference
    import glob
    for f in glob.glob(str(HERE/META_GLOB)):
        meta=json.loads(Path(f).read_text())
        i=bonds.index(meta['bond_angstrom'])
        assert abs(meta['exact_energy']-exact['H'][i])<1e-9, f'exact H mismatch: {f}'
    tab={k:{q:np.array([v[q][i] for i in range(len(bonds))]) for q in QUANTITIES}
         for k,v in err.items()}
    return np.array(bonds),exact,tab

def series_errors(tab):
    """Yield (label,color,marker,filled,err_dict) for the five series."""
    for variant,arm,label,color,marker,filled in SERIES:
        key=(variant,arm if arm is not None else CLEAN_ARM)
        yield label,color,marker,filled,{q:tab[key][q] for q in QUANTITIES}

def legend_handles():
    """Figure-level legend proxies, ordered for column-major ncol=2 filling so
    rows come out as: Exact | Noise-less / Uniform raw | Select raw /
    Uniform REM-lin+post | Select REM-lin+post."""
    from matplotlib.lines import Line2D
    def mk(color,marker,label):
        return Line2D([],[],color=color,ls='none',marker=marker,markersize=10,
                      mfc='white',mec=color,mew=1.6,label=label)
    d={label:mk(color,marker,label) for _,_,label,color,marker,_ in SERIES}
    return [mk(EXACT_STYLE['color'],EXACT_STYLE['marker'],EXACT_STYLE['label']),
            d['Uniform30 raw'],d['Uniform30 REM-lin + post'],
            d['Noise-less'],d[r'Select15 $\mu=2$ raw'],d[r'Select15 $\mu=2$ REM-lin + post']]

def make(q,bonds,eref,tab,scale):
    cfg=QUANTITIES[q];name=cfg['name']
    series=list(series_errors(tab))
    fig,(axa,axb,axc)=plt.subplots(3,1,figsize=(4.68,10.5),sharex=True)
    fig.subplots_adjust(left=.235,right=.975,bottom=.058,top=.915,hspace=.12)
    fig.legend(handles=legend_handles(),loc='lower center',ncol=2,frameon=False,
               bbox_to_anchor=(.495,.925),handlelength=1.3,handletextpad=.5,
               columnspacing=.8,labelspacing=.4,fontsize=11)

    # ---- panel a: absolute energies ---------------------------------------
    spline=CubicSpline(bonds,eref)
    dense=np.linspace(bonds[0],bonds[-1],400)
    axa.plot(dense,spline(dense),color=EXACT_STYLE['color'],ls='-',linewidth=1.8,zorder=2)
    axa.plot(bonds,eref,color=EXACT_STYLE['color'],ls='none',marker=EXACT_STYLE['marker'],
             markersize=8.3,mfc='white',mec=EXACT_STYLE['color'],mew=1.5,
             label=EXACT_STYLE['label'],zorder=3)
    for label,color,marker,filled,eq in series:
        axa.plot(bonds,eref+eq[q]/1000,color=color,ls='none',marker=marker,markersize=8.3,
                 mfc=color if filled else 'white',mec=color,mew=1.5,label=label,zorder=3)
    axa.set_ylabel(f'{name} Energy (Ha)')

    # ---- panel a inset: zoom around the PEC minimum (raw clipped) ----------
    zmask=(bonds>=.97)&(bonds<=1.28)
    znear=[eq[q][zmask] for label,color,marker,filled,eq in series if 'raw' not in label]
    zlo=min(np.min(eref[zmask]),*(np.min(eref[zmask]+e/1000) for e in znear))-.004
    zhi=max(np.max(eref[zmask]+e/1000) for e in znear)+.004
    axin=axa.inset_axes([.16,.55,.46,.34])
    zdense=np.linspace(.97,1.28,200)
    axin.plot(zdense,spline(zdense),color=EXACT_STYLE['color'],ls='-',linewidth=1.4,zorder=2)
    axin.plot(bonds[zmask],eref[zmask],color=EXACT_STYLE['color'],ls='none',
              marker=EXACT_STYLE['marker'],markersize=5.5,mfc='white',
              mec=EXACT_STYLE['color'],mew=1.1,zorder=3)
    for label,color,marker,filled,eq in series:
        if 'raw' in label:
            continue
        axin.plot(bonds[zmask],(eref+eq[q]/1000)[zmask],color=color,ls='none',marker=marker,
                  markersize=5.5,mfc=color if filled else 'white',mec=color,mew=1.1,zorder=3)
    axin.set_xlim(.97,1.28);axin.set_ylim(zlo,zhi)
    axin.set_xticks([1.00,1.10,1.25],labels=['1.00','1.10','1.25'])
    yt=np.arange(np.ceil(zlo*50)/50,zhi,.02)
    axin.set_yticks(yt,labels=[f'{v:.2f}' for v in yt])
    axin.tick_params(labelsize=9,length=2.6,width=.7)
    # inset tick labels on the right so they stay clear of the main y-axis labels
    axin.yaxis.tick_right()
    # draw only the zoom rectangle on the parent; connector lines would cross
    # the legend and markers in this busier single-stream layout
    ind=axa.indicate_inset_zoom(axin,edgecolor='0.35',lw=.9,alpha=.9,zorder=1)
    for conn in ind.connectors:
        conn.set_visible(False)

    # ---- panel b: signed error markers, all five series --------------------
    for label,color,marker,filled,eq in series:
        axb.plot(bonds,eq[q],color=color,ls='none',marker=marker,markersize=7.5,
                 mfc=color if filled else 'white',mec=color,mew=1.4,label=label,zorder=3)
    axb.axhline(0.,color='0.0',linewidth=.9,linestyle=':',zorder=2)
    allv=np.concatenate([eq[q] for *_,eq in series])
    axb.set_ylim(min(allv.min()*1.15,-12),allv.max()*1.10)
    axb.set_ylabel(f'{name} Energy Error (mHa)')

    # ---- panel c: signed error markers, rem_lin_post only ------------------
    rlp=[s for s in series if 'REM-lin' in s[0]]
    for label,color,marker,filled,eq in rlp:
        axc.plot(bonds,eq[q],color=color,ls='none',marker=marker,markersize=7.5,
                 mfc=color if filled else 'white',mec=color,mew=1.4,label=label,zorder=3)
    axc.axhline(0.,color='0.0',linewidth=.9,linestyle=':',zorder=2)
    vals=np.concatenate([eq[q] for *_,eq in rlp])
    vmin,vmax=vals.min(),vals.max()
    span=vmax-vmin
    lo=min(vmin-.12*span,-2.2);hi=max(vmax+.12*span,2.2)
    axc.set_ylim(lo,hi)
    axc.axhline(1.6,color='0.35',linewidth=.9,linestyle='--',zorder=2)
    base=[t for t in range(-80,81,10) if lo+.06*span<t<hi-.06*span]
    axc.set_yticks(base,labels=[f'{v:g}' for v in base])
    # the guides are too close to 0 for tick labels; annotate them instead.
    # the -1.6 guide is drawn (label below the line) only when there is room;
    # when the data sit far above zero it would be cramped into the bottom
    # edge, so both the line and the label are omitted.
    bbox=dict(fc='white',ec='none',pad=.15)
    axc.text(2.55,1.6,'1.6',color='0.35',fontsize=10,ha='right',va='bottom',bbox=bbox)
    if lo<-1.6-.05*(hi-lo):
        axc.axhline(-1.6,color='0.35',linewidth=.9,linestyle='--',zorder=2)
        axc.text(2.55,-1.6,'-1.6',color='0.35',fontsize=10,ha='right',va='top',bbox=bbox)
    axc.set_ylabel(f'{name} Energy Error (mHa)')
    axc.set_xlabel('Bond length (Å)')

    for ax,tag in ((axa,'a'),(axb,'b'),(axc,'c')):
        ax.text(-.26,1.02,tag,transform=ax.transAxes,fontsize=18,
                fontweight='bold',ha='left',va='bottom')
        ax.yaxis.set_label_coords(-.24,.5)
    axc.set_xlim(.72,2.58)
    axc.set_xticks(bonds,labels=['0.8','','','1.1','','1.45',
                                 '','1.8','2.0','2.2','2.5'])

    stag=scale.replace('.','')  # 'g0.2' -> 'g02'
    stem=OUT/f"{cfg['prefix']}_{stag}"
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    noise_desc=('calibration-derived effective gate parameters at scale=1.0' if scale=='g1.0'
                else 'gate noise scaled to 0.2 (1/5 gate noise)')
    (OUT/(stem.name+'_caption.txt')).write_text(
        'N2 bond-length scan (cc-pVDZ, CAS(10e,8o), 120K shots, single stream) under the Wukong-180-2\n'
        f'calibrated noise model with {noise_desc} (scale={scale[1:]}, readout noise unchanged), '
        f'{name} energy.\n'
        'Panel a: exact classical reference energies (triangles) with a cubic-spline curve through\n'
        'them; unconnected markers for five single-stream series (no error bars): Noise-less (clean\n'
        f'run, Select15 arm), scale={scale[1:]} raw for Uniform30 and Select15 mu=2, and scale={scale[1:]} REM-linear-\n'
        'inversion + postselection (rem_lin_post) for Uniform30 and Select15 mu=2. Inset zooms the\n'
        'PEC minimum (raw series clipped). Panel b: signed energy error (measured - exact, mHa) per\n'
        'bond length for the same five series. Panel c: signed error for the two rem_lin_post series\n'
        'only, with a +1.6 mHa dashed guide (-1.6 added when it fits the adaptive y-range).\n')
    print('saved',stem.with_suffix('.png'),flush=True)

def main():
    plt.rcParams.update({'font.family':'sans-serif',
                         'font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
                         'mathtext.fontset':'dejavusans',
                         'font.size':18,'axes.labelsize':14,'axes.titlesize':14,
                         'axes.linewidth':1.3,'xtick.labelsize':13.5,'ytick.labelsize':13.5,
                         'xtick.major.size':5.2,'ytick.major.size':5.2,
                         'xtick.major.width':1.05,'ytick.major.width':1.05,
                         'legend.fontsize':12,'lines.linewidth':1.8,
                         'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
                         'savefig.facecolor':'white'})
    for scale in SCALES:
        bonds,exact,tab=load(scale)
        for q in ('F','H'):
            make(q,bonds,exact[q],tab,scale)

if __name__=='__main__':main()
