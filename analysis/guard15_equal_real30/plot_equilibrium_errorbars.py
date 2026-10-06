#!/usr/bin/env python3
"""Plot separate H, F and Frobenius 2-RDM RMSEs with stream-bootstrap CIs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/equilibrium_errorbars_mpl')
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator, ScalarFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'/'equilibrium_errorbars'
SYSTEMS=('n2','co_eq')
BUDGETS=(30000,60000,120000,240000)
STREAMS=tuple(range(8))
ARMS=('uniform','guard15_equal_mu0','guard15_equal','exclusive_guard15_mu0','exclusive_guard15_mu2')
METRICS=('h_error_meh','f_error_meh','d2_error')
SEED=20261006
BOOTSTRAPS=100000
STYLES={
    'uniform':dict(color='#59636e',marker='D',ls='-',filled=True,label=r'Uniform30, $(\lambda,\mu)=(1,0)$'),
    'guard15_equal_mu0':dict(color='#0072B2',marker='o',ls='--',filled=False,label=r'Guard15, $(1,0)$'),
    'guard15_equal':dict(color='#0072B2',marker='o',ls='-',filled=True,label=r'Guard15, $(1,2)$'),
    'exclusive_guard15_mu0':dict(color='#D55E00',marker='^',ls='--',filled=False,label=r'Exclusive15, $(1,0)$'),
    'exclusive_guard15_mu2':dict(color='#D55E00',marker='^',ls='-',filled=True,label=r'Exclusive15, $(1,2)$'),
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={(r['system'],r['budget']):r for r in json.loads((HERE/'summary.json').read_text())}
    rows=[];raw_rows=[];provenance={};lookup={}
    for i,system in enumerate(SYSTEMS):
        # The same resampled stream IDs are used across all arms, budgets and
        # metrics within a molecule, preserving the paired/nested experiment.
        index=np.random.default_rng(SEED+i).integers(0,len(STREAMS),size=(BOOTSTRAPS,len(STREAMS)))
        for budget in BUDGETS:
            records={}
            for arm in ARMS:
                records[arm]=[]
                for stream in STREAMS:
                    path=HERE/'results'/system/arm/f'b{budget}_r{stream}.json'
                    d=json.loads(path.read_text())
                    assert d['status']=='optimal' and d['error_basis']=='spin'
                    assert d['stream']==stream and d['budget']==budget and d['system']==system
                    assert d['lambda_radius']==1 and d['mu_ftpbe']==(2 if arm in ('guard15_equal','exclusive_guard15_mu2') else 0)
                    provenance[str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    records[arm].append(d)
                    raw_rows.append(dict(system=system,budget=budget,arm=arm,stream=stream,
                                         h_error_mHa=d['h_error_meh'],f_error_mHa=d['f_error_meh'],
                                         d2_frobenius_error=d['d2_error']))
                for metric in METRICS:
                    values=np.array([d[metric] for d in records[arm]])
                    point=float(np.sqrt(np.mean(values**2)))
                    distribution=np.sqrt(np.mean(values[index]**2,axis=1))
                    low,high=np.quantile(distribution,[.025,.975])
                    assert low<=point<=high and np.isfinite(distribution).all()
                    if metric!='d2_error':
                        previous=summary[system,budget][arm+('_h_rmse' if metric=='h_error_meh' else '_f_rmse')]
                        np.testing.assert_allclose(point,previous,rtol=1e-12,atol=1e-12)
                    row=dict(system=system,budget=budget,arm=arm,metric=metric,n_streams=8,
                             rmse=point,ci95_low=float(low),ci95_high=float(high),
                             bootstrap_se=float(distribution.std(ddof=1)))
                    rows.append(row);lookup[system,budget,arm,metric]=row
            for stream in STREAMS:
                assert len({records[a][stream]['prefix500_digest'] for a in ARMS})==1
    assert len(rows)==120 and len(raw_rows)==320
    for name,table in [('rmse_ci95',rows),('stream_errors',raw_rows)]:
        with (OUT/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    method=dict(streams=STREAMS,bootstrap_replicates=BOOTSTRAPS,bootstrap_seed=SEED,
                uncertainty='Pointwise 95% percentile bootstrap confidence interval of RMSE across streams',
                resampling='Sample 8 stream IDs with replacement; shared indices across arms/budgets/metrics within each molecule; molecules use independent RNGs',
                energy_rmse='sqrt(mean((estimated energy - exact reference energy)^2)), mHa',
                d2_rmse='sqrt(mean(||estimated D2 - exact D2||_F^2)), unnormalized pair-basis Frobenius norm',
                d2_difference_from_previous_summary='This figure uses RMS Frobenius error; previous summary d2_mean is mean Frobenius error',
                scope='Conditional on the fixed Haar real30 frame pool; uncertainty is across 8 shot streams, not across frame pools',
                interval_note='Pointwise approximate intervals with n=8; not simultaneous confidence bands or significance tests',
                source_sha256=provenance,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'method.json').write_text(json.dumps(method,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.labelsize':11,
                         'axes.titlesize':11.5,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.8,'xtick.direction':'out','ytick.direction':'out',
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(13.8,8.5),sharex=True,sharey='col')
    fig.subplots_adjust(left=.075,right=.985,bottom=.145,top=.765,wspace=.26,hspace=.37)
    fig.suptitle(r'N$_2$ and CO at equilibrium',x=.075,ha='left',y=.967,fontsize=20,fontweight='semibold')
    fig.text(.075,.92,r'cc-pVDZ  ·  CAS(10e,8o)  ·  8 shot streams  ·  95% bootstrap confidence intervals',fontsize=11,color='#48515a')
    legend=[]
    for arm in ARMS:
        st=STYLES[arm]
        legend.append(Line2D([],[],color=st['color'],ls=st['ls'],marker=st['marker'],
                             markerfacecolor=st['color'] if st['filled'] else 'white',
                             markeredgecolor=st['color'],linewidth=1.7,markersize=5.5,label=st['label']))
    # Arrange a three-column legend as baseline / Guard15 weights / Exclusive15 weights.
    blank=Line2D([],[],linestyle='none',label=' ')
    fig.legend(handles=[legend[0],blank,legend[1],legend[2],legend[3],legend[4]],
               loc='upper left',bbox_to_anchor=(.071,.895),ncol=3,frameon=False,
               handlelength=2.3,columnspacing=2.5,labelspacing=.8,fontsize=10)
    x=np.array(BUDGETS)/1000
    titles=[r'Hamiltonian $H$',r'Energy $F$',r'2-RDM $D^{(2)}$']
    labels=[r'$H$ RMSE (mHa)',r'$F$ RMSE (mHa)',r'RMS Frobenius error']
    molecule_labels=[r'N$_2$  |  $R=1.10$ Å',r'CO  |  $R=1.128$ Å']
    for i,system in enumerate(SYSTEMS):
        for j,metric in enumerate(METRICS):
            ax=axes[i,j]
            for arm in ARMS:
                st=STYLES[arm]
                data=[lookup[system,b,arm,metric] for b in BUDGETS]
                y=np.array([d['rmse'] for d in data]);lo=np.array([d['ci95_low'] for d in data]);hi=np.array([d['ci95_high'] for d in data])
                ax.errorbar(x,y,yerr=np.vstack((y-lo,hi-y)),color=st['color'],ls=st['ls'],
                            marker=st['marker'],markersize=5.,mfc=st['color'] if st['filled'] else 'white',
                            mec=st['color'],mew=1.1,linewidth=1.6,elinewidth=.95,capsize=2.8,capthick=.95,
                            zorder=3 if st['filled'] else 2)
            ax.set_xscale('log',base=2);ax.set_xlim(26,278)
            ax.set_xticks(x,labels=['30K','60K','120K','240K']);ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='x',labelbottom=True)
            ax.set_ylim(bottom=0);ax.grid(axis='y',color='#d7dde3',linewidth=.65,alpha=.7,zorder=0)
            ax.set_axisbelow(True);ax.set_ylabel(labels[j])
            if i==1:ax.set_xlabel('Total measurement shots')
            ax.set_title(f'({chr(97+3*i+j)})  '+titles[j],loc='left',pad=10,fontweight='medium')
            if j==0:ax.text(0,1.20,molecule_labels[i],transform=ax.transAxes,fontsize=12,fontweight='semibold')
            if j==2:
                ax.yaxis.set_major_locator(MultipleLocator(.05))
                ax.yaxis.set_major_formatter(ScalarFormatter())
    fig.text(.075,.066,'Points: RMSE across 8 streams. Error bars: pointwise 95% percentile bootstrap CIs (100,000 resamples).',fontsize=9.5,color='#48515a')
    fig.text(.075,.039,'2-RDM: RMS of the unnormalized Frobenius error. All subset arms include 15K pilot shots in the total budget.',fontsize=9.5,color='#48515a')
    fig.savefig(OUT/'equilibrium_rmse_errorbars.png',dpi=240)
    fig.savefig(OUT/'equilibrium_rmse_errorbars.svg')
    plt.close(fig)
    (OUT/'caption.txt').write_text(
        'Equilibrium N2 (R=1.10 Angstrom) and CO (R=1.128 Angstrom), cc-pVDZ, CAS(10e,8o).\n'
        'H and F RMSEs are shown separately. The 2-RDM quantity is sqrt(mean(||D_est-D_ref||_F^2)),\n'
        'not the mean norm used in the earlier summary. Points use all eight streams r0-r7.\n'
        'Error bars are pointwise 95% percentile bootstrap confidence intervals (100000 paired stream resamples).\n'
        'All curves use a fixed 30-frame Haar real same-spin pool and spin-constrained E.\n'
        'Guard15 and its complementary Exclusive15 each charge 30x500 pilot shots, reuse their 15 pilots,\n'
        'and equally allocate the remaining shots. Uniform30 uses no pilot. These intervals describe\n'
        'shot-stream variation conditional on this pool; n=8 is small and the intervals are approximate.\n')
    print('Saved new PNG/SVG figure, 120 RMSE/CI rows and 320 raw stream records to',OUT,flush=True)

if __name__=='__main__':main()
