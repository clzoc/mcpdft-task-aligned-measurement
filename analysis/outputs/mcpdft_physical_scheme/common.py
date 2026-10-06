"""Physical feature maps; no on-top functional derivatives are evaluated."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,svd
from scipy import sparse
P=Path(__file__).resolve().parent
SRC=P.parents[1]/'mcpdft_measurement_revision/guarded_mcpdft_shadow_protocol/evidence_revision'
sys.path.insert(0,str(SRC))
from run_evidence import j,c,base_design
from oracle_signals import build_reference,run_design
from constrained_shadow import contract_one_rdm,exterior_square,spin_orbital_rotation,AffineRDMObjective

def save(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def context(system):
    args=c._arguments([]) if system=='c2' else j._arguments([])
    args.solver_threads=4
    sel,exact=build_reference(system,args)
    obj=j.FtPBEEnergyObjective(sel,grid_level=1)
    old=P.parent/'mcpdft_revision_strategy'
    geom=dict(np.load(old/f'{system}_coordinate_geometry.npz'))
    return args,sel,exact,obj,geom

def rowspace(A,tol=1e-9):
    _,s,v=svd(A,full_matrices=False)
    keep=s>max(s[0]*tol,1e-11) if len(s) else np.array([],bool)
    return v[keep],s

def feature_maps(sel,obj,geo):
    """Maps act on independent upper-triangular D coordinates."""
    from constrained_shadow import _canonical_symmetric_eigenbasis
    n=sel.n_spatial_orbitals;npair=len(sel.pairs);rows,cols=geo['rows'],geo['cols'];d=len(rows)
    assert np.array_equal(rows,obj.rows) and np.array_equal(cols,obj.cols)
    # Diagonalize projected bond-axis position. Degenerate subspaces use a
    # deterministic canonical gauge rather than finite Boys optimizer noise.
    with sel.molecule.with_common_origin((0,0,0)):
        zint=sel.molecule.intor('int1e_r',comp=3)[2]
    centers,U=_canonical_symmetric_eigenbasis(sel.active_mo_coeff.T@zint@sel.active_mo_coeff,
        degeneracy_tolerance=1e-7)
    for k in range(n):
        if U[np.argmax(abs(U[:,k])),k]<0:U[:,k]*=-1
    atom=np.zeros(n,dtype=int);atom[n//2:]=1
    R=exterior_square(spin_orbital_rotation(U),sel.pairs)
    lookup={pair:i for i,pair in enumerate(sel.pairs)}
    def linear(M):return M[rows,cols]+np.where(rows!=cols,M[cols,rows],0)
    def pairrow(i,j):
        M=np.outer(R[:,i],R[:,j]);return linear(M)
    gamma=[]
    for a,b in zip(rows,cols):
        D=np.zeros((npair,npair));D[a,b]=D[b,a]=1
        G=contract_one_rdm(D,2*n,sel.n_electrons,sel.pairs)
        gamma.append((G[:n,:n]+G[n:,n:]).ravel())
    gamma=np.array(gamma).T
    def onerow(M):return M.ravel()@gamma
    doublons=[];hops=[];intra=[];exchange=[]
    for i in range(n):doublons.append(pairrow(lookup[(i,i+n)],lookup[(i,i+n)]))
    for i in range(n):
        for k in range(i+1,n):
            v=2*pairrow(lookup[(i,i+n)],lookup[(k,k+n)])
            (hops if atom[i]!=atom[k] else intra).append(v)
            # Hermitian exchange coherence, sign convention explicitly retained.
            exchange.append(2*pairrow(lookup[(i,k+n)],lookup[(k,i+n)]))
    # Q=N_A-N_B. Q^2=sum n_i + 2 sum_{i<j} q_i q_j n_i n_j.
    q=np.tile(1-2*atom,2)
    Q2=onerow(np.eye(n))
    for k,(i,l) in enumerate(sel.pairs):Q2+=2*q[i]*q[l]*pairrow(k,k)
    groups={'density':gamma,'doublon':np.array(doublons),'charge_fluctuation':Q2[None,:],
            'pair_CT':np.array(hops),'intra_pair':np.array(intra),'exchange':np.array(exchange),
            'contact':obj.d2_to_quartic}
    groups['density_contact']=np.vstack([gamma,obj.d2_to_quartic])
    groups['density_doublon']=np.vstack([gamma,groups['doublon']])
    groups['density_charge']=np.vstack([gamma,Q2])
    groups['density_pair_CT']=np.vstack([gamma,groups['pair_CT']])
    # Euclidean coordinates represent the D Frobenius norm (offdiag sqrt(2)).
    lift=geo['Z']/geo['scale'][:,None]
    maps={};meta={}
    for name,A in groups.items():
        W,s=rowspace(A@lift);maps[name]=W
        meta[name]={'rank':len(W),'singular_values':s.tolist()}
    maps['all']=np.eye(lift.shape[1]);meta['all']={'rank':lift.shape[1]}
    from pyscf import symm
    return maps,dict(groups=meta,local_rotation=U.tolist(),local_z_centers=centers.tolist(),
        orbital_irreps=list(map(int,sel.orbital_irreps)),
        orbital_irrep_names=[symm.irrep_id2name(sel.molecule.groupname,int(x)) for x in sel.orbital_irreps],
        atom_assignment=atom.tolist(),localization='projected z eigenvectors, canonical degenerate gauge, columns sign fixed'),gamma

def matrix(theta,geo,npair):
    D=np.zeros((npair,npair));D[geo['rows'],geo['cols']]=theta
    D[geo['cols'],geo['rows']]=theta;return D

def affine_map(L,geo,npair,offset):
    nr,nc=L.shape
    map_=sparse.coo_matrix((L.ravel(),(np.repeat(np.arange(nr),nc),
        np.tile(geo['rows']*npair+geo['cols'],nr))),shape=(nr,npair*npair)).tocsr()
    return AffineRDMObjective(d2_map=map_,offset=offset,name='physical_moment_protection')
