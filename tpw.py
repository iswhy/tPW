from dataclasses import dataclass
import math, scipy, time
import numpy as np
from aopwints_xa import integral_mix_edit as aopw_xa
import cint_gpw as aopw_xn
from pyscf import scf, gto, ao2mo
from pyscf.tools import fcidump


#=============================================================
INTS_ALGO = 'xa'
#=============================================================


#=============================================================
@dataclass
class INT2e:
    index: tuple[int, int, int, int]
    value: complex
#============================================================


#============================================================
def set_algo(a):
    '''
    Sets the algorithm type to calculate the cross integrals between the AOs and
    PWs. Without setting the algorithm type, the default is "xa".

    a:
      A string specifying the algorithm type. It accepts "xa" or "xn".
    '''
    
    assert a.lower() == 'xa' or a.lower() == 'xn', \
        f'set_algo only accepts "xa" or "xn", but "{a}" is given.'
    global INTS_ALGO
    INTS_ALGO = a
    return
#============================================================


#============================================================
def ovlp_tpw_1d(kvec, xl, xr, w):
    '''
    Calculates the overlap integral matrix among 1D tapered PWs.

    kvec:
      k points of each tPW.
    xl and xr:
      The positions of the left and right walls of the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.

    Returns:
      A 3D array Sreg, where Sreg[0], Sreg[1], and Sreg[2] contain overlap
      matrices in the left, center, and right regions, respectively. Sum these
      three matrices to get the total overlap matrix integrated over the entire
      axis.
    '''
    
    norm = xr - xl + 2*w
    Sreg = np.zeros((3, len(kvec), len(kvec)), dtype=complex)
    for i, k1 in enumerate(kvec):
        for j, k2 in enumerate(kvec[:i+1]):
            q = k2 - k1
            el = np.exp(1j*q*xl)
            er = np.exp(1j*q*xr)
            Sreg[0][i,j] = el / (1/w + 1j*q)
            Sreg[0][j,i] = Sreg[0][i,j].conj()
            Sreg[2][i,j] = er / (1/w - 1j*q)
            Sreg[2][j,i] = Sreg[2][i,j].conj()
            if i != j:
                Sreg[1][i,j] = (er-el) / (1j*q)
                Sreg[1][j,i] = Sreg[1][i,j].conj()
            else:
                Sreg[1][i,i] = xr - xl

    return Sreg / norm
#============================================================


#============================================================
def kin_tpw_1d(kvec, w, *, Sreg=None, xl=None, xr=None):
    '''
    Calculates the kinetic energy matrix among 1D tapered PWs.

    kvec:
      k points of each tPW.
    xl and xr:
      The positions of the left and right walls of the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    Sreg:
      A 3D array representing the overlap matrices among 1D tPWs in the left, middle
      (pure PW), and right intervals, respectively. It can be obtained by calling
      ovlp_tpw_1d. If not given, xl and xr must be given.

    Returns:
      The kinetic energy matrix in 1D tPW representation.
    '''

    if Sreg is None:
        if all([var is not None for var in (xl, xr, w)]):
            Sreg = ovlp_tpw_1d(kvec, xl, xr, w)
        else:
            raise ValueError('If Sreg is not given or is None, all of xl, xr, and w' +
                             'must be given.')
        
    T = \
        np.outer(1/2/w-1j*kvec, 1/2/w+1j*kvec) * Sreg[0] + \
        np.outer(kvec, kvec) * Sreg[1] + \
        np.outer(-1/2/w-1j*kvec, -1/2/w+1j*kvec) * Sreg[2]
    return T / 2
#============================================================


#============================================================
def assemble_ks(kxvec, kyvec, kzvec, verb=0):
    '''
    Assembles a 3D k vector array from three 1D k vector arrays that corresponds to
    the x, y, and z components.
    '''
    
    if verb >= 1:
        print('Momentum intervals (kx, ky, kz) = %.6f, %.6f, %.6f'%
              (kxvec[1]-kxvec[0] if Nkx > 1 else 0.0,
               kyvec[1]-kyvec[0] if Nky > 1 else 0.0,
               kzvec[1]-kzvec[0] if Nkz > 1 else 0.0),
              ' a.u.', flush=True)

    kx1 = np.ones_like(kxvec)
    ky1 = np.ones_like(kyvec)
    kz1 = np.ones_like(kzvec)
    return np.hstack((np.kron(np.kron(kxvec,ky1),kz1)[:,None],
                      np.kron(np.kron(kx1,kyvec),kz1)[:,None],
                      np.kron(np.kron(kx1,ky1),kzvec)[:,None]))
#============================================================


#============================================================
def MO_transform(A_ao, cf):
    '''
    Transforms a full AO+tPW matrix into a full MO+tPW matrix using a set of MOs.
    '''
    
    nao, nmo = cf.shape
    nk = A_ao.shape[0] - nao
    nb = nmo + nk
    
    A_mo = np.zeros((nb,nb), dtype=complex)
    # Full-transform the MO diagonal block
    A_mo[:nmo,:nmo] = np.einsum('ij,jk,kl->il', cf.T, A_ao[:nao,:nao], cf, optimize=True)
    # Half-transform the upper off-diagonal block
    A_mo[:nmo,nmo:] = np.einsum('ij,jk->ik', cf.T, A_ao[:nao,nao:], optimize=True)
    # Half-transform the lower off-diagonal block
    A_mo[nmo:,:nmo] = A_mo[:nmo,nmo:].conj().T
    # Copy the tPW diagonal block
    A_mo[nmo:,nmo:] = A_ao[nao:,nao:].copy()

    return A_mo
#============================================================


#============================================================
def get_ovlp_cross(mol, pw_k, xlim, ylim, zlim, w):
    '''
    Compute the AO-PW overlap ``<G_a|P_k>`` matrix

    mol:
      PySCF mole object.
    pw_k:
      The k vectors arranged into rows (there are three columns with each row
      corresponding to a particular k vector).
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    
    Returns:
      AO-PW overlaps matrix with shape (nao, npw).
    '''
    
    Volume = (xlim[1]-xlim[0]+2*w) * (ylim[1]-ylim[0]+2*w) * (zlim[1]-zlim[0]+2*w)
    if INTS_ALGO == 'xa':
        return aopw_xa.ao_pw_ovlp(mol, pw_k) / np.sqrt(Volume)
    elif INTS_ALGO == 'xn':
        with aopw_xn.prepare(mol) as opt:
            return aopw_xn.ovlp(mol, pw_k, opt=opt).T / np.sqrt(Volume)
#============================================================


#============================================================
def get_ovlp_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w):
    Sx_tpw = np.sum( ovlp_tpw_1d(kxvec, xlim[0], xlim[1], w), axis=0 )
    Sy_tpw = np.sum( ovlp_tpw_1d(kyvec, ylim[0], ylim[1], w), axis=0 )
    Sz_tpw = np.sum( ovlp_tpw_1d(kzvec, zlim[0], zlim[1], w), axis=0 )

    return np.kron(np.kron(Sx_tpw, Sy_tpw), Sz_tpw)     # kz changes first, then ky, then kx.
#============================================================


#============================================================
def get_ovlp(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, cf=None):
    '''
    Computes the overlap matrix among AOs+tPWs.

    mol:
      PySCF mole object.
    kxvec, kyvec, kzvec:
      The x, y, and z components of the k vectors of the tPWs.
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    cf:
      MO coefficients, if provided, the AO parts of the matrix are transformed into
      its MO representation.

    Return:
      The overlap matrix among AOs+tPWs. The first nao rows (columns) corresponds to
      the AOs, and the remaining rows (columns) to the tPWs.
    '''
    
    nao = mol.nao
    pw_k = assemble_ks(kxvec, kyvec, kzvec)
    nb = nao + len(pw_k)
    S = np.zeros((nb,nb), dtype=complex)

    S[:nao, :nao] = mol.intor('int1e_ovlp_sph')

    S[:nao, nao:] = get_ovlp_cross(mol, pw_k, xlim, ylim, zlim, w)
    S[nao:, :nao] = S[:nao, nao:].conj().T
    
    S[nao:, nao:] = get_ovlp_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w)

    if cf is not None:
        return MO_transform(S, cf)   # The MO diag block should be an identity matrix since MOs are orthonormal.
    else:
        return S
#============================================================


#============================================================
def get_kin_cross_cart(mol, pw_k, xlim, ylim, zlim, w):
    '''
    Computes the Cartesian components of the kinetic energy matrix between AOs and
    PWs, i.e. <G_a|T_i|P_k> where i=x,y,z. Summing these three component matrices
    yields an identical matrix as that returned by get_kin_cross.

    mol:
      PySCF mole object.
    pw_k:
      The k vectors arranged into rows (there are three columns with each row
      corresponding to a particular k vector).
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.

    Returns:
      The kinetic energy matrices between AOs and PWs. The shape is (3,nao,npw), with
      the first dimension associated to the Cartesian components of the kinetic
      energy operator.
    '''
    
    S_cross = get_ovlp_cross(mol, pw_k, xlim, ylim, zlim, w)
    nao = mol.nao
    return np.array( [np.outer(np.ones(nao), pw_k[:,i]**2/2) * S_cross for i in range(3)] )
#============================================================


#============================================================
def get_kin_cross(mol, pw_k, xlim, ylim, zlim, w):
    '''
    Compute the AO-PW kinetic energy ``<G_a|T|P_k>`` matrix

    mol:
      PySCF mole object.
    pw_k:
      The k vectors arranged into rows (there are three columns with each row
      corresponding to a particular k vector).
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    
    Returns:
        AO-PW kinetic energy matrix with shape (nao, npw).
    '''
    
    Volume = (xlim[1]-xlim[0]+2*w) * (ylim[1]-ylim[0]+2*w) * (zlim[1]-zlim[0]+2*w)
    if INTS_ALGO == 'xa':
        pw_kin = 0.5 * np.einsum("pi,pi->p", pw_k, pw_k)
        return aopw_xa.ao_pw_ovlp(mol, pw_k) * pw_kin[None,:] / np.sqrt(Volume)
    elif INTS_ALGO == 'xn':
        with aopw_xn.prepare(mol) as opt:
            return aopw_xn.kin(mol, pw_k, opt=opt).T / np.sqrt(Volume)
#============================================================


#============================================================
def get_kin_tpw_cart(kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx=None, Sy=None, Sz=None):
    '''
    Computes the Cartesian components of the kinetic energy matrix among tPWs, i.e.
    <P_k|T_i|P_l> where i=x,y,z. Summing these three component matrices
    yields an identical matrix as that returned by get_kin_tpw.

    mol:
      PySCF mole object.
    kxvec, kyvec, kzvec:
      The x, y, and z components of the k vectors of the tPWs.
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    Sx, Sy, Sz:
      The overlap matrices among the x, y, and z parts of the tPWs, respectively.
      They can be calculated using ovlp_tpw_1d.

    Returns:
      The kinetic energy matrices among tPWs. The shape is (3,npw,npw), with
      the first dimension associated to the Cartesian components of the kinetic
      energy operator.
    '''
    
    if Sx is None:
        Sx = ovlp_tpw_1d(kxvec, xlim[0], xlim[1], w)
    Tx_tpw = kin_tpw_1d(kxvec, w, Sreg=Sx)
    Sx = np.sum(Sx, axis=0)   # ATT: Sx is now summed.

    if Sy is None:
        Sy = ovlp_tpw_1d(kyvec, ylim[0], ylim[1], w)
    Ty_tpw = kin_tpw_1d(kyvec, w, Sreg=Sy)
    Sy = np.sum(Sy, axis=0)   # ATT: Sy is now summed.

    if Sz is None:
        Sz = ovlp_tpw_1d(kzvec, zlim[0], zlim[1], w)
    Tz_tpw = kin_tpw_1d(kzvec, w, Sreg=Sz)
    Sz = np.sum(Sz, axis=0)   # ATT: Sz is now summed.
    
    return np.array([
        np.kron(np.kron(Tx_tpw, Sy), Sz),
        np.kron(np.kron(Sx, Ty_tpw), Sz),
        np.kron(np.kron(Sx, Sy), Tz_tpw) ])
#============================================================


#============================================================
def get_kin_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx=None, Sy=None, Sz=None):
    '''
    Computes the Cartesian components of the kinetic energy matrix among tPWs, i.e.
    <P_k|T|P_l>.

    kxvec, kyvec, kzvec:
      The x, y, and z components of the k vectors of the tPWs.
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    Sx, Sy, Sz:
      The overlap matrices among the x, y, and z parts of the tPWs, respectively.
      They can be calculated using ovlp_tpw_1d.

    Returns:
      The kinetic energy matrix among tPWs.
    '''

    return np.sum(get_kin_tpw_cart(kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx, Sy, Sz), axis=0)
#============================================================


#============================================================
def get_kin_cart(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx=None, Sy=None, Sz=None, cf=None):
    '''
    Computes the Cartesian components of the kinetic energy matrix among AOs+tPWs.
    Summing these three component matrices yields an identical matrix as that
    returned by get_kin.

    mol:
      PySCF mole object.
    kxvec, kyvec, kzvec:
      The x, y, and z components of the k vectors of the tPWs.
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    cf:
      MO coefficients, if provided, the AO parts of the matrix are transformed into
      its MO representation.
    Sx, Sy, Sz:
      The overlap matrices among the x, y, and z parts of the tPWs, respectively.
      They can be calculated using ovlp_tpw_1d.

    Return:
      The kinetic energy matrices among AOs+tPWs. The shape is (3,nao+npw,nao+npw),
      with the first dimension associated to the Cartesian components of the kinetic
      energy operator. For the 2nd and 3rd dimensions, the first nao rows (columns)
      corresponds to the AOs, and the remaining rows (columns) to the tPWs.
    '''

    nao = mol.nao
    pw_k = assemble_ks(kxvec, kyvec, kzvec)
    nb = nao + len(pw_k)
    T = np.zeros((3,nb,nb), dtype=complex)

    T[:, :nao, :nao] = - np.array( [mol.intor('int1e_ipipovlp', comp=9).reshape(3, 3, nao, nao)[i,i,:,:]
                                    for i in range(3)] ) / 2
    #^^ ip is nabla, not -i.nabla, which is the momentum. Hence ipip needs a -1 to become pp.

    T[:, :nao, nao:] = get_kin_cross_cart(mol, pw_k, xlim, ylim, zlim, w)
    T[:, nao:, :nao] = T[:, :nao, nao:].conj().transpose(0, 2, 1)

    T[:, nao:, nao:] = get_kin_tpw_cart(kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx, Sy, Sz)

    if cf is not None:
        return np.array( [MO_transform(T_ao, cf) for T_ao in T] )
    else:
        return T
#============================================================


#============================================================
def get_kin(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx=None, Sy=None, Sz=None, cf=None):
    '''
    Computes the kinetic energy matrix among AOs+tPWs.

    mol:
      PySCF mole object.
    kxvec, kyvec, kzvec:
      The x, y, and z components of the k vectors of the tPWs.
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    cf:
      MO coefficients, if provided, the AO parts of the matrix are transformed into
      its MO representation.

    Return:
      The kinetic energy matrix among AOs+tPWs. The first nao rows (columns)
      corresponds to the AOs, and the remaining rows (columns) to the tPWs.
    '''
    
    nao = mol.nao
    pw_k = assemble_ks(kxvec, kyvec, kzvec)
    nb = nao + len(pw_k)
    T = np.zeros((nb,nb), dtype=complex)

    T[:nao, :nao] = mol.intor('int1e_kin_sph')

    T[:nao, nao:] = get_kin_cross(mol, pw_k, xlim, ylim, zlim, w)
    T[nao:, :nao] = T[:nao, nao:].conj().T

    T[nao:, nao:] = get_kin_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx, Sy, Sz)

    if cf is not None:
        return MO_transform(T, cf)
    else:
        return T
#============================================================


#============================================================
def get_vnuc_cross(mol, pw_k, xlim, ylim, zlim, w):
    '''
    Compute the AO-PW nuclear attraction ``<G_a|V_{nuc}|P_k>`` matrix

    mol:
      PySCF mole object.
    pw_k:
      The k vectors arranged into rows (there are three columns with each row
      corresponding to a particular k vector).
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    
    Returns:
        AO-PW nuclear attraction matrix with shape (nao, npw).
    '''
    
    Volume = (xlim[1]-xlim[0]+2*w) * (ylim[1]-ylim[0]+2*w) * (zlim[1]-zlim[0]+2*w)
    if INTS_ALGO == 'xa':
        return aopw_xa.ao_pw_pot(mol, pw_k) / np.sqrt(Volume)
    elif INTS_ALGO == 'xn':
        with aopw_xn.prepare(mol) as opt:
            return aopw_xn.nuc(mol, pw_k, opt=opt).T / np.sqrt(Volume)
#============================================================


#============================================================
def get_vnuc(mol, pw_k, xlim, ylim, zlim, w, cf=None):
    '''
    Computes the nuclear attraction matrix among AOs+tPWs.

    mol:
      PySCF mole object.
    kxvec, kyvec, kzvec:
      The x, y, and z components of the k vectors of the tPWs.
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    cf:
      MO coefficients, if provided, the AO parts of the matrix are transformed into
      its MO representation.

    Return:
      The nuclear attraction matrix among AOs+tPWs. The first nao rows (columns)
      corresponds to the AOs, and the remaining rows (columns) to the tPWs.
    '''

    nao = mol.nao
    nk = len(pw_k)
    nb = nao + nk
    V = np.zeros((nb,nb), dtype=complex)

    V[:nao, :nao] = mol.intor('int1e_nuc_sph')

    V[:nao, nao:] = get_vnuc_cross(mol, pw_k, xlim, ylim, zlim, w)
    V[nao:, :nao] = V[:nao, nao:].conj().T

    #V[nao:, nao:] = get_vnuc_tpw()

    if cf is not None:
        return MO_transform(V, cf)
    else:
        return V
#============================================================


#============================================================
def get_v2e(mol, pw_k, xlim, ylim, zlim, w, types, cf=None):

    '''
    Computes 2-electron integrals of the type:
      1) bbbb, which gives (G_a G_b|G_c G_d), or
      2) kbbb, which gives (P_k G_a|G_b G_c)
    where P_k is a tPW and G_a is a GTO.

    pw_k:
      The k vectors arranged into rows (there are three columns with each row
      corresponding to a particular k vector).
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    cf:
      MO coefficients, if provided, the AO parts of the integrals are transformed into
      its MO representation.
    types:
      A 1D array-like object with elements 'bbbb' or 'kbbb'. It specifies which integral types
      are requested.

    Return:
      A dictionary with elements corresponding to a particular integral type, 'bbbb' or 'kbbb'.
      The dicionary value for all integral types is a list of independent INT2e objects ordered
      so that the 4-th index runs the fastest, then the 3rd, then the 2nd, and finally the 1st.
      For the 'bbbb' and 'kbbb' types, the list contains npair*(npair+1)/2 and nk*nmo*npair
      elements, respectively, where npair=nmo*(nmo+1)/2, nk is the number of k vectors in pw_k, 
      nmo is the number of AOs (or MOs).
    '''
    
    nao = mol.nao
    nk = len(pw_k)
    Volume = (xlim[1]-xlim[0]+2*w) * (ylim[1]-ylim[0]+2*w) * (zlim[1]-zlim[0]+2*w)
    types = list(set(types))
    
    g2e = {}
    for i, v2e_type in enumerate(types):
        if v2e_type == 'bbbb':
            # For the bbbb type, the 2e integrals are calculated through PySCF.
            if cf is None:
                nmo = mol.nao
                #OLD aopw_xa.int_2e(mol, pw_k, set(['bbbb']))
                mmmm = mol.intor('int2e_sph', aosym="s8")
                #^^ mmmm is a vector of size npair*(npair+1)/2, where npair=nmo*(nmo+1)/2.
                #Only the independent elements according to PySCF symmetry 's8' are stored
                #in mmmm. 
            else:
                nmo = cf.shape[1]
                mmmm = ao2mo.kernel(mol, cf, intor='int2e_sph')
                #^^ mmmm is a square matrix of size npair x npair, where npair=nmo*(nmo+1)/2,
                # which corresponds to PySCF's 's4' symmetry.
            g2e['bbbb'] = []
            nu = 0
            for p in range(nmo):
                for q in range(p+1):
                    pq = round(p*(p+1)/2) + q
                    for r in range(nmo):
                        for s in range(r+1):
                            rs = round(r*(r+1)/2) + s

                            if pq >= rs:
                                if cf is None:
                                    g2e['bbbb'].append(INT2e((p, q, r, s), mmmm[nu]))
                                else:
                                    g2e['bbbb'].append(INT2e((p, q, r, s), mmmm[pq,rs]))
                                nu += 1

        elif v2e_type == 'kbbb':
            # For the kbbb type, the 2e integrals are calculated through either xa or xn codes.
            if INTS_ALGO == 'xa':
                g2e['kbbb'] = aopw_xa.int_2e(mol, pw_k, set(['kbbb']), pw_norm=1/np.sqrt(Volume))
                #^^ int_2e returns (P_k G_a|G_b G_c) in an unsorted list.
            elif INTS_ALGO == 'xn':
                with aopw_xn.prepare(mol) as opt:
                    bkbb = aopw_xn.gpgg(mol, pw_k, opt=opt)          
                    #^^ gpgg returns (G_a P_k|G_b G_c) with shape (nk, nao, nao, nao),
                    # but we want (P_k G_a|G_b G_c) in a list.
                g2e['kbbb'] = []
                for k in range(nk):
                    for p in range(nao):
                        for q in range(nao):
                            for r in range(q+1):
                                g2e['kbbb'].append(INT2e((nao+k, p, q, r),
                                                         bkbb[k,p,q,r].conj()/np.sqrt(Volume)))
            sortid = aopw_xa.sort_kbbb(g2e['kbbb'], nao)
            if cf is None:
                g2e['kbbb'] = [g2e['kbbb'][i] for i in sortid]
            else:
                g2e['kbbb'] = aopw_xa.int2e_ao2mo_kbbb(cf, [g2e['kbbb'][i] for i in sortid],
                                                       len(pw_k))
        else:
            raise ValueError("Only 'bbbb' and 'kbbb' integral types are currently " +
                             "supported by get_v2e.")

    return g2e
#============================================================


#============================================================
def get_momentum_cross(mol, pw_k, xlim, ylim, zlim, w):
    '''
    Compute the AO-PW overlap ``<G_a|P_i|P_k>`` matrix for i=x,y,z.

    mol:
      PySCF mole object.
    pw_k:
      The k vectors arranged into rows (there are three columns with each row
      corresponding to a particular k vector).
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    
    Returns:
        AO-PW momentum matrices with shape (3,nao,npw).
    '''

    S_cross = get_ovlp_cross(mol, pw_k, xlim, ylim, zlim, w)
    nao = mol.nao
    return np.array( [np.outer(np.ones(nao), pw_k[:,i]) * S_cross for i in range(3)] )
#============================================================


#============================================================
def get_momentum_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w, S_tpw=None):

    pw_k = assemble_ks(kxvec, kyvec, kzvec)
    nk = len(pw_k)
    if S_tpw is None:
        S_tpw = get_ovlp_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w)
    kfactor = np.zeros((3,nk,nk))
    for i in range(3):
        kfactor[i] = np.outer(np.ones(nk), pw_k[:,i])
        kfactor[i] = (kfactor[i] + kfactor[i].T) / 2
    
    return np.array( [kfactor[i]*S_tpw for i in range(3)] )
#============================================================


#============================================================
def get_momentum(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, S_tpw=None, cf=None):
    '''
    Computes the momentum matrix among AOs+tPWs.

    mol:
      PySCF mole object.
    kxvec, kyvec, kzvec:
      The x, y, and z components of the k vectors of the tPWs.
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    cf:
      MO coefficients, if provided, the AO parts of the matrix are transformed into
      its MO representation.

    Return:
      The momentum matrices among AOs+tPWs. The shape is (3,nao+npw,nao+npw), with
      the first dimension associated to the Cartesian components of the momentum
      operator. For the 2nd and 3rd dimensions, the first nao rows (columns)
      corresponds to the AOs, and the remaining rows (columns) to the tPWs.
    '''
    
    nao = mol.nao
    pw_k = assemble_ks(kxvec, kyvec, kzvec)
    nb = nao + len(pw_k)
    P = np.zeros((3,nb,nb), dtype=complex)

    P[:, :nao, :nao] = -1j * mol.intor('int1e_ipovlp')

    P[:, :nao, nao:] = get_momentum_cross(mol, pw_k, xlim, ylim, zlim, w)
    P[:, nao:, :nao] = P[:, :nao, nao:].conj().transpose(0,2,1)

    P[:, nao:, nao:] = get_momentum_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w, S_tpw)

    if cf is not None:
        return np.array( [MO_transform(P_ao, cf) for P_ao in P] )
    else:
        return P
#============================================================


#============================================================
def get_dipole_cross(mol, pw_k, xlim, ylim, zlim, w):
    '''
    Compute the AO-PW overlap ``<G_a|d|P_k>`` matrix for d=x,y,z.

    mol:
      PySCF mole object.
    pw_k:
      The k vectors arranged into rows (there are three columns with each row
      corresponding to a particular k vector).
    xlim, ylim, zlim:
      Each is a tuple of two elements that defines the volume for the pure PW region.
    w:
      The 1/e width of the exponential tails of the tPWs.
    
    Returns:
        AO-PW dipole matrices with shape (3,nao,npw).
    '''

    Volume = (xlim[1]-xlim[0]+2*w) * (ylim[1]-ylim[0]+2*w) * (zlim[1]-zlim[0]+2*w)
    if INTS_ALGO == 'xa':
        return aopw_xa.ao_pw_dipole(mol, pw_k) / np.sqrt(Volume)
    elif INTS_ALGO == 'xn':
        with aopw_xn.prepare(mol) as opt:
            #vv aopw_xn.dipole returns (nk,3,nao) while we want (3,nao,nk).
            return aopw_xn.dipole(mol, pw_k, opt=opt).transpose(1,2,0) / np.sqrt(Volume)
#============================================================


#============================================================
def projected_tpw(mol, pw_k, xlim, ylim, zlim, w, cf, verb=0):

    if verb >= 1:
        print(f'Computing <a|k> integrals with nAO = {mol.nao} and nPW = {len(pw_k)}. ' + 
              'Please wait ...', flush=True)
    S_cross_ao = get_ovlp_cross(mol, pw_k, xlim, ylim, zlim, w)

    return -1 * np.einsum('ij, ik -> jk', cf, S_cross_ao, optimize=True)
#============================================================


#============================================================
def get_symm_X(S, max_order=10, tol=1E-10, enforce=True):
    '''
    Calculates the transformation matrix in the symmetric orthogonalization method using
    matrix iteration.
    '''
    
    norm = np.sqrt(np.diag(S))
    S_norm = np.diag(1/norm) @ S @ np.diag(1/norm)
    E = S_norm - np.eye(S_norm.shape[0])
    
    X = np.eye(S_norm.shape[0], dtype=complex)
    for i in range(1,max_order+1):
        correction = E.copy()
        if i > 1:
            correction = correction @ E
        cc = (-1)**i * comb(2*i, i) / 4**i
        X += cc * correction
        error_series = np.linalg.norm(correction)
        error_ortho = np.linalg.norm(X.conj().T @ S_norm @ X - np.eye(S_norm.shape[0]))
        isconv = error_series < tol and error_ortho < tol
        if isconv:
            break

    conv_msg = 'Matrix iteration has reached the specified limit but convergence ' + \
            'has not been attained.'
    if enforce:
        assert isconv, 'ERROR: ' + conv_msg
    else:
        if not isconv: print('WARNING: ' + conv_msg)
        
    return X
#============================================================


#============================================================
def orthogonalized_tpw(ortho, e_proj, canon_thr=1E-4, verb=0, *, S_tpw=None, kxvec=None, 
                      kyvec=None, kzvec=None, xlim=None, ylim=None, zlim=None, w=None):

    if verb >= 1:
        print(f'Performing canonical orthogonalization of projected PWs. Please wait ...',
          flush=True)
        
    if S_tpw is None:
        if all([var is not None for var in (kxvec, kyvec, kzvec, xlim, ylim, zlim, w)]):
            time0 = time.time()
            S_tpw = get_ovlp_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w)
            print('S_tpw elapsed time = %.3f s' % (time.time() - time0), flush=True)
        else:
            raise ValueError('If S_tpw is not given or is None, all of kxvec, kyvec, kzvec, ' +
                             'xlim, ylim, zlim, and w must be given.')

    S_cross_mo = -e_proj
    S_proj = S_tpw - S_cross_mo.conj().T @ S_cross_mo    # S_proj is the overlap matrix among projected TPWs.
    
    #vv These orthogonalization methods also orthonormalize the output vectors regardless
    # of the normalization of the input vectors.
    if ortho == 'canon':
        s, U = scipy.linalg.eigh(S_proj)
        X = U[:,s>=canon_thr] @ np.diag([1/np.sqrt(s0) for s0 in s[s>=canon_thr]])
    elif ortho == 'symm':
        X = get_symm_X(S_proj)
        #X = U @ np.diag([1/np.sqrt(s0) for s0 in s]) @ U.conj().T
    else:
        raise ValueError(f'Unavailable option specified for ortho, ortho = {ortho}.')
    #print('ovl_kp = ', X.conj().T @ S_proj @ X)

    if verb >= 1:
        print('No. of discarded projected plane waves = ', S_proj.shape[1] - X.shape[1],
              flush=True)
    
    return X, np.einsum('ij, jk -> ik', e_proj, X, optimize=True)
#============================================================


#============================================================
def get_coupling(cf, S_cross_ao, T_mol_ao, T_cross_ao, V_mol_ao, V_cross_ao, g2e_mmmm,
                 g2e_kbbb, X, efield=None, D_mol_ao=None, D_cross_ao=None):
    '''
    efield:
      The three Cartesian components of electric field vector in a.u.
    D_mol_ao:
      Dipole matrix components in AO basis. Only used/computed if efield is given. The
      shape is (3,nao,nao).
    D_cross_ao:
      Dipole matrix components between AO and PW, <G_a|d|P_k> where d=x,y,z. Only used/
      computed if efield is given. The shape is (3,nao,npw).
    g2e_kbbb:
      2e integral objects INT2e of the type (P_k G_a|G_b G_c) (involves one PW and three
      AOs) arranged into a 1D list. c runs the fastest and k runs the slowest. k runs over
      all PWs, a and b run over all AOs, and 0<=c<=b. It can be obtained by calling get_v2e()
      with types=['kbbb']. 
    g2e_mmmm:
      2e integrals involving only MOs arranged into a matrix of shape (nmo*(nmo+1)/2, nmo^2)
      where nmo is the number of MOs in cf. The rows correspond to independent pairs (r,s)
      with 0<=s<=r and s runs faster than r. The columns correspond to all pairs (p,q) with p
      and q running from 0 to nmo-1. It is typically obtained by calling pyscf.ao2mo.kernel
      with intor='int2e_sph' and aosym='s2ij'.
    '''

    nao, nmo = cf.shape
    
    #==== Computation of 1e operator matrix ====#
    S_cross_mo = np.einsum('ij, ik -> jk', cf, S_cross_ao, optimize=True)
    h1e_mol_mo = np.einsum('ij, ik, kl -> jl', cf, T_mol_ao + V_mol_ao, cf, optimize=True)
    h1e_cross_mo = np.einsum('ij, ik -> jk', cf, T_cross_ao + V_cross_ao, optimize=True)
    #later dip_ao = get_dip_cross(mol, pw_k, op='r')
    g2e_mmmm = g2e_mmmm.reshape((nmo*g2e_mmmm.shape[0], nmo), order='F')
    #^^ At input, g2e_mmmm is a matrix of shape (nmo*(nmo+1)/2, nmo^2), so, we use order='F' to match
    # the index ordering of the rows of g2e_mmmm to that of g2e_mmmk. In this reordering, the first nmo
    # columns are stacked on top of each other to form the first column of the reordered matrix. The
    # remaining columns are reordered accordingly.

        
    #==== Computation of (mm|mk) matrix from (ka|aa) ====#
    #g2e_mmmk = aopw_xa.int2e_ao2mo(cf, g2e_kbbb, nao, nk)
    kbbb_sort = aopw_xa.sort_kbbb(g2e_kbbb, nao)
    nk = S_cross_ao.shape[1]
    g2e_kmmm = aopw_xa.int2e_ao2mo_kbbb(cf, [g2e_kbbb[i] for i in kbbb_sort], nk)
    last_id = aopw_xa.get_kbbb_id(nmo, nmo+nk-1, nmo-1, nmo-1, nmo-1)
    assert len(g2e_kmmm) == last_id, f'The last kkbb ID is {last_id}, whereas the length of ' + \
        f'2e integrals having kbbb type is {len(g2e_kmmm)}.'
    

    #==== # (mm|mk) = (mm|km)* = (km|mm)* ====#
    sorted_id = aopw_xa.sort_kbbb(g2e_kmmm, nmo)
    n3b = round(len(g2e_kbbb)/nk)
    g2e_mmmk = np.array([g2e_kmmm[i].value.conj() for i in sorted_id]).reshape((n3b,nk), order='F')
    #^^ g2e_mmmk is a matrix where the columns identify PWs and the rows identify groups
    # of three MOs. Since k is the slowest moving index in g2e_kmmm, doing reshape((n3b,nk), order='F')
    # as above rearrange g2e_kmmm into a matrix where a given column is associated to a particular k and
    # a particular row corresponds to three-MO index (q,r,s) with q and r running from 0 to nmo-1 and s
    # from 0 to r.
            
    
    #==== Calculate the 1e and 2e parts of coupling matrix ====#
    h1e_cross_kp = (h1e_cross_mo - h1e_mol_mo @ S_cross_mo) @ X
    g2e_mmm_kp = (g2e_mmmk - g2e_mmmm @ S_cross_mo) @ X

    if efield is not None:
        dip_mol_mo = np.einsum('d, ij, dik, kl -> jl', efield, cf, D_mol_ao, cf, optimize=True)
        dip_cross_mo = np.einsum('d, ij, dik -> jk', efield, cf, D_cross_ao, optimize=True)
        dip_cross_kp = (dip_cross_mo - dip_mol_mo @ S_cross_mo) @ X
        return np.vstack((h1e_cross_kp, g2e_mmm_kp, dip_cross_kp))
    else:
        return np.vstack((h1e_cross_kp, g2e_mmm_kp))
#============================================================


#============================================================
def effective_tpw(d_ortho, e_ortho, cf, mol, sv_thr=-1, verb=0, *, S_cross_ao=None, T_cross_ao=None, 
                  V_cross_ao=None, g2e_kbbb=None, pw_k=None, xlim=None, ylim=None, 
                  zlim=None, w=None, efield=None, D_cross_ao=None):

    '''
    efield:
      The three Cartesian components of electric field vector in a.u.
    D_cross_ao:
      Only used/computed if efield is given.
    g2e_kbbb:
      2e integral objects INT2e of the type (P_k G_a|G_b G_c) (involves one PW and three
      AOs) arranged into a 1D list. c runs the fastest and k runs the slowest. k runs over
      all PWs, a and b run over all AOs, and 0<=c<=b. It can be obtained by calling get_v2e()
      with types=['kbbb'].
    '''

    T_mol_ao = mol.intor('int1e_kin_sph')
    V_mol_ao = mol.intor('int1e_nuc_sph')
    g2e_mmmm = ao2mo.kernel(mol, cf, intor='int2e_sph', aosym='s2ij')
    #^^ g2e_mmmm is a matrix, 's2ij' means the first two indices form independent pairs,
    # nao*(nao+1)/2. The last two indices  cover all pairs, nao*nao.
    #debug print('Shape of (mm|mm) = ', g2e_mmmm.shape)
    if efield is None:
        D_mol_ao = None
    else:
        D_mol_ao = mol.intor('int1e_r_sph')
    
    if verb >= 1:
        thistime = time.time()

    def _check_existence(M, getfunc, code, *args, **kwargs):
        if M is None:
            if all([var is not None for var in list(args)+list(kwargs.values())]):
                M = getfunc(*args, **kwargs)
            else:
                raise ValueError(f'code {code}: Inconsistent arguments passed to {getfunc.__name__}.')
        return M

    S_cross_ao = _check_existence(S_cross_ao, get_ovlp_cross, 0, mol, pw_k, xlim, ylim, zlim, w)
    T_cross_ao = _check_existence(T_cross_ao, get_kin_cross, 1, mol, pw_k, xlim, ylim, zlim, w)
    V_cross_ao = _check_existence(V_cross_ao, get_vnuc_cross, 2, mol, pw_k, xlim, ylim, zlim, w)
    if efield is not None:
        D_cross_ao = _check_existence(D_cross_ao, get_dipole_cross, 3, mol, pw_k, xlim, ylim, zlim, w)
    if verb >= 1:
        print(f'Time spent for computing 1e integrals = {time.time()-thistime:.3f}')
        thistime = time.time()
    g2e_kbbb = _check_existence(g2e_kbbb, get_v2e, 3, mol, pw_k, xlim, ylim, zlim, w, ['kbbb'])
    if verb >= 1:
        print(f'Time spent for computing 2e integrals = {time.time()-thistime:.3f}')
        thistime = time.time()
    if isinstance(g2e_kbbb, dict):
        g2e_kbbb = g2e_kbbb['kbbb']

    X = d_ortho
    G = get_coupling(cf, S_cross_ao, T_mol_ao, T_cross_ao, V_mol_ao, V_cross_ao, g2e_mmmm,
                     g2e_kbbb, X, efield, D_mol_ao, D_cross_ao)
    #G = np.random.rand(G.shape[0],G.shape[1])-0.5 + 1j*(np.random.rand(G.shape[0],G.shape[1])-0.5)
    #use the line above and the effective PW becomes complex.
    U, sv, Vt = scipy.linalg.svd(G, full_matrices=False)
    B = Vt[sv>sv_thr,:].conj().T

    return \
        sv[sv>sv_thr], \
        np.einsum('ij,jk->ik', d_ortho, B, optimize=True), \
        np.einsum('ij,jk->ik', e_ortho, B, optimize=True)
#============================================================


#============================================================
def get_effective_tpw(mol, cf, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, sv_thr=-1, verb=0):

    Volume = (xlim[1]-xlim[0]+2*w) * (ylim[1]-ylim[0]+2*w) * (zlim[1]-zlim[0]+2*w)
    norm = np.sqrt(Volume)
    nao = mol.nao
    pw_k = assemble_ks(kxvec, kyvec, kzvec)
    nk = len(pw_k)
        
    
    #==== Step 1: Orthogonal projection from MO space ====#
    e_proj = projected_tpw(mol, pw_k, xlim, ylim, zlim, w, cf, verb)


    #==== Step 2: Orthogonalization of the projected vectors ====!
    S_tpw = get_ovlp_tpw(kxvec, kyvec, kzvec, xlim, ylim, zlim, w)
    d_ortho, e_ortho = orthogonalized_tpw(ortho, e_proj, canon_thr, verb, S_tpw=S_tpw)


    #==== Step 3: Perform SVD on the coupling matrix ====#
    if INTS_ALGO == 'xa':
        S_cross_ao = aopw_xa.ao_pw_ovlp(mol, pw_k) / norm
        pw_kin = 0.5 * np.einsum("pi,pi->p", pw_k, pw_k)
        T_cross_ao = pw_kin[None,:] * S_cross_ao
        V_cross_ao = aopw_xa.ao_pw_pot(mol, pw_k) / norm
        g2e_kbbb = aopw_xa.int_2e(mol, pw_k, set(['kbbb']), pw_norm=1/norm)
        sortid = aopw_xa.sort_kbbb(g2e_kbbb, nao)
        g2e_kbbb = [g2e_kbbb[i] for i in sortid]
    elif INTS_ALGO == 'xn':
        with aopw_xn.prepare(mol) as opt:
            S_cross_ao = aopw_xn.ovlp(mol, pw_k, opt=opt).T / norm
            T_cross_ao = aopw_xn.kin(mol, pw_k, opt=opt).T / norm
            V_cross_ao = aopw_xn.nuc(mol, pw_k, opt=opt).T / norm
            bkbb = aopw_xn.gpgg(mol, pw_k, opt=opt)
            
        g2e_kbbb = []
        for k in range(nk):
            for p in range(nao):
                for q in range(nao):
                    for r in range(q+1):
                        g2e_kbbb.append(INT2e((nao+k, p, q, r), bkbb[k,p,q,r].conj()/norm))

    sv, d_eff, e_eff = effective_tpw(d_ortho, e_ortho, cf, mol, sv_thr, verb,
                                     S_cross_ao=S_cross_ao, T_cross_ao=T_cross_ao,
                                     V_cross_ao=V_cross_ao, g2e_kbbb=g2e_kbbb)

    return sv, d_eff, e_eff
#============================================================


#============================================================
def compare_pyscf(mol, cf, pw_k):

    nao, nmo = cf.shape
    
    #==== Check against PySCF ====#
    compare_check = False
    if compare_check:
        #==== Compare (aa|aa) against PySCF ====#
        int2e_tp = set(['bbbb'])
        g2e_aaaa_ref = aopw_xa.int_2e(mol, pw_k, int2e_tp)
        g2e_aaaa = mol.intor('int2e_sph', aosym="s2ij")
        #^^ g2e_aaaa is a 3D array.
        # 's2ij' means the first two indices form independent pairs, nao*(nao+1)/2.
        # The last two indices covers all AOs, each has nao elements.
        print("Shape of (aa|aa) = ", g2e_aaaa.shape)
        print('\nComparing (aa|aa) against PySCF:')
        for i, int2e in enumerate(g2e_aaaa_ref):
            p, q, r, s = int2e.index
            pq = round(p * (1+p) / 2) + q
            print('%8d %4s  %24.16e  %24.16e  %24.16e' %
                  (i, int2e.type, int2e.value.real, g2e_aaaa[pq, r, s], g2e_aaaa[pq, s, r]), int2e.index)
        
        #==== Compare (mm|mm) with PySCF ====#
        g2e_mmmm_ref = aopw_xa.int2e_ao2mo(cf, g2e_aaaa_ref, mol.nao, len(pw_k))
        g2e_mmmm = ao2mo.kernel(mol, cf, intor='int2e_sph', aosym='s2ij')
        g2e_mmmm = g2e_mmmm.reshape((g2e_mmmm.shape[0]*nmo, nmo), order='C')
        print('\nComparing (mm|mm) against PySCF:')
        for i, int2e in enumerate(g2e_mmmm_ref):
            p, q, r, s = int2e.index
            pq = round(p * (1+p) / 2) + q
            pqr = pq*nmo + r
            pqs = pq*nmo + s
            print('%8d %4s  %24.16e  %24.16e  %24.16e' %
                  (i, int2e.type, int2e.value.real, g2e_mmmm[pqr, s], g2e_mmmm[pqs, r]), int2e.index)
            
    return
#============================================================


#============================================================
def analyze(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, cf, e_proj=None, d_ortho=None,
            e_ortho=None, d_eff=None, e_eff=None, sv=None, nk_tpw=None, nk_proj=None,
            nk_ortho=None, nk_eff=None):

    #--------------------------------------------------------
    def _print_orthonormal(ovlp):
        ds = np.sum(np.diag(ovlp))
        ods = np.sum(ovlp) - ds
        print('    Sum of diagonals = (%.10e, %.10ej)' % (ds.real, ds.imag))
        print('    Sum of off-diagonals = (%.10e, %.10ej)' % (ods.real, ods.imag))
        return
    #--------------------------------------------------------

    #--------------------------------------------------------
    def _print_ptable(M_expec, M_sqr, nstates, sv=None):
        with_sv = sv is not None
        if with_sv:
            print('%4s  %14s' % ('No.', 'S. values'), end='')
        else:
            print('%4s' % 'No.', end='')
        print('    %19s %11s     %19s %11s      %19s %11s  '%
              ('Re.<px>', 'Im.<px>', 'Re.<py>', 'Im.<py>', 'Re.<pz>', 'Im.<pz>'),
              '    %13s %11s      %13s %11s      %13s %11s  '%
              ('Re.dpx', 'Im.dpx', 'Re.dpy', 'Im.dpy', 'Re.dpz', 'Im.dpz'),
              flush=True)
        for i in range(nstates):
            sigma = np.sqrt( M_sqr[i,:] - M_expec[i,:]**2 )
            p0 = M_expec[i,:]
            if with_sv:
                print('%4d  %14.6e' % (i, sv[i]), end='')
            else:
                print('%4d' % i, end='')
            print('   (%19.11e %11.3ej)   (%19.11e %11.3ej)   (%19.11e %11.3ej)'%
                  (p0[0].real, p0[0].imag,
                   p0[1].real, p0[1].imag,
                   p0[2].real, p0[2].imag),
                  '   (%13.5e %11.3ej)   (%13.5e %11.3ej)   (%13.5e %11.3ej)'%
                  (sigma[0].real, sigma[0].imag,
                   sigma[1].real, sigma[1].imag,
                   sigma[2].real, sigma[2].imag),
                  flush=True)
        return
    #--------------------------------------------------------

    #--------------------------------------------------------
    def _print_ttable(M_expec, nstates, sv=None):
        with_sv = sv is not None
        if with_sv:
            print('%4s  %14s' % ('No.', 'S. values'), end='')
        else:
            print('%4s' % 'No.', end='')
        print('  %19s  %19s  %19s' % ('Re.<T>', 'Im.<T>', 'Re.<T> (eV)'),
              flush=True)
        for i in range(nstates):
            t0 = M_expec[i]
            if with_sv:
                print('%4d  %14.6e' % (i, sv[i]), end='')
            else:
                print('%4d' % i, end='')
            print('  %19.11e  %19.11e  %19.11e' % (t0.real, t0.imag, t0.real*27.2113860200),
                  flush=True)
        return
    #--------------------------------------------------------



    Volume = (xlim[1]-xlim[0]+2*w) * (ylim[1]-ylim[0]+2*w) * (zlim[1]-zlim[0]+2*w)
    nao, nmo = cf.shape
    nk = len(kxvec) * len(kyvec) * len(kzvec)
    print('AO-PW integral algorithm = ', INTS_ALGO)
    print('No. of GTO basis functions = ', nao, flush=True)
    print('No. of MOs = ', nmo, flush=True)
    print('No. of k points (kx,ky,kz) = ', (len(kxvec), len(kyvec), len(kzvec)), flush=True)
    print('kx bounds = %.5e and %.5e' % (kxvec[0], kxvec[-1]), flush=True)
    print('ky bounds = %.5e and %.5e' % (kyvec[0], kyvec[-1]), flush=True)
    print('kz bounds = %.5e and %.5e' % (kzvec[0], kzvec[-1]), flush=True)
    print('Norm of tPWs = %.10e' % np.sqrt(Volume), flush=True)
    print('x wall locations = %.5f and %.5f' % (xlim[0], xlim[1]), flush=True)
    print('y wall locations = %.5f and %.5f' % (ylim[0], ylim[1]), flush=True)
    print('z wall locations = %.5f and %.5f' % (zlim[0], zlim[1]), flush=True)
    print('Exponential tail 1/e width = %.10e' % w, flush=True)

    
    #==== Get the necessary operators ====#
    S = get_ovlp(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, cf=cf)
    P = get_momentum(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, cf=cf)
    Sx = ovlp_tpw_1d(kxvec, xlim[0], xlim[1], w)
    Sy = ovlp_tpw_1d(kyvec, ylim[0], ylim[1], w)
    Sz = ovlp_tpw_1d(kzvec, zlim[0], zlim[1], w)
    P2 = 2 * get_kin_cart(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx=Sx, Sy=Sy, Sz=Sz,
                          cf=cf)
    T = get_kin(mol, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Sx=Sx, Sy=Sy, Sz=Sz, cf=cf)
    
    
    #==== II Orthonormality checks ====#
    print('\nI.1 Orthonormality checks of tapered PWs:')
    _print_orthonormal(S[nmo:,nmo:])

    if e_proj is not None:
        cf_proj = np.vstack((e_proj, np.eye(nk, dtype=complex)))
        S_proj = np.einsum('ij,ik,kl->jl', cf_proj.conj(), S, cf_proj, optimize=True)
        print('\nI.2 Orthonormality checks of projected tPWs:')
        _print_orthonormal(S_proj)

    if d_ortho is not None and e_ortho is not None:
        cf_ortho = np.vstack((e_ortho, d_ortho))
        S_ortho = np.einsum('ij,ik,kl->jl', cf_ortho.conj(), S, cf_ortho, optimize=True)
        print('\nI.3 Orthonormality checks of orthogonalized PWs:')
        _print_orthonormal(S_ortho)

    if d_eff is not None and e_eff is not None:
        cf_eff = np.vstack((e_eff, d_eff))
        S_eff = np.einsum('ij,ik,kl->jl', cf_eff.conj(), S, cf_eff, optimize=True)
        print('\nI.4 Orthonormality checks of effective PWs:')
        _print_orthonormal(S_eff)

    
    #==== II.1 <P> and sigma(P) of tPWs ====#
    P_expec_tpw = np.diagonal(P, axis1=1, axis2=2).T[nmo:,:]
    P2_expec_tpw = np.diagonal(P2, axis1=1, axis2=2).T[nmo:,:]
    print('\nII.1 Momentum expectation values and std. deviations of tapered PWs:')
    nk_tpw = nk if nk_tpw is None else min(nk,nk_tpw)
    _print_ptable(P_expec_tpw, P2_expec_tpw, nk_tpw)
    
    #==== II.2 <P> and sigma(P) of projected tPWs ====#
    if e_proj is not None:
        proj_norm = np.sqrt( np.diag(S_proj).copy().real )
        P_expec_proj = np.einsum('ij,dik,kj,j->jd', cf_proj.conj(), P, cf_proj, 1/proj_norm**2,
                                 optimize=True)
        P2_expec_proj = np.einsum('ij,dik,kj,j->jd', cf_proj.conj(), P2, cf_proj, 1/proj_norm**2,
                                  optimize=True)
        nk_proj = cf_proj.shape[1] if nk_proj is None else min(cf_proj.shape[1],nk_proj)
        print('\nII.2 Momentum expectation values and std. deviations of projected tPWs:')
        _print_ptable(P_expec_proj, P2_expec_proj, nk_proj)
    
    #==== II.3 <P> and sigma(P) of orthogonalized tPWs ====#
    if d_ortho is not None and e_ortho is not None:
        P_expec_ortho = np.einsum('ij,dik,kj->jd', cf_ortho.conj(), P, cf_ortho, optimize=True)
        P2_expec_ortho = np.einsum('ij,dik,kj->jd', cf_ortho.conj(), P2, cf_ortho, optimize=True)
        nk_ortho = cf_ortho.shape[1] if nk_ortho is None else min(cf_ortho.shape[1],nk_ortho)
        print('\nII.3 Momentum expectation values and std. deviations of orthogonalized tPWs:')
        _print_ptable(P_expec_ortho, P2_expec_ortho, nk_ortho)

    #==== II.4 <P> and sigma(P) of effective tPWs ====#
    if d_eff is not None and e_eff is not None:
        P_expec_eff = np.einsum('ij,dik,kj->jd', cf_eff.conj(), P, cf_eff, optimize=True)
        P2_expec_eff = np.einsum('ij,dik,kj->jd', cf_eff.conj(), P2, cf_eff, optimize=True)
        nk_eff = cf_eff.shape[1] if nk_eff is None else min(cf_eff.shape[1],nk_eff)
        print('\nII.4 Momentum expectation values and std. deviations of effective tPWs:')
        _print_ptable(P_expec_eff, P2_expec_eff, nk_eff, sv=sv)

    
    #==== III.1 <T> of tPWs ====#
    T_expec_tpw = np.diag(T)[nmo:]
    print('\nIII.1 Kinetic energy expectation values of tapered PWs:')
    _print_ttable(T_expec_tpw, nk_tpw)
    
    #==== III.2 <T> of projected tPWs ====#
    if e_proj is not None:
        T_expec_proj = np.einsum('ij,ik,kj->j', cf_proj.conj(), T, cf_proj, optimize=True) / \
            proj_norm**2
        print('\nIII.2 Kinetic energy expectation values of projected tPWs:')
        _print_ttable(T_expec_proj, nk_proj)
    
    #==== III.3 <T> of orthogonalized tPWs ====#
    if d_ortho is not None and e_ortho is not None:
        T_expec_ortho = np.einsum('ij,ik,kj->j', cf_ortho.conj(), T, cf_ortho, optimize=True)
        print('\nIII.3 Kinetic energy expectation values of orthogonalized tPWs:')
        _print_ttable(T_expec_ortho, nk_ortho)

    #==== III.4 <T> of effective tPWs ====#
    if d_eff is not None and e_eff is not None:
        T_expec_eff = np.einsum('ij,ik,kj->j', cf_eff.conj(), T, cf_eff, optimize=True)
        print('\nIII.4 Kinetic energy expectation values of effective tPWs:')
        _print_ttable(T_expec_eff, nk_eff, sv=sv)    

    return
#============================================================


#====================================================#
def eval_tpw_1d(x, k, xlim, w):

    N = xlim[1] - xlim[0] + 2*w
    if x < xlim[0]:
        A = np.exp((x-xlim[0])/2/w)
    elif x >= xlim[0] and x <= xlim[1]:
        A = 1.0
    else:
        A = np.exp(-(x-xlim[1])/2/w)
        
    return A * np.exp(1j*k*x) / np.sqrt(N)
#====================================================#


#====================================================#
def eval_tpw_3d(pos, kvec, xlim, ylim, zlim, w):

    pos = np.atleast_2d(pos)
    kvec = np.atleast_2d(kvec)
    bfun = np.zeros((pos.shape[0], kvec.shape[0]), dtype=complex)
    for i, pos0 in enumerate(pos):
        for j, kvec0 in enumerate(kvec):
            fx = eval_tpw_1d(pos0[0], kvec0[0], xlim, w)
            fy = eval_tpw_1d(pos0[1], kvec0[1], ylim, w)
            fz = eval_tpw_1d(pos0[2], kvec0[2], zlim, w)
            bfun[i,j] = fx * fy * fz
            
    return bfun
#====================================================#


#====================================================#
def print_line(fname, kp_coeff, mol, mo_coeff, pw_k, xlim, ylim, zlim, w, eval_ax,
               line_grids, x1, x2):

    ax = np.array(['x', 'y', 'z'])
    assert eval_ax in ax, f'The chosen evaluation axis: {eval_ax} must be a Cartesian axis.'
    ortho = ax[ax!=eval_ax]    # ortho is the plane perpendicular to the print axis.
    bfun = np.zeros(mo_coeff.shape[1]+pw_k.shape[0], dtype=complex)

    with open(fname, 'w') as fn:
        fn.write(f'{"#":1s} x walls = [{xlim[0]:.3f}, {xlim[1]:.3f}], ' +
                 f'y walls = [{ylim[0]:.3f}, {ylim[1]:.3f}], z walls = [{zlim[0]:.3f}, {zlim[1]:.3f}]\n')
        fn.write(f'{"#":1s} Tail width = {w:.3f}\n')
        fn.write(f'{"#":1s} kx points = [{kxlim[0]:.3f}, {kxlim[1]:.3f}, {Nkx:d}]\n')
        fn.write(f'{"#":1s} ky points = [{kylim[0]:.3f}, {kylim[1]:.3f}, {Nky:d}]\n')
        fn.write(f'{"#":1s} kz points = [{kzlim[0]:.3f}, {kzlim[1]:.3f}, {Nkz:d}]\n')
        fn.write(f'{"#":1s} Print axis = {eval_ax} at {ortho[0]} = {x1:.3f} and {ortho[1]} = {x2:.3f}\n')
        #fn.write(f'{"#":1s}  {"No.":6s}  {eval_ax:14s}  {"Real part":14s}  {"Imag part":14s}\n')
        fn.write('#  %6s  %14s  %15s  %15s  %15s  %15s\n'%
                 ('No', eval_ax, 'Real MO part', 'Imag MO part', 'Real k part', 'Imag k part'))

        for i, grid in enumerate(line_grids):
            
            pos = np.zeros(3)
            pos[ax==eval_ax] = grid
            pos[ax!=eval_ax] = np.array([x1, x2])
    
            ao = mol.eval_gto('GTOval_sph', np.atleast_2d(pos))
            bfun[:nmo] = ao.dot(mo_coeff)
            wfn_mo = np.dot(kp_coeff[:nmo], bfun[:nmo])
            bfun[nmo:] = eval_tpw_3d(pos, pw_k, xlim, ylim, zlim, w)
            wfn_k = np.dot(kp_coeff[nmo:], bfun[nmo:])

            fn.write(f'{" ":1s}  {i:6d}  {grid:14.6e}  {wfn_mo.real:15.6e}  {wfn_mo.imag:15.6e}' +
                     f'  {wfn_k.real:15.6e}  {wfn_k.imag:15.6e}\n')

    return
#====================================================#


##====================================================#
def print_plane(fname, kp_coeff, mol, mo_coeff, pw_k, xlim, ylim, zlim, w, eval_plane,
                line_grids_x0, line_grids_x1, x2):
    '''
    eval_plane :
      Evaluation plane, e.g., xy, yz, zx, etc.
    line_grids_x0 :
      The first axis grid points on the evaluation plane. These are taken to be the grids along the
      first axis specified in eval_plane. For example, if eval_plane=zx, then line_grids_x0 contains
      grids along the z axis.
    line_grids_x1 :
      The second axis grid points on the evaluation plane. See the description of line_grids_x0 above
      for an explanation of which axis is assigned to this grid.
    x2 :
      The coordinate of the evaluation plane on the Cartesian axis orthogonal to it, e.g., if
      eval_plane=xz, then x2 is the y coordinate of the evaluation plane.
    '''

    ax = np.array(['x', 'y', 'z'])
    eval_plane_l = list(eval_plane)
    assert all([ev in ax for ev in list(eval_plane_l)]), f'The chosen evaluation plane: {eval_plane} ' + \
        'must be a Cartesian plane, e.g., xy, yz, zy, etc.'
    ortho = (set(ax) - set(list(eval_plane_l))).pop()
    #^^ ortho is the axis perpendicular to the evaluation plane.
    axis_label = np.array(eval_plane_l + [ortho])
    sortid = np.argsort(axis_label)

    bfun = np.zeros(mo_coeff.shape[1]+pw_k.shape[0], dtype=complex)

    with open(fname, 'w') as fn:
        fn.write(f'{"#":1s} x walls = [{xlim[0]:.3f}, {xlim[1]:.3f}], ' +
                 f'y walls = [{ylim[0]:.3f}, {ylim[1]:.3f}], z walls = [{zlim[0]:.3f}, {zlim[1]:.3f}]\n')
        fn.write(f'{"#":1s} Tail width = {w:.3f}\n')
        fn.write(f'{"#":1s} kx points = [{kxlim[0]:.3f}, {kxlim[1]:.3f}, {Nkx:d}]\n')
        fn.write(f'{"#":1s} ky points = [{kylim[0]:.3f}, {kylim[1]:.3f}, {Nky:d}]\n')
        fn.write(f'{"#":1s} kz points = [{kzlim[0]:.3f}, {kzlim[1]:.3f}, {Nkz:d}]\n')
        fn.write(f'{"#":1s} Print plane = {eval_plane} at {ortho} = {x2:.3f}\n')
        #fn.write(f'{"#":1s}  {"No.":6s}  {eval_ax:14s}  {"Real part":14s}  {"Imag part":14s}\n')
        fn.write('#  %6s  %14s  %14s  %15s  %15s  %15s  %15s\n'%
                 ('No', axis_label[0], axis_label[1], 'Real MO part', 'Imag MO part', 'Real k part',
                  'Imag k part'))

        for i, grid0 in enumerate(line_grids_x0):
            for j, grid1 in enumerate(line_grids_x1):                
                pos = np.array([grid0, grid1, x2])
                pos = pos[sortid]
        
                ao = mol.eval_gto('GTOval_sph', np.atleast_2d(pos))
                bfun[:nmo] = ao.dot(mo_coeff)
                wfn_mo = np.dot(kp_coeff[:nmo], bfun[:nmo])
                bfun[nmo:] = eval_tpw_3d(pos, pw_k, xlim, ylim, zlim, w)
                wfn_k = np.dot(kp_coeff[nmo:], bfun[nmo:])

                fn.write(f'{" ":1s}  {i:6d}  {grid0:14.6e}  {grid1:14.6e}  {wfn_mo.real:15.6e}' +
                         f'  {wfn_mo.imag:15.6e}  {wfn_k.real:15.6e}  {wfn_k.imag:15.6e}\n')

            fn.write('\n')

    return
##====================================================#
