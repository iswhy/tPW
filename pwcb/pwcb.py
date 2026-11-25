import math, itertools, time
from functools import partial
import numpy as np
from scipy.integrate import quad, tanhsinh
import tPW.approx.approx as approx
import tPW.exact.exact as exact


#===============================================================
def sum_ttt(nx, tx0, ny, ty0, nz, tz0, K):
    cf = 0.0
    for mx in range(nx+1):
        for my in range(ny+1):
            for mz in range(nz+1):

                if mx+my+mz == K:
                    cf += tx0[mx] * ty0[my] * tz0[mz]
    return cf
#===============================================================


#===============================================================
def count_nnn(N_max):
    n3 = round(N_max**3/36 + 7*N_max**2/24 + 11*N_max/12 + 119/144 +
               (-1)**N_max/16 + 2/9/np.sqrt(3)*math.sin(np.pi/3*(2*N_max+1)))
    return n3
#===============================================================


#===============================================================
def get_sum_ttt(tx, ty, tz):
    '''
    tx, ty, tz:
      Power series coefficients of all Chebyshev polynomials used in the expansion.
      The rows of tx, ty, and tz identify the Chebyshev polynomials, while the columns
      of a given row identifies the powers of the argument of the corresponding
      Chebyshev polynomial. tx, ty, tz are square matrices.
    '''
    
    assert tx.shape == ty.shape and ty.shape == tz.shape
    N_max = tx.shape[0] - 1
    n3 = count_nnn(N_max)
    cf_3t = np.zeros((N_max+1, n3))
    for K in range(N_max+1):
        i = 0
        for nx in range(N_max+1):
            for ny in range(N_max+1):
                for nz in range(N_max+1):
    
                    if nx+ny+nz <= N_max and nx <= ny and ny <= nz:
                        cf_3t[K,i] = sum_ttt(nx, tx[nx,:], ny, ty[ny,:], nz, tz[nz,:], K)
                        i += 1
    return cf_3t
#===============================================================


#===============================================================
def get_ccc(ccx, ccy, ccz):
    assert len(ccx) == len(ccy) and len(ccy) == len(ccz)
    N_max = len(ccx) - 1
    n3 = count_nnn(N_max)
    cf_3c = np.zeros(n3, dtype=complex)
    i = 0
    for nx in range(N_max+1):
        for ny in range(N_max+1):
            for nz in range(N_max+1):
    
                if nx+ny+nz <= N_max and nx <= ny and ny <= nz:
                    sum_list = set(itertools.permutations((nx,ny,nz)))
                    #test sum_list = [(nx,ny,nz)]
                    for n_trio in sum_list:
                        nx0, ny0, nz0 = n_trio
                        cf_3c[i] += ccx[nx0] * ccy[ny0] * ccz[nz0]

                    i += 1

    return cf_3c
#===============================================================


#===============================================================
def get_coef_b(ccx, ccy, ccz, tx, ty, tz):
    '''
    Get the b coefficients for a particular set of cc (Chebyshev) coefficients.
    '''
    cf_3c = get_ccc(ccx, ccy, ccz)
    cf_3t = get_sum_ttt(tx, ty, tz)

    return cf_3t @ cf_3c
#===============================================================


#===============================================================
def get_coeff_LR_cb(c, q, w, xL, xR, N, lbound, ubound):
    '''
    Returns:
    cc:
      An array with N+1 columns containing Chebyshev coefficients for the small-t part of 
      E_i(t). It has three rows if q equals zero, otherwise, it has two rows. The first 
      and second rows correspond to the left and right tail regions. If q equals zero, the
      third row corresponds to the small-t part of the middle region.
    '''
    
    zero_q = abs(q) < 1E-12
    cc = []
    cc += [approx.get_coeff_GL_cb(c, q, w, xL, N, lbound, ubound, not zero_q)]
    cc += [approx.get_coeff_GR_cb(c, q, w, xR, N, lbound, ubound, not zero_q)]
    if zero_q:
        cc += [-approx.get_coeff_FLR_q0_cb(c, xL, xR, N, lbound, ubound)]
    return np.array(cc)
#===============================================================


#===============================================================
def get_F_approx_pl(c, q, w, wallL, wallR, N, tsmall, tlarge, nu=4, nfactor=1.0):

    xwalls = (wallL[0], wallR[0])
    ywalls = (wallL[1], wallR[1])
    zwalls = (wallL[2], wallR[2]) 


    widths = \
        (1/abs(c[0]-xwalls[0]), 1/abs(c[0]-xwalls[1]),
         1/abs(c[1]-ywalls[0]), 1/abs(c[1]-ywalls[1]),
         1/abs(c[2]-zwalls[0]), 1/abs(c[2]-zwalls[1]))
    lbound = tsmall[0]
    ubound0 = nu * max(widths)
    if ubound0 < tsmall[-1]:
        idu = (np.abs(tsmall - ubound0)).argmin()
        ubound = tsmall[idu]
    else:
        idu = len(tsmall) - 1
        ubound = tsmall[-1]
    #^^ The true upper bound should be an element of tsmall.
    

    
    c_eye = np.eye(N+1)
    tx = np.zeros((N+1,N+1))
    for i in range(N+1):
        tx[i,:i+1] = np.polynomial.chebyshev.cheb2poly(c_eye[i,:])
    ty = tx
    tz = tx
    cf_3t = get_sum_ttt(tx, ty, tz)
    
    ccx = get_coeff_LR_cb(c[0], q[0], w, wallL[0], wallR[0], N, lbound, ubound)
    ccy = get_coeff_LR_cb(c[1], q[1], w, wallL[1], wallR[1], N, lbound, ubound)
    ccz = get_coeff_LR_cb(c[2], q[2], w, wallL[2], wallR[2], N, lbound, ubound)

    F_small = np.zeros(len(tsmall), dtype=complex)
    zz = (2 * np.asarray(tsmall[:idu+1]) - lbound - ubound) / (ubound - lbound)
    zz_powers = np.array( [zz**K for K in range(N+1)] ).T      # zz.shape = (len(zz),N+1)

    scalex = [(c[0] - xwalls[i])**2 for i in range(2)]
    if abs(q[0]) < 1E-12:
        scalex += [0.0]
    scaley = [(c[1] - ywalls[i])**2 for i in range(2)]
    if abs(q[1]) < 1E-12:
        scaley += [0.0]
    scalez = [(c[2] - zwalls[i])**2 for i in range(2)]
    if abs(q[2]) < 1E-12:
        scalez += [0.0]
    
    for i in range(len(ccx)):
        for j in range(len(ccy)):
            for k in range(len(ccz)):
            
                cf_3c = get_ccc(ccx[i,:], ccy[j,:], ccz[k,:])
                b = cf_3t @ cf_3c
                exp_scale = scalex[i] + scaley[j] + scalez[k]
                exp_factor = np.exp(-tsmall[:idu+1]**2 * exp_scale)
                F_small[:idu+1] += exp_factor * (zz_powers @ b)

    F_small = F_small / nfactor


    
    F_large = np.sqrt(np.pi**3) / tlarge**3 * \
        np.exp(1j*np.dot(q,c) - np.linalg.norm(q)**2/4/tlarge**2)
    F_large = F_large / nfactor

    return F_small, F_large
#===============================================================


#===============================================================
def vnuc_elm_tsmall(Jn, qx0, ccx, qy0, ccy, qz0, ccz, cf_3t):

    '''
    The zeroness of q_i is taken care of in the number of rows of the cci. If qx0 
    is True (qx=0), the first three rows of ccx are referenced, otherwise, only 
    the first two rows are referenced. The same rule applies to qy0 and qz0.
    '''
    
    nccx = 3 if qx0 else 2
    nccy = 3 if qy0 else 2
    nccz = 3 if qz0 else 2
    
    vnuc = complex(0.0, 0.0)
    for i in range(nccx):
        for j in range(nccy):
            for k in range(nccz):
            
                cf_3c = get_ccc(ccx[i,:], ccy[j,:], ccz[k,:])
                b = cf_3t @ cf_3c
                vnuc += np.dot(b, Jn[i,j,k,:])
    
    return vnuc
#===============================================================


#===============================================================
def vnuc_elm_q0(c, xlim, ylim, zlim, w, nu=6, atol=1E-10, rtol=1E-10):

    widths = \
        (1/abs(c[0]-xlim[0]), 1/abs(c[0]-xlim[1]),
         1/abs(c[1]-ylim[0]), 1/abs(c[1]-ylim[1]),
         1/abs(c[2]-zlim[0]), 1/abs(c[2]-zlim[1]))
    ubound = nu * max(widths)

    _calc_F_q0 = partial(exact.calc_F,
                         c=c, xlim=xlim, ylim=ylim, zlim=zlim, w=w, q=np.zeros(3))
    elm, error_est = quad(lambda x: np.real(_calc_F_q0(x)), 0.0, ubound, epsabs=atol,
                                  epsrel=rtol)
    elm += np.sqrt(np.pi**3) * 1/(2*ubound**2)
    
    return elm
#===============================================================


#===============================================================
def get_gauss_moment(a, b, c, N):

    J = np.zeros(N+1)

    J[0] = np.sqrt(np.pi) / 2 / c
    if N == 0:
        return J
    
    J[1] = a / 2 / c**2 - b * J[0]
    if N == 1:
        return J

    for n in range(2,N+1):
        J[n] = a*(-b)**(n-1)/2/c**2 - b*J[n-1] + a**2*(n-1)/2/c**2*J[n-2]

    return J
#===============================================================


#===============================================================
def trapezoidal_integral(f, a, b, n=1000):

    """
    Approximates the definite integral of f(x) from a to b using the trapezoidal rule.
    
    Parameters:
        f (callable): The function to integrate.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        n (int): Number of subintervals (default: 1000).
        
    Returns:
        float: The approximated integral.
    """
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        total += f(a + i * h)
        
    return total * h
#===============================================================


#===============================================================
def calc_vnuc_circulant(Zn, c, nk, qint, nM, w, xlim, ylim, zlim, N, lbound=0.0, nu=4, nu0=6, 
                        nfactor=1.0, rtol0=1e-10, atol0=1e-10, larget=True, verb=0):

    '''
    nk:
      A tuple of three elements, the number of k points in each Cartesian direction.
    qint:
      A tuple of three elements, the interval of k points in each Cartesian direction.
    nM:
      A tuple of three elements, the dimensions of the circulant matrix.
    larget:
      If True (default), the matrix elements include a contribution from the large-t part of F.
      Otherwise, this contribution is set to zero. The case of larget=False is mainly used for
      debugging.
    '''

    #==== Determine the upperbound for the Chebyshev expansion ====#
    widths = \
        (1/abs(c[0]-xlim[0]), 1/abs(c[0]-xlim[1]),
         1/abs(c[1]-ylim[0]), 1/abs(c[1]-ylim[1]),
         1/abs(c[2]-zlim[0]), 1/abs(c[2]-zlim[1]))
    ubound = nu * max(widths)

    
    scalex = ( (c[0] - xlim[0])**2, (c[0] - xlim[1])**2, 0.0 )
    scaley = ( (c[1] - ylim[0])**2, (c[1] - ylim[1])**2, 0.0 )
    scalez = ( (c[2] - zlim[0])**2, (c[2] - zlim[1])**2, 0.0 )

    #==== Build Gaussian moment tensor ====#
    Jn = np.zeros((3,3,3,N+1))
    for i in range(3):
        for j in range(3):
            for k in range(3):

                exp_scale = scalex[i] + scaley[j] + scalez[k]
                if any([jj != 2 for jj in (i,j,k)]):
                    Jn[i,j,k,:] = get_gauss_moment(2/(ubound-lbound), (ubound+lbound)/(ubound-lbound),
                                                   np.sqrt(exp_scale), N)
                

    c_eye = np.eye(N+1)
    tx = np.zeros((N+1,N+1))
    for i in range(N+1):
        tx[i,:i+1] = np.polynomial.chebyshev.cheb2poly(c_eye[i,:])
    ty = tx
    tz = tx
    cf_3t = get_sum_ttt(tx, ty, tz)

    
    #---------------------------------
    def _get_nrow(ii):
        return 3 if ii==0 else 2
    #---------------------------------

    
    # The the first index of the cc arrays (ccx, ccy, and ccz) below runs from the most negative
    # q value through the most positive q value. If the dependence of cc on q=k2-k1 is viewed as
    # a matrix whose row and column correspond to k1 and k2, respectively, cc contains the first
    # row starting from the right-most element to the left-most element, then down along the first
    # column to the bottom-most element. So, the length of cc is 2*nk-1. For any Cartesian
    # component below, the following indexing applies:
    # ---------------------------------------
    # i (loop)   cc row number        q value
    # ---------------------------------------
    #  -(nk-1)               0   -(nk-1)*qint
    #      ...             ...            ...
    #       -1          nk - 2          -qint
    #        0          nk - 1              0
    #                   nk + 0          +qint
    #                      ...            ...
    #                 2*nk - 1   +(nk-1)*qint

    #==== ccx array ====#
    ccx = np.zeros((2*nk[0]-1, 3, N+1), dtype=complex)
    for i in range(-(nk[0]-1), 1):
        ccx[i+(nk[0]-1) , :_get_nrow(i) , :] = get_coeff_LR_cb(c[0], qint[0]*i, w, xlim[0], xlim[1],
                                                               N, lbound, ubound)
    ccx[nk[0]: , :2 , :] = np.conj(ccx[nk[0]-2::-1 , :2 , :])
    ccx_q = np.linspace(-(nk[0]-1)*qint[0], +(nk[0]-1)*qint[0], 2*nk[0]-1)

    #==== ccy array ====#
    ccy = np.zeros((2*nk[1]-1, 3, N+1), dtype=complex)
    for i in range(-(nk[1]-1), 1):
        ccy[i+(nk[1]-1) , :_get_nrow(i) , :] = get_coeff_LR_cb(c[1], qint[1]*i, w, ylim[0], ylim[1],
                                                               N, lbound, ubound)
    ccy[nk[1]: , :2 , :] = np.conj(ccy[nk[1]-2::-1 , :2 , :])
    ccy_q = np.linspace(-(nk[1]-1)*qint[1], +(nk[1]-1)*qint[1], 2*nk[1]-1)

    #==== ccz array ====#
    ccz = np.zeros((2*nk[2]-1, 3, N+1), dtype=complex)
    for i in range(-(nk[2]-1), 1):
        ccz[i+(nk[2]-1) , :_get_nrow(i) , :] = get_coeff_LR_cb(c[2], qint[2]*i, w, zlim[0], zlim[1],
                                                               N, lbound, ubound)
    ccz[nk[2]: , :2 , :] = np.conj(ccz[nk[2]-2::-1 , :2 , :])
    ccz_q = np.linspace(-(nk[2]-1)*qint[2], +(nk[2]-1)*qint[2], 2*nk[2]-1)


    #-----------------------------------------
    def _get_cc_id(i_, nk_, nM_):
        '''
        Maps an index of the circulant matrix (0 <= i_ <= nM_-1) to the corresponding index
        of the cc matrices above (iq_).
        '''
        assert i_ >= 0 and i_ <= nM_-1
        if i_ <= nk_-1:
            iq_ = i_ + nk_ - 1
        elif i_ >= nM_ - nk_ + 1:
            iq_ = i_ - (nM_ - nk_ + 1)
        else:
            iq_ = None          # Padding elements
        return iq_
    #-----------------------------------------


    
    #==== Build the 3D circulant tensor ====#
    prefactor = - Zn * 2 / np.sqrt(np.pi) / nfactor
    if verb >= 1:
        time0 = time.time()
    Cr = np.zeros(nM, dtype=complex)
    for i in range(nM[0]):
        iq = _get_cc_id(i, nk[0], nM[0])
        zero_qx = abs(ccx_q[iq]) < 1E-12
        for j in range(nM[1]):
            jq = _get_cc_id(j, nk[1], nM[1])
            zero_qy = abs(ccy_q[jq]) < 1E-12
            for k in range(nM[2]):
                kq = _get_cc_id(k, nk[2], nM[2])
                zero_qz = abs(ccz_q[kq]) < 1E-12

                if all([idq is not None for idq in (iq,jq,kq)]):
                    q = np.array([ccx_q[iq], ccy_q[jq], ccz_q[kq]])
                    if any([not zero_qx, not zero_qy, not zero_qz]):
                        smallt_part = vnuc_elm_tsmall(Jn, zero_qx, ccx[iq,:,:], zero_qy, 
                                                      ccy[jq,:,:], zero_qz, ccz[kq,:,:], cf_3t)
                        if larget:
                            larget_part = 2 * np.sqrt(np.pi**3) * np.exp(1j*np.dot(q,c)) / np.linalg.norm(q)**2
                        else:
                            larget_part = complex(0.0, 0.0)
                    else:
                        smallt_part = complex(0.0, 0.0)
                        larget_part = vnuc_elm_q0(c, xlim, ylim, zlim, w, nu=nu0, atol=atol0,
                                                  rtol=rtol0)
                    Cr[i,j,k] = prefactor * (larget_part + smallt_part)
                    print(f'ijk: {i}  {j}  {k} | {q[0]:.4f}  {q[1]:.4f}  {q[2]:.4f} | ' +
                          f'{prefactor*larget_part:.6e}  {prefactor*smallt_part:.6e}', flush=True)
                        

    if verb >= 1:
        print('Chebyshev elapsed time = %.3f s' % (time.time() - time0), flush=True)
    return Cr
#===============================================================

    
#===============================================================
def calc_vnuc_circulant_ts(Zn, c, nk, qint, nM, w, xlim, ylim, zlim, nfactor=1.0, nu0=6, rtol=1e-13,
                           atol=1e-13, rtol0=1e-10, atol0=1e-10, larget=True, verb=0):

    '''
    larget:
      If True (default), the matrix elements include a contribution from the large-t part of F.
      Otherwise, this contribution is set to zero. The case of larget=False is mainly used for
      debugging.
    '''
    
    #---------------------------------
    def _get_nrow(ii):
        return 3 if ii==0 else 2
    #---------------------------------

    
    # The the first index of the cc arrays (ccx, ccy, and ccz) below runs from the most negative
    # q value through the most positive q value. If the dependence of cc on q=k2-k1 is viewed as
    # a matrix whose row and column correspond to k1 and k2, respectively, cc contains the first
    # row starting from the right-most element to the left-most element, then down along the first
    # column to the bottom-most element. So, the length of cc is 2*nk-1. For any Cartesian
    # component below, the following indexing applies:
    # ---------------------------------------
    # i (loop)   cc row number        q value
    # ---------------------------------------
    #  -(nk-1)               0   -(nk-1)*qint
    #      ...             ...            ...
    #       -1          nk - 2          -qint
    #        0          nk - 1              0
    #                   nk + 0          +qint
    #                      ...            ...
    #                 2*nk - 1   +(nk-1)*qint

    ccx_q = np.linspace(-(nk[0]-1)*qint[0], +(nk[0]-1)*qint[0], 2*nk[0]-1)
    ccy_q = np.linspace(-(nk[1]-1)*qint[1], +(nk[1]-1)*qint[1], 2*nk[1]-1)
    ccz_q = np.linspace(-(nk[2]-1)*qint[2], +(nk[2]-1)*qint[2], 2*nk[2]-1)


    #-----------------------------------------
    def _get_cc_id(i_, nk_, nM_):
        '''
        Maps an index of the circulant matrix (0 <= i_ <= nM_) to the corresponding index
        of the cc matrices above (iq_).
        '''
        assert i_ >= 0 and i_ <= nM_-1
        if i_ <= nk_-1:
            iq_ = i_ + nk_ - 1
        elif i_ >= nM_ - nk_ + 1:
            iq_ = i_ - (nM_ - nk_ + 1)
        else:
            iq_ = None          # Padding elements
        return iq_
    #-----------------------------------------

    
    #==== Build the 3D circulant tensor ====#
    prefactor = - Zn * 2 / np.sqrt(np.pi) / nfactor
    if verb >= 1:
        time0 = time.time()
    Cr = np.zeros(nM, dtype=complex)
    for i in range(nM[0]):
        iq = _get_cc_id(i, nk[0], nM[0])
        zero_qx = abs(ccx_q[iq]) < 1E-12
        for j in range(nM[1]):
            jq = _get_cc_id(j, nk[1], nM[1])
            zero_qy = abs(ccy_q[jq]) < 1E-12
            for k in range(nM[2]):
                kq = _get_cc_id(k, nk[2], nM[2])
                zero_qz = abs(ccz_q[kq]) < 1E-12

                if all([idq is not None for idq in (iq,jq,kq)]):
                    q = np.array([ccx_q[iq], ccy_q[jq], ccz_q[kq]])
                    if any([not zero_qx, not zero_qy, not zero_qz]):
                        _calc_F_tsmall = partial(exact.calc_F,
                                                 c=c, xlim=xlim, ylim=ylim, zlim=zlim, w=w, q=q,
                                                 smallt=True)
                        smallt_part = tanhsinh(_calc_F_tsmall, 0.0, np.inf, rtol=rtol,
                                               atol=atol).integral

                        #rpart, _ = quad(lambda x: np.real(_calc_F_tsmall(x)), 0.0, np.inf, epsabs=atol,
                        #                epsrel=rtol)
                        #ipart, _ = quad(lambda x: np.imag(_calc_F_tsmall(x)), 0.0, np.inf, epsabs=atol,
                        #                epsrel=rtol)
                        #smallt_part = complex(rpart, ipart)
                        
                        #smallt_part = trapezoidal_integral(_calc_F_tsmall, 0.0, 0.007, 1000)
                        if larget:
                            larget_part = 2 * np.sqrt(np.pi**3) * np.exp(1j*np.dot(q,c)) / np.linalg.norm(q)**2
                        else:
                            larget_part = complex(0.0, 0.0)
                    else:
                        smallt_part = complex(0.0, 0.0)
                        larget_part = vnuc_elm_q0(c, xlim, ylim, zlim, w, nu=nu0, atol=atol0,
                                                  rtol=rtol0)
                    Cr[i,j,k] = prefactor * (larget_part + smallt_part)
                    print(f'ref ijk: {i}  {j}  {k} | {q[0]:.4f}  {q[1]:.4f}  {q[2]:.4f} |' +
                          f' {prefactor*larget_part:.6e}  {prefactor*smallt_part:.6e}', flush=True)
                        
                    

    if verb >= 1:
        print('Tanhsinh elapsed time = %.3f s' % (time.time() - time0), flush=True)
        
    return Cr
#=====================================================================


#=====================================================================
def matvec_3d_toeplitz(FCr, x, N_dims):

    """
    Computes T @ x for a 3-level block-Toeplitz matrix using 3D FFT.

    FCr:
      Fcr = np.fft.fftn(Cr)
    """
    
    N1, N2, N3 = N_dims
    M1, M2, M3 = FCr.shape
  
    X = np.asarray(x, dtype=np.complex128).reshape((N1, N2, N3))
    X_pad = np.zeros((M1, M2, M3), dtype=np.complex128)
    X_pad[:N1, :N2, :N3] = X
  
    #Y_pad = np.fft.ifftn(np.fft.fftn(c) * np.fft.fftn(X_pad))
    Y_pad = np.fft.ifftn(FCr * np.fft.fftn(X_pad))

    return Y_pad[:N1, :N2, :N3].reshape(-1)
#=====================================================================
