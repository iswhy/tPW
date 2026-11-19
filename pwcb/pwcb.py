import math, itertools
import numpy as np
import test_vnuc_tpw.approx.approx as approx


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
def get_F_approx_pl(c, q, w, wallL, wallR, N, tsmall, tlarge, nu=4):

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
    
    
    def _get_coeff_LR(c_cart, q_cart, xL, xR):
        '''
        Returns:
        cc:
          An array with N+1 columns containing Chebyshev coefficients for the small-t part of 
          E_i(t). It has three rows if q equals zero, otherwise, it has two rows. The first 
          and second rows correspond to the left and right tail regions. If q equals zero, the
          third row corresponds to the small-t part of the middle region.
        '''
        #cc = np.zeros((2,N+1), dtype=complex)
        zero_q = abs(q_cart) < 1E-12
        cc = []
        cc += [approx.get_coeff_GL_cb(c_cart, q_cart, w, xL, N, lbound, ubound, not zero_q)]
        cc += [approx.get_coeff_GR_cb(c_cart, q_cart, w, xR, N, lbound, ubound, not zero_q)]
        if zero_q:
            cc += [-approx.get_coeff_FLR_q0_cb(c_cart, xL, xR, N, lbound, ubound)]
        return np.array(cc)


    norm_factor = \
        (wallR[0] - wallL[0] + 2*w) * \
        (wallR[1] - wallL[1] + 2*w) * \
        (wallR[2] - wallL[2] + 2*w)

    c_eye = np.eye(N+1)
    tx = np.zeros((N+1,N+1))
    for i in range(N+1):
        tx[i,:i+1] = np.polynomial.chebyshev.cheb2poly(c_eye[i,:])
    ty = tx
    tz = tx
    cf_3t = get_sum_ttt(tx, ty, tz)
    
    # x left and right
    ccx = _get_coeff_LR(c[0], q[0], wallL[0], wallR[0])
    ccy = _get_coeff_LR(c[1], q[1], wallL[1], wallR[1])
    ccz = _get_coeff_LR(c[2], q[2], wallL[2], wallR[2])

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
                #exp_scale = (c[0]-xwalls[i])**2 + (c[1]-ywalls[j])**2 + (c[2]-zwalls[k])**2
                exp_scale = scalex[i] + scaley[j] + scalez[k]
                exp_factor = np.exp(-tsmall[:idu+1]**2 * exp_scale)
                F_small[:idu+1] += exp_factor * (zz_powers @ b)

    F_small = F_small / norm_factor


    
    F_large = np.sqrt(np.pi**3) / tlarge**3 * \
        np.exp(1j*np.dot(q,c) - np.linalg.norm(q)**2/4/tlarge**2)
    F_large = F_large / norm_factor

    return F_small, F_large
#===============================================================

