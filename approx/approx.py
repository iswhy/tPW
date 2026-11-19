import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial, erf, erfcx


#===========================================================
def get_d(N):
    '''
    N:
      N=max_power/2 where max_power is the largest power desired.
      Note that N is not the number of terms in the sum, which would be
      N+1.
    '''

    d = []
    for K in range(N+1):
        for l in range(K+1):
            d0 = (-2)**K / 2**l * factorial(K+l) / factorial(l) / factorial(K-l)
            d += [d0]

    return np.array(d)
#===========================================================


#===========================================================
def get_coeff_GL_(c, q, w, xL, N, d):

    cL = np.zeros(N+1, dtype=complex)
    i = 0
    for K in range(N+1):
        for l in range(K+1):
            p = -(K+l+1)
            cL[K] += d[i] * (c-xL)**(K-l) * ((1/w+1j*q)**p - (1j*q)**p)
            i += 1

    return np.exp(1j*q*xL) * cL
#===========================================================


#===========================================================
def get_coeff_GR_(c, q, w, xR, N, d):

    cR = np.zeros(N+1, dtype=complex)
    i = 0
    for K in range(N+1):
        for l in range(K+1):
            p = -(K+l+1)
            cR[K] += d[i] * (xR-c)**(K-l) * ((1/w-1j*q)**p - (-1j*q)**p)
            i += 1

    return np.exp(1j*q*xR) * cR
#===========================================================


#===========================================================
def get_E_approx_ty(c, q, w, xL, xR, N, tsmall, tlarge):

    d = get_d(N)
    
    cL = get_coeff_GL_(c, q, w, xL, N, d)
    GL_ = np.zeros(len(tsmall), dtype=complex)
    for i, t in enumerate(tsmall):
        for K in range(N+1):
            GL_[i] += cL[K] * t**(2*K)
        GL_[i] *= np.exp(-t**2 * (c-xL)**2)

    cR = get_coeff_GR_(c, q, w, xR, N, d)
    GR_ = np.zeros(len(tsmall), dtype=complex)
    for i, t in enumerate(tsmall):
        for K in range(N+1):
            GR_[i] += cR[K] * t**(2*K)
        GR_[i] *= np.exp(-t**2 * (c-xR)**2)

    E_small = (GL_ + GR_) / (xR - xL + 2*w)
    E_large = np.sqrt(np.pi) / tlarge * np.exp(1j*q*c - q**2/4/tlarge**2) / \
        (xR - xL + 2*w)

    return E_small, E_large
#===========================================================


#===========================================================
def get_coeff_GL_cb(c, q, w, xL, N, lbound, ubound, with_F=True):
    '''
    Calculates the Chebyshev coefficients of G_L' = G_L - F_L if with_F is true,
    calculates only G_L otherwise. When with_F is True, q must be nonzero.
    '''

    if with_F and abs(q) < 1E-13:
        raise ValueError('get_coeff_GL_cb: When with_F is true, q must be nonzero.')
    
    if lbound >= ubound:
        raise ValueError("Require lbound < ubound")

    t_mid = 0.5 * (lbound + ubound)
    t_half = 0.5 * (ubound - lbound)
    prefactor = np.sqrt(np.pi) / 2 * np.exp(1j*q*xL)

    #--------------------------------------------------------
    def eval_erfcx_GL_(z):
        '''
        Given points within [-1,1] interval, it evaluates erf(at)+erf(bt) at 
        the corresponding points within an arbitrary interval.
        '''        
        t = t_mid + t_half * np.asarray(z)
        aG = t*(c-xL) + (1/w+1j*q)/2/t
        
        f = np.zeros(len(t), dtype=complex)
        mask = abs(t) <= 1E-13
        nmask = abs(t) > 1E-13
        f[nmask] = prefactor * erfcx(aG[nmask]) / t[nmask]
        f[mask] = w * np.exp(1j*q*xL) / (1+1j*q*w)
        if with_F:
            aF = t*(c-xL) + 1j*q/2/t
            f[nmask] -= prefactor * erfcx(aF[nmask]) / t[nmask]
            f[mask] -= -1j * np.exp(1j*q*xL) / q
            
        return f
    #--------------------------------------------------------


    chebinterpolate = np.polynomial.chebyshev.chebinterpolate
    cL = chebinterpolate(eval_erfcx_GL_, N)

    ## Exploit odd symmetry when the interval is symmetric about zero.
    #if np.isclose(lbound, -ubound):
    #    cL[::2] = 0.0

    return cL
#===========================================================


#===========================================================
def get_coeff_GR_cb(c, q, w, xR, N, lbound, ubound, with_F=True):
    '''
    Calculates the Chebyshev coefficients of G_R' = G_R - F_R if with_F is true,
    calculates only G_L otherwise. When with_F is True, q must be nonzero.
    '''

    if with_F and abs(q) < 1E-13:
        raise ValueError('get_coeff_GR_cb: When with_F is true, q must be nonzero.')

    if lbound >= ubound:
        raise ValueError("Require lbound < ubound")

    t_mid = 0.5 * (lbound + ubound)
    t_half = 0.5 * (ubound - lbound)
    prefactor = np.sqrt(np.pi) / 2 * np.exp(1j*q*xR)

    #--------------------------------------------------------
    def eval_erfcx_GR_(z):
        '''
        Given points within [-1,1] interval, it evaluates erf(at)+erf(bt) at 
        the corresponding points within an arbitrary interval.
        '''        
        t = t_mid + t_half * np.asarray(z)
        aG = t*(xR-c) + (1/w-1j*q)/2/t        
        
        f = np.zeros(len(t), dtype=complex)
        mask = abs(t) <= 1E-13
        nmask = abs(t) > 1E-13
        f[nmask] = prefactor * erfcx(aG[nmask]) / t[nmask]
        f[mask] = w * np.exp(1j*q*xR) / (1-1j*q*w)
        if with_F:
            aF = t*(xR-c) - 1j*q/2/t
            f[nmask] -= prefactor * erfcx(aF[nmask]) / t[nmask]
            f[mask] -= 1j * np.exp(1j*q*xR) / q

        return f
    #--------------------------------------------------------


    chebinterpolate = np.polynomial.chebyshev.chebinterpolate
    cR = chebinterpolate(eval_erfcx_GR_, N)

    ## Exploit odd symmetry when the interval is symmetric about zero.
    #if np.isclose(lbound, -ubound):
    #    cR[::2] = 0.0

    return cR
#===========================================================


#===========================================================
def get_coeff_GL(c, q, w, xL, N, d):

    cL = np.zeros(N+1, dtype=complex)
    i = 0
    for K in range(N+1):
        for l in range(K+1):
            p = -(K+l+1)
            cL[K] += d[i] * (c-xL)**(K-l) * (1/w+1j*q)**p
            i += 1

    return np.exp(1j*q*xL) * cL
#===========================================================


#===========================================================
def get_coeff_GR(c, q, w, xR, N, d):

    cR = np.zeros(N+1, dtype=complex)
    i = 0
    for K in range(N+1):
        for l in range(K+1):
            p = -(K+l+1)
            cR[K] += d[i] * (xR-c)**(K-l) * (1/w-1j*q)**p
            i += 1

    return np.exp(1j*q*xR) * cR
#===========================================================


#===========================================================
def get_coeff_FLR_q0_ty(c, xL, xR, N):

    cF = np.zeros(N+1)
    for K in range(N+1):
        cF[i] = (-1)**K * ((xR-c)**(2*K+1) + (c-xL)**(2*K+1)) / factorial(K) / (2*K+1)

    return -cF
#===========================================================


#===========================================================
def get_coeff_FLR_q0_cb(c, xL, xR, N, lbound, ubound):
        
    if lbound >= ubound:
        raise ValueError("Require lbound < ubound")

    t_mid = 0.5 * (lbound + ubound)
    t_half = 0.5 * (ubound - lbound)    

    #--------------------------------------------------------
    def eval_erf_factor(z):
        '''
        Given points within [-1,1] interval, it evaluates erf(at)+erf(bt) at 
        the corresponding points within an arbitrary interval.
        '''        
        t = t_mid + t_half * np.asarray(z)
        f = np.zeros(len(t), dtype=complex)
        mask = abs(t) <= 1E-13
        nmask = abs(t) > 1E-13
        f[nmask] = - np.sqrt(np.pi) / 2 * (erf((xR-c)*t) + erf((c-xL)*t)) / t
        f[mask] = - (xR - xL)
        return f
    #--------------------------------------------------------


    chebinterpolate = np.polynomial.chebyshev.chebinterpolate
    cF = chebinterpolate(eval_erf_factor, N).real

    # Exploit odd symmetry when the interval is symmetric about zero.
    if np.isclose(lbound, -ubound):
        cF[::2] = 0.0

    return cF
#===========================================================


#===========================================================
def get_E_approx_q0_ty(c, w, xL, xR, N, tsmall, tlarge):

    d = get_d(N)
    
    cL = get_coeff_GL(c, 0.0, w, xL, N, d)
    GL = np.zeros(len(tsmall), dtype=complex)
    for i, t in enumerate(tsmall):
        for K in range(N+1):
            GL[i] += cL[K] * t**(2*K)
        GL[i] *= np.exp(-t**2 * (c-xL)**2)

    cR = get_coeff_GR(c, 0.0, w, xR, N, d)
    GR = np.zeros(len(tsmall), dtype=complex)
    for i, t in enumerate(tsmall):
        for K in range(N+1):
            GR[i] += cR[K] * t**(2*K)
        GR[i] *= np.exp(-t**2 * (c-xR)**2)

    FLR = np.zeros(len(tsmall))
    cF = get_coeff_FLR_q0_ty(c, xL, xR, N)
    for i, t in enumerate(tsmall):
        for K in range(N+1):
            FLR[i] += cF[K] * t**(2*K)        
        
    E_small = ((GL + GR) - FLR) / (xR - xL + 2*w)
    E_large = np.sqrt(np.pi) / tlarge / (xR - xL + 2*w)

    return E_small, E_large
#===========================================================


#===========================================================
def get_E_approx_q0_cb(c, w, xL, xR, N, tsmall, tlarge, nu=4):

    chebval = np.polynomial.chebyshev.chebval
    
    lbound = tsmall[0]
    ubound0 = nu * max(1/abs(c-xL), 1/abs(xR-c))
    assert ubound0 <= tsmall[-1], \
        f'ubound0={ubound0:.3e} while the last tsmall is {tsmall[-1]:.3e}.'
    idu = (np.abs(tsmall - ubound0)).argmin()
    ubound = tsmall[idu]
    #^^ The true upper bound should be an element of tsmall.
    print('Chebyshev bound = ', ubound)

    zz = (2 * np.asarray(tsmall[:idu+1]) - lbound - ubound) / (ubound - lbound)
    
    cL = get_coeff_GL_cb(c, 0.0, w, xL, N, lbound, ubound, False).real
    GL = np.zeros(len(tsmall))
    GL[:idu+1] = np.exp(-tsmall[:idu+1]**2 * (c-xL)**2) * chebval(zz, cL)

    cR = get_coeff_GR_cb(c, 0.0, w, xR, N, lbound, ubound, False).real
    GR = np.zeros(len(tsmall))
    GR[:idu+1] = np.exp(-tsmall[:idu+1]**2 * (c-xR)**2) * chebval(zz, cR)

    FLR = np.zeros(len(tsmall))
    cF = get_coeff_FLR_q0_cb(c, xL, xR, N, lbound, ubound)
    FLR[:idu+1] = chebval(zz, cF) 
    FLR[idu+1:] = - np.sqrt(np.pi) / tsmall[idu+1:]
                
    E_small = ((GL + GR) - FLR) / (xR - xL + 2*w)
    E_large = np.sqrt(np.pi) / tlarge / (xR - xL + 2*w)

    return E_small, E_large
#===========================================================


#===========================================================
def get_E_approx_cb(c, q, w, xL, xR, N, tsmall, tlarge, nu=4):

    chebval = np.polynomial.chebyshev.chebval
    
    lbound = tsmall[0]
    ubound0 = nu * max(1/abs(c-xL), 1/abs(xR-c))
    assert ubound0 <= tsmall[-1], \
        f'ubound0={ubound0:.3e} while the last tsmall is {tsmall[-1]:.3e}.'
    idu = (np.abs(tsmall - ubound0)).argmin()
    ubound = tsmall[idu]
    #^^ The true upper bound should be an element of tsmall.
    print('Chebyshev bound = ', ubound)

    zz = (2 * np.asarray(tsmall[:idu+1]) - lbound - ubound) / (ubound - lbound)
    
    cL = get_coeff_GL_cb(c, q, w, xL, N, lbound, ubound, True)
    GL_ = np.zeros(len(tsmall), dtype=complex)
    GL_[:idu+1] = np.exp(-tsmall[:idu+1]**2 * (c-xL)**2) * chebval(zz, cL)

    cR = get_coeff_GR_cb(c, q, w, xR, N, lbound, ubound, True)
    GR_ = np.zeros(len(tsmall), dtype=complex)
    GR_[:idu+1] = np.exp(-tsmall[:idu+1]**2 * (c-xR)**2) * chebval(zz, cR)

    E_small = (GL_ + GR_) / (xR - xL + 2*w)
    E_large = np.sqrt(np.pi) / tlarge * np.exp(1j*q*c - q**2/4/tlarge**2) / \
        (xR - xL + 2*w)

    return E_small, E_large
#===========================================================


#=====================================================================
def savetab_F(tsmall, F_small, tlarge, F_large, fsmall='tsmall.txt', flarge='tlarge.txt'):

    with open(fsmall, 'w') as f:
        f.write(f'#{"No.":>5s} {"t":>10s}   {"Re[F]":>15s} {"Im[F]":>15s}\n')
        for i, t in enumerate(tsmall):
            f.write(f' {i:5d} {t:10.3e}   {F_small[i].real:15.6e} {F_small[i].imag:15.6e}\n')

    with open(flarge, 'w') as f:
        f.write(f'#{"No.":>5s} {"t":>10s}   {"Re[F]":>15s} {"Im[F]":>15s}\n')
        for i, t in enumerate(tlarge):
            f.write(f' {i:5d} {t:10.3e}   {F_large[i].real:15.6e} {F_large[i].imag:15.6e}\n')

    return
#=====================================================================

        
#=====================================================================
def plot_G(x1, f1, x2, f2, filename="complex_plots.pdf", figsize=(12, 5), 
            margins=(0.1, 0.1, 0.9, 0.9), wspace=0.3, xlim1=None, ylim1=None, 
            xlim2=None, ylim2=None, lb_prev='', title=''):
    """
    Plots the real and imaginary parts of two complex vectors against their respective 
    real vectors, saving the result directly to a PDF file with custom figure size and axis limits.

    Parameters:
    - x1, f1: Real input vector and complex vector for the first subplot.
    - x2, f2: Real input vector and complex vector for the second subplot.
    - filename: String, output PDF filename.
    - figsize: Tuple (width, height) in inches for the overall figure.
    - margins: Tuple of (left, bottom, right, top) relative positions (0 to 1).
    - wspace: Float, width space (gap) between the two subplots.
    - xlim1, ylim1: Tuples (min, max) for custom axis limits on the first subplot.
    - xlim2, ylim2: Tuples (min, max) for custom axis limits on the second subplot.
    """

    left, bottom, right, top = margins
    
    # Create figure with custom figure size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: f1 vs x1 ---
    ax1.plot(x1, np.real(f1), color='red', linestyle='-', label=r'$Re[$'+lb_prev+r'$]$')
    ax1.plot(x1, np.imag(f1), color='blue', linestyle='-', label=r'$Im[$'+lb_prev+r'$]$')
    ax1.set_xlabel('t')
    ax1.set_ylabel('G')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Apply custom limits if provided
    if xlim1 is not None: ax1.set_xlim(xlim1)
    if ylim1 is not None: ax1.set_ylim(ylim1)

    # --- Plot 2: f2 vs x2 ---
    ax2.plot(x2, np.real(f2), color='red', linestyle='-', label=r'$Re[$'+lb_prev+r'$]$')
    ax2.plot(x2, np.imag(f2), color='blue', linestyle='-', label=r'$Im[$'+lb_prev+r'$]$')
    ax2.set_xlabel('t')
    ax2.set_ylabel('G')
    ax2.set_title(title)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Apply custom limits if provided
    if xlim2 is not None: ax2.set_xlim(xlim2)
    if ylim2 is not None: ax2.set_ylim(ylim2)

    # Adjust margins and the gap between plots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace)

    # Save figure to PDF and close to prevent showing interactively
    fig.savefig(filename, format='pdf')
    plt.close(fig)

    return
#=====================================================================






if __name__ == '__main__':

    c = (0.0, -1.0, 3.5)
    xL = -600.0
    xR = 600.0
    yL = -600.0
    yR = 600.0
    zL = -1000.0
    zR = 1000.0
    wexp = 80.0
    q = np.array([0.08, 0.0, 0.0])
    nsmall = 500
    nlarge = 500
    tlast = 0.5
    Napp = 15
    expan = 'C'
    
    figsize = (12, 3)
    margins = (0.07, 0.15, 0.98, 0.9)
    wspace = 0.2
            
    
    tsmall = np.linspace(0.0, min(q[abs(q)>0.0])/10, nsmall)
    tlarge = np.linspace(min(q[abs(q)>0.0])/10, tlast, nlarge)
    xsmall = (tsmall[0], tsmall[-1])
    xlarge = (tlarge[0], tlarge[-1])
    
    
    wallL = (xL, yL, zL)
    wallR = (xR, yR, zR)
    labels = ('x', 'y', 'z')
    F_small = np.ones(len(tsmall), dtype=complex)
    F_large = np.ones(len(tlarge), dtype=complex)
    for i in range(3):
        if abs(q[i]) < 1E-12:
            if expan == 'T':
                E_small, E_large = get_E_approx_q0_ty(c[i], wexp, wallL[i], wallR[i], Napp, tsmall, tlarge)
            elif expan == 'C':
                E_small, E_large = get_E_approx_q0_cb(c[i], wexp, wallL[i], wallR[i], Napp, tsmall, tlarge)
        else:
            if expan == 'T':
                E_small, E_large = get_E_approx_ty(c[i], q[i], wexp, wallL[i], wallR[i], Napp, tsmall, tlarge)
            elif expan == 'C':
                E_small, E_large = get_E_approx_cb(c[i], q[i], wexp, wallL[i], wallR[i], Napp, tsmall, tlarge)
                
        if i==1:
            ylim1 = None#(-0.1,1)
        else:
            ylim1 = None
            
        plot_G(tsmall, E_small, tlarge, E_large, filename="E_"+labels[i]+".pdf",
               xlim1=xsmall, xlim2=xlarge, lb_prev=rf'$E_{{{labels[i]}}}(t)$',
               figsize=figsize, margins=margins, wspace=wspace, ylim1=ylim1)
    
        F_small *= E_small
        F_large *= E_large
    
    
    plot_G(tsmall, F_small, tlarge, F_large, filename="F.pdf", xlim1=xsmall, xlim2=xlarge,
           lb_prev=rf'$F(t)$', figsize=figsize, margins=margins, wspace=wspace)
    savetab_F(tsmall, F_small, tlarge, F_large, fsmall='Fsmall.txt', flarge='Flarge.txt')
