
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfcx



#=====================================================================
def calc_G0_erf(t, c, xR, xL, q):
    t = np.atleast_1d(t)
    G0 = []
    for i, t_ in enumerate(t):
        if abs(t_) < 1E-12:
            G0 += [(np.exp(1j*q*xR) - np.exp(1j*q*xL))/1j/q]
        else:
            G0_ = np.sqrt(np.pi) / 2 / t_ * np.exp(1j*q*c - q**2/4/t_**2)
            aR = t_*(xR-c) - 1j*q/2/t_
            aL = t_*(xL-c) - 1j*q/2/t_
            #print('%5d %10.3e   %15.6e %15.6e %15.6e %15.6e' % (i, t_, aR.real, aR.imag, aL.real, aL.imag))
            G0_ *= erf(aR) - erf(aL)
            G0 += [G0_]

    return np.array(G0)
#=====================================================================


#=====================================================================
def calc_G0(t, c, xR, xL, q):
    #print('q = ', q, type(q))
    #t = np.atleast_1d(t)
    #G0 = []
    #for i, t_ in enumerate(t):
    #    if abs(t_) < 1E-12:
    #        if abs(q) < 1E-12:
    #            G0 += [np.complex128(xR-xL+0j)]
    #        else:
    #            G0 += [(np.exp(1j*q*xR) - np.exp(1j*q*xL))/1j/q]
    #        print('t0 t_ = ', t_, G0[-1], type(t_), type(G0[-1]))
    #    else:
    #        G0_ = np.sqrt(np.pi) / t_ * np.exp(1j*q*c - q**2/4/t_**2)
    #        aR = t_*(xR-c) - 1j*q/2/t_
    #        aL = t_*(c-xL) + 1j*q/2/t_
    #        G0_ += - np.sqrt(np.pi) / 2 / t_ * (np.exp(1j*q*xL-t_**2*(xL-c)**2) * erfcx(aL) +
    #                                            np.exp(1j*q*xR-t_**2*(xR-c)**2) * erfcx(aR))
    #        #print('%5d %10.3e   %15.6e %15.6e %15.6e %15.6e' % (i, t_, aR.real, aR.imag, aL.real, aL.imag))
    #        G0 += [G0_]
    #        print('t_ = ', t_, G0[-1], type(t_), type(G0[-1]), np.shape(c), np.shape(xL), np.shape(xR), np.shape(q))

            


    if np.ndim(t) == 0:
        tt = np.array([t])
    else:
        tt = np.ravel(t)
    maskt0 = np.abs(tt) < 1E-12
    masktn0 = np.abs(tt) >= 1E-12

    aL = np.zeros(len(tt), dtype=complex)
    aR = np.zeros(len(tt), dtype=complex)
    G0 = np.zeros(len(tt), dtype=complex)


    #==== when t /= 0 ====#
    G0[masktn0] = np.sqrt(np.pi) / tt[masktn0] * np.exp(1j*q*c - q**2/4/tt[masktn0]**2)
    aR[masktn0] = tt[masktn0]*(xR-c) - 1j*q/2/tt[masktn0]
    aL[masktn0] = tt[masktn0]*(c-xL) + 1j*q/2/tt[masktn0]
    G0[masktn0] += - np.sqrt(np.pi) / 2 / tt[masktn0] * \
        (np.exp(1j*q*xL - tt[masktn0]**2*(xL-c)**2) * erfcx(aL[masktn0]) + \
         np.exp(1j*q*xR - tt[masktn0]**2*(xR-c)**2) * erfcx(aR[masktn0]))

    #==== When t = 0 ====#
    if abs(q) < 1E-12:
        G0[maskt0] = np.complex128(xR-xL + 0j)
    else:
        G0[maskt0] = (np.exp(1j*q*xR) - np.exp(1j*q*xL))/1j/q

        
    
    shapes = [np.shape(item) for item in G0]
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        for idx, (ti, item, shp) in enumerate(zip(t, G0, shapes)):
            print('lll', idx, item)
            if shp != shapes[0]:
                print(f"Mismatch at idx={idx}, t={ti}: shape={shp}, item={item!r}")
                #break
    #print('a2 : ', len(G0))

    if np.ndim(t) == 0:
        return G0.item()
    else:
        return G0.reshape(t.shape)
#=====================================================================


#=====================================================================
def calc_GL(t, c, xL, w, q):
    #print('dc: ', t, t.shape)
    #t = np.atleast_1d(t)
    #GL = []
    #for i, t_ in enumerate(t):
    #    #print('c: ', i, t_)
    #    if abs(t_) < 1E-12:
    #        GL += [np.exp(1j*q*xL) / (1/w+1j*q)]
    #    else:
    #        aL = t_*(c-xL) + (1/w+1j*q)/2/t_
    #        GL_ = np.sqrt(np.pi) / 2 / t_ * np.exp(1j*q*xL-t_**2*(xL-c)**2) * erfcx(aL)
    #        #print('GL t_ = ', t_, GL_, type(t_), type(GL_))
    #        GL += [GL_]


    if np.ndim(t) == 0:
        tt = np.array([t])
    else:
        tt = np.ravel(t)
    maskt0 = np.abs(tt) < 1E-12
    masktn0 = np.abs(tt) >= 1E-12
    
    aL = np.zeros(len(tt), dtype=complex)
    GL = np.zeros(len(tt), dtype=complex)

    aL[masktn0] = tt[masktn0]*(c-xL) + (1/w+1j*q)/2/tt[masktn0]
    GL[masktn0] = np.sqrt(np.pi) / 2 / tt[masktn0] * \
        np.exp(1j*q*xL - tt[masktn0]**2*(xL-c)**2) * erfcx(aL[masktn0])

    GL[maskt0] = np.exp(1j*q*xL) / (1/w+1j*q)

    if np.ndim(t) == 0:
        return GL.item()
    else:
        return GL.reshape(t.shape)
#=====================================================================


#=====================================================================
def calc_GR(t, c, xR, w, q):
    #t = np.atleast_1d(t)
    #GR = []
    #for i, t_ in enumerate(t):
    #    if abs(t_) < 1E-12:
    #        GR += [np.exp(1j*q*xR) / (1/w-1j*q)]
    #    else:
    #        aR = t_*(xR-c) + (1/w-1j*q)/2/t_
    #        GR_ = np.sqrt(np.pi) / 2 / t_ * np.exp(1j*q*xR-t_**2*(xR-c)**2) * erfcx(aR)
    #        GR += [GR_]

    if np.ndim(t) == 0:
        tt = np.array([t])
    else:
        tt = np.ravel(t)
    maskt0 = np.abs(tt) < 1E-12
    masktn0 = np.abs(tt) >= 1E-12


    aR = np.zeros(len(tt), dtype=complex)
    GR = np.zeros(len(tt), dtype=complex)

    aR[masktn0] = tt[masktn0]*(xR-c) + (1/w-1j*q)/2/tt[masktn0]
    GR[masktn0] = np.sqrt(np.pi) / 2 / tt[masktn0] * \
        np.exp(1j*q*xR - tt[masktn0]**2*(xR-c)**2) * erfcx(aR[masktn0])

    GR[maskt0] = np.exp(1j*q*xR) / (1/w-1j*q)

    if np.ndim(t) == 0:
        return GR.item()
    else:
        return GR.reshape(t.shape)
#=====================================================================


#=====================================================================
def calc_E(t, c, xL, xR, w, q):

    a1 = calc_GL(t, c, xL, w, q)
    a2 = calc_G0(t, c, xR, xL, q)
    a3 = calc_GR(t, c, xR, w, q)
    return a1 + a2 + a3
#=====================================================================


#=====================================================================
def calc_F_tsmall(t, c, xlim, ylim, zlim, w, q, nfactor=1.0):

    if np.ndim(t) == 0:
        tt = np.array([t])
    else:
        tt = np.ravel(t)
    maskt0 = np.abs(tt) < 1E-12
    masktn0 = np.abs(tt) >= 1E-12

    #t = np.squeeze(t)
    #print('            >> calcF q = ', q, np.shape(q))
    
    #t = np.atleast_1d(t)
    wallL = (xlim[0], ylim[0], zlim[0])
    wallR = (xlim[1], ylim[1], zlim[1])
    F = np.ones(len(tt), dtype=complex)
    for i in range(3):
        #print('                 calcF q = ', i, q[i], type(q[i]))
        F *= calc_E(tt, c[i], wallL[i], wallR[i], w, q[i])

    #for i, t_ in enumerate(np.atleast_1d(t)):
    #    if abs(t_) < 1E-12:
    #        F[i] -= 0.0
    #    else:
    #        F[i] -= np.sqrt(np.pi**3) / t_**3 * \
    #            np.exp(1j*np.dot(q,c) - np.linalg.norm(q)**2/(4*t_**2))

    F[masktn0] -= np.sqrt(np.pi**3) / tt[masktn0]**3 * \
                np.exp(1j*np.dot(q,c) - np.linalg.norm(q)**2/(4*tt[masktn0]**2))
    F[maskt0] -= 0.0
    
    #print('t F = ', t, F)
    #print(' *** F shape =', F.shape)

    if np.ndim(t) == 0:
        return F.item() / nfactor
    else:
        return F.reshape(t.shape) / nfactor
    
    #return F
#=====================================================================


#=====================================================================
def savetab_G(tsmall, G0_small, GL_small, GR_small, tlarge, G0_large, GL_large, GR_large,
              fsmall='tsmall.txt', flarge='tlarge.txt'):

    with open(fsmall, 'w') as f:
        f.write(f'#{"No.":>5s} {"t":>10s}' +
                f'   {"Re[G0]":>15s} {"Im[G0]":>15s}   {"Re[GL]":>15s} {"Im[GL]":>15s}   {"Re[GR]":>15s} {"Im[GR]":>15s}\n')
        for i, t in enumerate(tsmall):
            f.write(f' {i:5d} {t:10.3e}' +
                    f'   {G0_small[i].real:15.6e} {G0_small[i].imag:15.6e}' +
                    f'   {GL_small[i].real:15.6e} {GL_small[i].imag:15.6e}' +
                    f'   {GR_small[i].real:15.6e} {GR_small[i].imag:15.6e}\n')

    with open(flarge, 'w') as f:
        f.write(f'#{"No.":>5s} {"t":>10s}' +
                f'   {"Re[G0]":>15s} {"Im[G0]":>15s}   {"Re[GL]":>15s} {"Im[GL]":>15s}   {"Re[GR]":>15s} {"Im[GR]":>15s}\n')
        for i, t in enumerate(tlarge):
            f.write(f' {i:5d} {t:10.3e}' +
                    f'   {G0_large[i].real:15.6e} {G0_large[i].imag:15.6e}' +
                    f'   {GL_large[i].real:15.6e} {GL_large[i].imag:15.6e}' +
                    f'   {GR_large[i].real:15.6e} {GR_large[i].imag:15.6e}\n')

    return
#=====================================================================


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
    q = np.array([0.08, 0.00, 0.0])
    nsmall = 500
    nlarge = 500
    tlast = 0.5
    
    figsize = (12, 3)
    margins = (0.07, 0.15, 0.98, 0.9)
    wspace = 0.2
            
    
    
    small_scale_x = 2 / (abs(xR-c[0]) + abs(c[0]-xL))
    small_scale_y = 2 / (abs(yR-c[1]) + abs(c[1]-yL))
    small_scale_z = 2 / (abs(zR-c[2]) + abs(c[2]-zL))
    print('t scales for the small-t part (x,y,z) = %.3e %.3e %.3e' %
          (small_scale_x, small_scale_y, small_scale_z))
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
        G0_small = calc_G0(tsmall, c[i], wallR[i], wallL[i], q[i])
        G0_large = calc_G0(tlarge, c[i], wallR[i], wallL[i], q[i])
        plot_G(tsmall, G0_small, tlarge, G0_large, filename="G_"+labels[i]+"0.pdf",
               xlim1=xsmall, xlim2=xlarge, lb_prev=rf'$G_{{{labels[i]}0}}(t)$',
               figsize=figsize, margins=margins, wspace=wspace,
               title=rf'$q_{{{labels[i]}}}={q[i]:.3f}$')
        
        GL_small = calc_GL(tsmall, c[i], wallL[i], wexp, q[i])
        GL_large = calc_GL(tlarge, c[i], wallL[i], wexp, q[i])
        plot_G(tsmall, GL_small, tlarge, GL_large, filename="G_"+labels[i]+"L.pdf",
               xlim1=xsmall, xlim2=xlarge, lb_prev=rf'$G_{{{labels[i]}L}}(t)$',
               figsize=figsize, margins=margins, wspace=wspace,
               title=rf'$q_{{{labels[i]}}}={q[i]:.3f}$')
        
        GR_small = calc_GR(tsmall, c[i], wallR[i], wexp, q[i])
        GR_large = calc_GR(tlarge, c[i], wallR[i], wexp, q[i])
        plot_G(tsmall, GR_small, tlarge, GR_large, filename="G_"+labels[i]+"R.pdf",
               xlim1=xsmall, xlim2=xlarge, lb_prev=rf'$G_{{{labels[i]}R}}(t)$',
               figsize=figsize, margins=margins, wspace=wspace,
               title=rf'$q_{{{labels[i]}}}={q[i]:.3f}$')
        
        E_small = (GL_small + G0_small + GR_small) / (wallR[i] - wallL[i] + 2*wexp)
        E_large = (GL_large + G0_large + GR_large) / (wallR[i] - wallL[i] + 2*wexp)
        if i==1:
            ylim1 = None#(-0.1,1)
        else:
            ylim1 = None
        plot_G(tsmall, E_small, tlarge, E_large, filename="E_"+labels[i]+".pdf",
               xlim1=xsmall, xlim2=xlarge, lb_prev=rf'$E_{{{labels[i]}}}(t)$',
               figsize=figsize, margins=margins, wspace=wspace,
               title=rf'$q_{{{labels[i]}}}={q[i]:.3f}$', ylim1=ylim1)
    
        F_small *= E_small
        F_large *= E_large
        
        savetab_G(tsmall, G0_small, GL_small, GR_small, tlarge, G0_large, GL_large, GR_large,
                  fsmall='tsmall-'+labels[i]+'.txt', flarge='tlarge-'+labels[i]+'.txt')
    
    
    plot_G(tsmall, F_small, tlarge, F_large, filename="F.pdf", xlim1=xsmall, xlim2=xlarge,
           lb_prev=rf'$F(t)$', figsize=figsize, margins=margins, wspace=wspace,
           title=r'$F(t) = E_x(t) E_y(t) E_z(t)$')
    savetab_F(tsmall, F_small, tlarge, F_large, fsmall='Fsmall.txt', flarge='Flarge.txt')
