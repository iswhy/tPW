from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tPW.approx.approx as approx
import tPW.exact.exact as exact
import tPW.pwcb.pwcb as pwcb




#======================================================
def savetab_error(t, F_app, F_ex, q, c, wallL, wallR, w, Napp, fname='error.txt'):
    
    with open(fname, 'w') as f:
        f.write(f'#Momentum transfer (x,y,z) = {q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f}\n')
        f.write(f'#Center coordinate (x,y,z) = {c[0]:.3f}  {c[1]:.3f}  {c[2]:.3f}\n')
        f.write(f'#x pure plane wave bounds = {wallL[0]:.3f}  {wallR[0]:.3f}\n')
        f.write(f'#y pure plane wave bounds = {wallL[1]:.3f}  {wallR[1]:.3f}\n')
        f.write(f'#z pure plane wave bounds = {wallL[2]:.3f}  {wallR[2]:.3f}\n')
        f.write(f'#Exponential tail = {w:.3f}\n')
        f.write(f'#Max. Chebyshev order = {Napp:d}\n')
        
        f.write(f'#{"No.":>5s} {"t":>10s}' +
                f'   {"exact Re[F]":>15s} {"exact Im[F]":>15s}' +
                f'   {"approx. Re[F]":>15s} {"approx. Im[F]":>15s}' +
                f'   {"Re[error]":>15s} {"Im[error]":>15s}\n')
        
        for i, t in enumerate(t):
            er = F_app[i] - F_ex[i]
            f.write(f' {i:5d} {t:10.3e}' +
                    f'   {F_ex[i].real:15.6e} {F_ex[i].imag:15.6e}' +
                    f'   {F_app[i].real:15.6e} {F_app[i].imag:15.6e}' +
                    f'   {er.real:15.6e} {er.imag:15.6e}\n')
            
    return
#======================================================


#=====================================================================
def plot_data(x1, f1_app, f1_ex, x2, f2_app, f2_ex, filename="complex_plots.pdf", 
              figsize=(12, 5), margins=(0.1, 0.1, 0.9, 0.9), wspace=0.3, xlim1=None, 
              ylim1=None, xlim2=None, ylim2=None, lb_prev='', title=''):
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
    ax1.plot(x1, np.real(f1_ex), color='red', linestyle='-', label=r'exact $Re[$'+lb_prev+r'$]$')
    ax1.plot(x1, np.imag(f1_ex), color='blue', linestyle='-', label=r'exact $Im[$'+lb_prev+r'$]$')
    ax1.plot(x1, np.real(f1_app), color='#ff9395', linestyle='--', label=r'approx. $Re[$'+lb_prev+r'$]$')
    ax1.plot(x1, np.imag(f1_app), color='#7083ff', linestyle='--', label=r'approx. $Im[$'+lb_prev+r'$]$')
    ax1.set_xlabel('t')
    ax1.set_ylabel('F')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Apply custom limits if provided
    if xlim1 is not None: ax1.set_xlim(xlim1)
    if ylim1 is not None: ax1.set_ylim(ylim1)

    # --- Plot 2: f2 vs x2 ---
    ax2.plot(x2, np.real(f2_ex), color='red', linestyle='-', label=r'exact $Re[$'+lb_prev+r'$]$')
    ax2.plot(x2, np.imag(f2_ex), color='blue', linestyle='-', label=r'exact $Im[$'+lb_prev+r'$]$')
    ax2.plot(x2, np.real(f2_app), color='#ff9395', linestyle='--', label=r'approx. $Re[$'+lb_prev+r'$]$')
    ax2.plot(x2, np.imag(f2_app), color='#7083ff', linestyle='--', label=r'approx. $Im[$'+lb_prev+r'$]$')
    ax2.set_xlabel('t')
    ax2.set_ylabel('F')
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


#=====================================================================
def plot_error(x1, f1_app, f1_ex, x2, f2_app, f2_ex, filename="error_plots.pdf", 
               figsize=(12, 5), margins=(0.1, 0.1, 0.9, 0.9), wspace=0.3, xlim1=None, 
               ylim1=None, xlim2=None, ylim2=None, lb_prev='', title=''):
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

    def get_rel_error(f_app, f_ex):
        re = abs((f_app-f_ex).real) / abs(f_ex.real)
        im = abs((f_app-f_ex).imag) / abs(f_ex.imag)
        return re, im
    
    # --- Plot 1: f1 vs x1 ---
    rel_error_r, rel_error_i = get_rel_error(f1_app, f1_ex)
    ax1.plot(x1, rel_error_r, color='red', linestyle='-', label=r'$Re[$'+lb_prev+r'$]$')
    ax1.plot(x1, rel_error_i, color='blue', linestyle='--', label=r'$Im[$'+lb_prev+r'$]$')
    ax1.set_yscale("log", base=10)
    ax1.set_xlabel('t')
    ax1.set_ylabel('Rel. error')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Apply custom limits if provided
    if xlim1 is not None: ax1.set_xlim(xlim1)
    if ylim1 is not None: ax1.set_ylim(ylim1)

    # --- Plot 2: f2 vs x2 ---
    rel_error_r, rel_error_i = get_rel_error(f2_app, f2_ex)
    ax2.plot(x2, rel_error_r, color='red', linestyle='-', label=r'$Re[$'+lb_prev+r'$]$')
    ax2.plot(x2, rel_error_i, color='blue', linestyle='--', label=r'$Im[$'+lb_prev+r'$]$')
    ax2.set_yscale("log", base=10)
    ax2.set_xlabel('t')
    ax2.set_ylabel('Rel. error')
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




c = (0.0, -1.0, 3.5)
xL = -800.0
xR =  800.0
yL = -800.0
yR =  800.0
zL = -1000.0
zR =  1000.0
wexp = 10.0
q = np.array([0.0, -0.0, 0.5])
#q = np.array([1.0, 1.0, 2.0])
nsmall = 500
nlarge = 500
tlast = 6.0
Napp = 15
expan = 'C'

figsize = (12, 3)
margins = (0.07, 0.15, 0.98, 0.9)
wspace = 0.2

wallL = (xL, yL, zL)
wallR = (xR, yR, zR)
nu = 3
max_width = max( [nu * max(1/abs(c[i]-wallL[i]), 1/abs(wallR[i]-c[i])) for i in range(3)] )
min_onset = min( [ np.sqrt(q_**2/4/12*(9-np.sqrt(33))) for q_ in q[abs(q)>0.0] ] )
print(f'Max. small-t part width = {max_width:.3e}')
print(f'Min. large-t part onset = {min_onset:.3e}')
assert max_width < min_onset, 'Consider decreasing nu.'
t_boundary = min(2*max_width, (max_width+min_onset)/2)
tsmall = np.linspace(0.0, t_boundary, nsmall)
tlarge = np.linspace(t_boundary, tlast, nlarge)
xsmall = (tsmall[0], tsmall[-1])
xlarge = (tlarge[0], tlarge[-1])


labels = ('x', 'y', 'z')
F_small = np.ones(len(tsmall), dtype=complex)
F_large = np.ones(len(tlarge), dtype=complex)
F_small_ex = np.ones(len(tsmall), dtype=complex)
F_large_ex = np.ones(len(tlarge), dtype=complex)

for i in range(3):
    #==== Approximate data ====#
    if abs(q[i]) < 1E-12:
        if expan == 'T':
            E_small, E_large = approx.get_E_approx_q0_ty(c[i], wexp, wallL[i], wallR[i], Napp, tsmall, tlarge)
        elif expan == 'C':
            E_small, E_large = approx.get_E_approx_q0_cb(c[i], wexp, wallL[i], wallR[i], Napp, tsmall, tlarge, nu)
    else:
        if expan == 'T':
            E_small, E_large = approx.get_E_approx_ty(c[i], q[i], wexp, wallL[i], wallR[i], Napp, tsmall, tlarge)
        elif expan == 'C':
            E_small, E_large = approx.get_E_approx_cb(c[i], q[i], wexp, wallL[i], wallR[i], Napp, tsmall, tlarge, nu)

    F_small *= E_small
    F_large *= E_large


    #==== Exact data ====#
    E_small_ex = exact.calc_E(tsmall, c[i], wallL[i], wallR[i], wexp, q[i]) / (wallR[i] - wallL[i] + 2*wexp)
    E_large_ex = exact.calc_E(tlarge, c[i], wallL[i], wallR[i], wexp, q[i]) / (wallR[i] - wallL[i] + 2*wexp)

    F_small_ex *= E_small_ex
    F_large_ex *= E_large_ex


    plot_data(tsmall, E_small, E_small_ex, tlarge, E_large, E_large_ex, filename="E_"+labels[i]+".pdf", 
              figsize=figsize, margins=margins, wspace=0.3, xlim1=xsmall, xlim2=xlarge, 
              lb_prev=rf'$E_{{{labels[i]}}}(t)$', title=rf'$q_{{{labels[i]}}}={q[i]:.3f}$')


#==== PLotting approximate and exact F ====#
plot_data(tsmall, F_small, F_small_ex, tlarge, F_large, F_large_ex, filename="F.pdf",
          figsize=figsize, margins=margins, wspace=0.3, xlim1=xsmall, xlim2=xlarge,
          lb_prev=rf'$F(t)$', title=f'F(t) for q = ({q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f})')
plot_error(tsmall, F_small, F_small_ex, tlarge, F_large, F_large_ex, filename="error.pdf",
           figsize=figsize, margins=margins, wspace=0.3, xlim1=xsmall, xlim2=xlarge,
           lb_prev=rf'$\epsilon$', title=f'F(t) error for q = ({q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f})')

savetab_error(tsmall, F_small, F_small_ex, q, c, wallL, wallR, wexp, Napp, 'error_tsmall.txt')
error_small = F_small - F_small_ex
print('Maximum errors for the small-t part:')
print(f'  - Real part = {np.max( np.abs(error_small.real) ):.8e} \n' +
      f'  - Imaginary part = {np.max( np.abs(error_small.imag) ):.8e}')

savetab_error(tlarge, F_large, F_large_ex, q, c, wallL, wallR, wexp, Napp, 'error_tlarge.txt')
error_large = F_large - F_large_ex
print('Maximum errors for the large-t part:')
print(f'  - Real part = {np.max( np.abs(error_large.real) ):.8e} \n' +
      f'  - Imaginary part = {np.max( np.abs(error_large.imag) ):.8e}')




F_small_pl, F_large_pl = pwcb.get_F_approx_pl(c, q, wexp, wallL, wallR, Napp, tsmall, tlarge, nu,
                                              (wallR[0] - wallL[0] + 2*wexp) * \
                                              (wallR[1] - wallL[1] + 2*wexp) * \
                                              (wallR[2] - wallL[2] + 2*wexp))
plot_data(tsmall, F_small_pl, F_small_ex, tlarge, F_large_pl, F_large_ex, filename="F_pl.pdf",
          figsize=figsize, margins=margins, wspace=0.3, xlim1=xsmall, xlim2=xlarge,
          lb_prev=rf'$F(t)$', title=f'F(t) for q = ({q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f})')
plot_error(tsmall, F_small_pl, F_small_ex, tlarge, F_large_pl, F_large_ex, filename="error_pl.pdf",
           figsize=figsize, margins=margins, wspace=0.3, xlim1=xsmall, xlim2=xlarge,
           lb_prev=rf'$\epsilon$', title=f'F(t) error for q = ({q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f})')

savetab_error(tsmall, F_small_pl, F_small_ex, q, c, wallL, wallR, wexp, Napp, 'error_pl_tsmall.txt')
error_small = F_small_pl - F_small_ex
print('Maximum errors for the polynomial-approximated small-t part:')
print(f'  - Real part = {np.max( np.abs(error_small.real) ):.8e} \n' +
      f'  - Imaginary part = {np.max( np.abs(error_small.imag) ):.8e}')

savetab_error(tlarge, F_large_pl, F_large_ex, q, c, wallL, wallR, wexp, Napp, 'error_pl_tlarge.txt')
error_large = F_large_pl - F_large_ex
print('Maximum errors for the polynomial-approximated large-t part:')
print(f'  - Real part = {np.max( np.abs(error_large.real) ):.8e} \n' +
      f'  - Imaginary part = {np.max( np.abs(error_large.imag) ):.8e}')





nfactor = (wallR[0] - wallL[0] + 2*wexp) * \
    (wallR[1] - wallL[1] + 2*wexp) * \
    (wallR[2] - wallL[2] + 2*wexp)
#nfactor = 1.0
assert np.linalg.norm(q) >= 1E-12, 'The following codes only works on nonzero |q|.'
F_small_pl2 = exact.calc_F(tsmall, c, (wallL[0], wallR[0]), (wallL[1], wallR[1]), (wallL[2], wallR[2]), wexp, q, True, nfactor)
F_large_pl2 = exact.calc_F(tlarge, c, (wallL[0], wallR[0]), (wallL[1], wallR[1]), (wallL[2], wallR[2]), wexp, q, True, nfactor)
plot_data(tsmall, F_small_pl2, F_small_ex, tlarge, F_large_pl2, F_large_ex, filename="F_pl2.pdf",
          figsize=figsize, margins=margins, wspace=0.3, xlim1=xsmall, xlim2=xlarge,
          lb_prev=rf'$F(t)$', title=f'F(t) for q = ({q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f})')
print('\n !!! WARNING !!! The dashed lines in the large-t portion of F_pl2.pdf is just a dummy small-valued function.\n')

_calc_F_tsmall = partial(exact.calc_F,
                         c=c, xlim=(wallL[0], wallR[0]), ylim=(wallL[1], wallR[1]),
                         zlim=(wallL[2], wallR[2]), w=wexp, q=q, smallt=True, nfactor=nfactor)
intcheck = 2 / np.sqrt(np.pi) * pwcb.trapezoidal_integral(_calc_F_tsmall, 0.0, 0.005, 5000)
print(f'Integral of the small-t part = {intcheck:.10e}')
