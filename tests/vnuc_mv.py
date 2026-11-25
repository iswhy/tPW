import time
import numpy as np
import tPW.tpw as tpw
import tPW.pwcb.pwcb as pwcb


c = (0.0, -1.0, 3.5)
Zn = 1.0
Nkx = 4
Nky = 4
Nkz = 6

kxlim = (-0.8, 0.8)
kylim = (-0.8, 0.8)
kzlim = (-3.0, 3.0)
kxvec = np.linspace(kxlim[0], kxlim[1], Nkx)
kyvec = np.linspace(kylim[0], kylim[1], Nky)
kzvec = np.linspace(kzlim[0], kzlim[1], Nkz)
#pw_k = tpw.assemble_ks(kxvec, kyvec, kzvec)
w = 10.0
xlim = (-800.0, 800.0)
ylim = (-800.0, 800.0)
zlim = (-1000.0, 1000.0)
Napp = 15

nfactor = \
    (xlim[1] - xlim[0] + 2*w) * \
    (ylim[1] - ylim[0] + 2*w) * \
    (zlim[1] - zlim[0] + 2*w)

vnuc = tpw.get_vnuc_tpw_ctr(Zn, c, kxvec, kyvec, kzvec, xlim, ylim, zlim, w, Napp, nfactor=nfactor)
Nks = (Nkx, Nky, Nkz)
qint = (kxvec[1]-kxvec[0],
        kyvec[1]-kyvec[0],
        kzvec[1]-kzvec[0])
nM = (2*Nkx, 2*Nky, 2*Nkz)
Cr = pwcb.calc_vnuc_circulant_ts(Zn, c, Nks, qint, nM, w, xlim, ylim, zlim, nfactor=nfactor)
FCr = np.fft.fftn(Cr)

vin = np.random.rand(Nkx*Nky*Nkz)

time0 = time.time()
vout_ref = vnuc @ vin
print('Elapsed time for direct matvec = %.3f s' % (time.time() - time0), flush=True)

time0 = time.time()
vout = pwcb.matvec_3d_toeplitz(FCr, vin, Nks)
print('Elapsed time for BCCB matvec = %.3f s' % (time.time() - time0), flush=True)

print('vout_ref = ', vout_ref)
print('vout = ', vout)
print(f'Max error of real part = {np.max(np.abs(vout.real - vout_ref.real)):.5e}')
print(f'Max error of imag part = {np.max(np.abs(vout.imag - vout_ref.imag)):.5e}')
