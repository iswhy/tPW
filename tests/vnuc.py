import numpy as np
import tPW.pwcb.pwcb as pwcb

c = (0.0, -1.0, 3.5)
#nk = (20, 20, 150)
nk = (2, 2, 2)
qint = (0.08, 0.08, 0.04)
#qint = (1.0, 1.0, 0.5)
nM = (2*nk[0], 2*nk[1], 2*nk[2])
w = 10.0
xlim = (-800.0, 800.0)
ylim = (-800.0, 800.0)
zlim = (-1000.0, 1000.0)
Napp = 15


Zn = 1.0
nfactor = \
    (xlim[1] - xlim[0] + 2*w) * \
    (ylim[1] - ylim[0] + 2*w) * \
    (zlim[1] - zlim[0] + 2*w)
larget = True
Cr = pwcb.calc_vnuc_circulant(Zn, c, nk, qint, nM, w, xlim, ylim, zlim, Napp, lbound=0.0, nu=4,
                              nfactor=nfactor, verb=2, larget=larget)
Cr_ref = pwcb.calc_vnuc_circulant_ts(Zn, c, nk, qint, nM, w, xlim, ylim, zlim, nfactor=nfactor,
                                     verb=2, larget=larget)



padded = np.zeros(nM, dtype=bool)
for i in range(nM[0]):
    xpad = i >= nk[0] and i <= nM[0]-nk[0]
    for j in range(nM[1]):
        ypad = j >= nk[1] and j <= nM[1]-nk[1]
        for k in range(nM[2]):
            zpad = k >= nk[2] and k <= nM[2]-nk[2]

            padded[i,j,k] = xpad or ypad or zpad

error = Cr[~padded] - Cr_ref[~padded]
max_error_r = np.max(np.abs(error.real))
max_error_i = np.max(np.abs(error.imag))
max_r_id = np.argmax(np.abs(error.real))
max_i_id = np.argmax(np.abs(error.imag))
print(f'Absolute errors (r,i) = {max_error_r:.5e}  {max_error_i:.5e}')
print(f'Cr at max error (r,i) = {Cr.real[~padded][max_r_id]:.10e}  {Cr.imag[~padded][max_i_id]:.10e}')
print(f'Cr_ref at max error (r,i) = {Cr_ref.real[~padded][max_r_id]:.10e}  {Cr_ref.imag[~padded][max_i_id]:.10e}')


rel_error_r = error.real / ((Cr+Cr_ref)[~padded]/2).real
rel_error_i = error.imag / ((Cr+Cr_ref)[~padded]/2).imag
max_rel_error_r = np.max(np.abs(rel_error_r))
max_rel_error_i = np.max(np.abs(rel_error_i))
print(f'Relative errors (r,i) = {max_rel_error_r:.5e}  {max_rel_error_i:.5e}')
