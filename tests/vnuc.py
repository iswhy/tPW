import numpy as np
import tPW.pwcb.pwcb as pwcb

c = (0.0, -1.0, 3.5)
#nk = (20, 20, 150)
nk = (1, 1, 2)
#qint = (0.08, 0.08, 0.04)
qint = (1.0, 1.0, 2.0)
nM = (2*nk[0], 2*nk[1], 2*nk[2])
w = 1.0
xlim = (-800.0, 800.0)
ylim = (-800.0, 800.0)
zlim = (-1000.0, 1000.0)
Napp = 20



nfactor = \
    (xlim[1] - xlim[0] + 2*w) * \
    (ylim[1] - ylim[0] + 2*w) * \
    (zlim[1] - zlim[0] + 2*w)
Cr = pwcb.calc_vnuc_circulant(c, nk, qint, nM, w, xlim, ylim, zlim, Napp, lbound=0.0, nu=4,
                              nfactor=nfactor)
Cr_ref = pwcb.calc_vnuc_circulant_ts(c, nk, qint, nM, w, xlim, ylim, zlim, nfactor=nfactor)
