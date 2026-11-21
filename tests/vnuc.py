import numpy as np
import tPW.pwcb.pwcb as pwcb

c = (1.0, -2.5, 3.0)
nk = (20, 20, 150)
#nk = (3, 3, 4)
qint = (0.08, 0.08, 0.04)
nM = (2*nk[0], 2*nk[1], 2*nk[2])
w = 100.0
wall = 1000.0
xlim = (-wall, wall)
ylim = (-wall, wall)
zlim = (-wall, wall)
Napp = 20




Cr = pwcb.calc_vnuc_circulant(c, nk, qint, nM, w, xlim, ylim, zlim, Napp, lbound=0.0, nu=4)
