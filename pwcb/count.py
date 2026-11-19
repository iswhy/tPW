import numpy as np
import math


def count_tcoeffs(nx, ny, nz, K):
    n = 0
    for mx in range(nx+1):
        for my in range(ny+1):
            for mz in range(nz+1):

                if mx+my+mz == K:
                    n += 1
    return n



N_max = 20

#i = 0
#for nx in range(N_max+1):
#    for ny in range(N_max+1):
#        for nz in range(N_max+1):
#
#            if nx+ny+nz <= N_max and nx <= ny and ny <= nz:
#                i += 1
#
#print(i)
I = round(N_max**3/36 + 7*N_max**2/24 + 11*N_max/12 + 119/144
          + (-1)**N_max/16 + 2/9/np.sqrt(3)*math.sin(np.pi/3*(2*N_max+1)))
print(I)
nk = np.zeros((N_max+1,I), dtype=int)
for K in range(N_max+1):
    i = 0
    for nx in range(N_max+1):
        for ny in range(N_max+1):
            for nz in range(N_max+1):

                if nx+ny+nz <= N_max and nx <= ny and ny <= nz:
                    nk[K,i] = count_tcoeffs(nx, ny, nz, K)
                    i += 1


                    

print(nk)
print(np.sum(nk))
