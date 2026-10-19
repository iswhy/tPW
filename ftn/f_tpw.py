import ctypes as ct
import numpy as np
from os.path import dirname, abspath

thisdir = dirname(abspath(__file__))
fortlib = ct.CDLL(thisdir + '/../lib/libftpw.so')


#============================================================
def FTPW_orthogonalize(ortho, nk, nmo, e_proj, s_tpw, max_order, tol, ocheck):


    #==== Set argument types ====#
    fortlib.ftpw_orthogonalize.argtypes = \
        [
            ct.POINTER(ct.c_int)
            ct.POINTER(ct.c_int)
            ct.POINTER(ct.c_int)
            ct.POINTER(ct.c_double_complex)
            ct.POINTER(ct.c_double_complex)
            ct.POINTER(ct.c_int)
            ct.POINTER(ct.c_double)
            ct.POINTER(ct.c_bool)
            ct.POINTER(ct.c_int)
            ct.POINTER(ct.c_double_complex)
            ct.POINTER(ct.c_double_complex)
        ]

    #==== Prepare array inputs ====#
    e_proj = e_proj.astype(ct.c_double_complex, order="F")
    e_proj_ptr = e_proj.ctypes.data_as(ct.POINTER(ct.c_double_complex))

    s_tpw = s_tpw.astype(ct.c_double_complex, order="F")
    s_tpw_ptr = s_tpw.ctypes.data_as(ct.POINTER(ct.c_double_complex))

    #==== Prepare scalar outputs ====#
    northo = ct.c_int(0)
    
    #==== Prepare array outputs ====#
    d_ortho = np.zeros((nk,nk), dtype=complex)
    d_ortho = d_ortho.astype(ct.c_double_complex, order="F")
    d_ortho_ptr = d_ortho.ctypes.data_as(ct.POINTER(ct.c_double_complex))

    e_ortho = np.zeros((nmo,nk), dtype=complex)
    e_ortho = e_ortho.astype(ct.c_double_complex, order="F")
    e_ortho_ptr = e_ortho.ctypes.data_as(ct.POINTER(ct.c_double_complex))    
    
    #==== Call Fortran subroutine ====#
    fortlib.ftpw_orthogonalize(ct.byref(ct.c_int(ortho)),
                               ct.byref(ct.c_int(nk)),
                               ct.byref(ct.c_int(nmo)),
                               e_proj_ptr,
                               s_tpw_ptr,
                               ct.byref(ct.c_int(max_order)),
                               ct.byref(ct.c_double(tol)),
                               ct.byref(ct.c_bool(ocheck)),
                               ct.byref(northo),
                               d_ortho_ptr,
                               e_ortho_ptr)

    return northo.value, d_ortho, e_ortho
#============================================================
