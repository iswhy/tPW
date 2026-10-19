
module mod_numc

  use, intrinsic :: iso_c_binding
  implicit none

  real(c_double), parameter :: pinum = 3.14159265359d0
  complex(c_double_complex), parameter :: czero = cmplx(0.0d0, 0.0d0, c_double_complex)
  complex(c_double_complex), parameter :: cone = cmplx(1.0d0, 0.0d0, c_double_complex)
  complex(c_double_complex), parameter :: imnum = cmplx(0.0d0, 1.0d0, c_double_complex)

end module mod_numc
