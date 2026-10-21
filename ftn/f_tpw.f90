module mod_ftpw

  use, intrinsic :: iso_c_binding
  implicit none

contains

  !=======================================================
  subroutine tpw_get_symm_x(n, s_no, max_order, tol, ocheck, x)

    ! s_no:
    !   On input, the overlap matrix of the nonorthogonal vectors. On output, the normalized overlap
    !   version of itself on input.
    ! tol:
    !   Tolerance for terminating the approximation series of X and also for checking the orthogonality
    !   when ocheck=.true.
    use mod_fnumc, only : czero
    use mod_ftools, only : tools_binom, tools_mat_series, tools_check_ortho
    implicit none
    integer(c_int), intent(in) :: n
    complex(c_double_complex), intent(inout) :: s_no(n,n)   
    integer(c_int), intent(in) :: max_order
    real(c_double), intent(in) :: tol     
    logical(c_bool), intent(in) :: ocheck
    complex(c_double_complex), intent(out) :: x(n,n)

    logical(c_bool) :: isconv, isortho
    integer(c_int) :: i, j, used_order
    real(c_double) :: max_err
    real(c_double), allocatable :: norm(:), cf(:)
    complex(c_double_complex), allocatable :: s_diag(:)
    

    !==== Get the norms ====!
    allocate(norm(n), source=0.0d0)
    do i = 1, n
       norm(i) = sqrt( real(s_no(i,i)) )
    end do

    !==== Initiate E=S-I ====!
    if (ocheck) allocate(s_diag(n), source=czero)
    do j = 1, n
       do i = 1, j
          if (i /= j) then
             s_no(i,j) = s_no(i,j) / norm(i) / norm(j)
             s_no(j,i) = conjg(s_no(i,j))
          else
             if (ocheck) s_diag(i) = s_no(i,i) / norm(i)**2
             !^^ store the diag elements of s_no because it will be reintroduced again later
             ! if orthogonality check is requested.
             s_no(i,i) = czero
          end if
       end do
    end do
    !ATTENTION: From this point on, the off-diag elements of s_no contains normalized
    !overlaps, while its diag elements are zeros. Therefore, s_no stores E=S-I.
    deallocate(norm)
    write(*, '("check sum s_no = ", es18.10)') sum(abs(s_no))
    call flush()

    
    !==== Calculate X ====!
    allocate(cf(0:max_order), source=0.0d0)
    call tools_binom(max_order, cf)
    call tools_mat_series(n, max_order, tol, s_no, cf, x, used_order, isconv)
    deallocate(cf)

    if (.not. isconv) then
       write(*, '("WARNING: Convergence not reached when computing the transformation '// &
            'matrix for the symmetric orthogonalization. Consider increasing max_order '// &
            'and/or decreasing tol that are inputs to tools_mat_series.")')
    endif


    !==== Check orthogonality ====!
    if (ocheck) then
       do i = 1, n
          s_no(i,i) = s_diag(i)
       end do
       call tools_check_ortho(n, n, x, s_no, tol, isortho, max_err)
       if (ocheck) deallocate(s_diag)
       if (.not. isortho) then
          write(*, '("Maximum error = ", es11.3, " is larger than tol = ", es11.3)') max_err, tol
          stop 'ERROR: Symmetric orthogonalization does not produce sufficiently '// &
               'orthonormal vectors. Consider increasing max_order and/or decreasing tol '// &
               'that are inputs to tools_mat_series.'
       end if
    end if
    
  end subroutine tpw_get_symm_x
  !=======================================================


  !=======================================================
  subroutine tpw_orthogonalize(ortho, nk, nmo, e_proj, s_tpw, max_order, tol, ocheck, &
       northo, d_ortho, e_ortho) bind(c, name='ftpw_orthogonalize')

    use mod_fnumc, only : czero, cone
    use mod_ftools, only : tools_zgemm, tools_zheev
    implicit none
    integer(c_int), intent(in) :: ortho         ! 1 or 2.
    integer(c_int), intent(in) :: nk
    integer(c_int), intent(in) :: nmo
    complex(c_double_complex), intent(in) :: e_proj(nmo,nk)
    complex(c_double_complex), intent(inout) :: s_tpw(nk,nk)
    integer(c_int), intent(in) :: max_order
    real(c_double), intent(in) :: tol
    logical(c_bool), intent(in) :: ocheck
    integer(c_int), intent(out) :: northo
    complex(c_double_complex), intent(out) :: d_ortho(nk,nk)
    complex(c_double_complex), intent(out) :: e_ortho(nmo,nk)

    integer(c_int) :: i, j
    real(c_double), allocatable :: s(:)

    
    ! Here, s_tpw = overlap matrix between tPWs.
    call tools_zgemm('c', 'n', nk, nmo, nk, -cone, -e_proj, -e_proj, cone, s_tpw)
    ! After the call above, s_tpw = overlap matrix between projected tPWs.

    
    !==== Calculate d_ortho = X ====!
    if (ortho == 1) then
       allocate(s(nk), source=0.0d0)
       call tools_zheev(nk, s_tpw, s)
       !^^ After the call above, s_tpw contains the eigenvectors of s_tpw before the call.
       d_ortho = czero
       northo = 0
       !vv Columns of d_ortho correspond to the eigenvectors of s_tpw in DESCENDING order of the eigenvalues.
       do i = nk, 1, -1
          j = i - nk + 1
          if (real(s(i)) >= tol) then
             northo = northo + 1
             d_ortho(:,j) = s_tpw(:,i) / sqrt(s(i))
          end if
       end do
       deallocate(s)
    else if (ortho == 2) then
       northo = nk
       call tpw_get_symm_x(nk, s_tpw, max_order, tol, ocheck, d_ortho)
       !^^ After the call above, s_tpw corresponds to the normalized overlap version of s_tpw before the call.
    else
       stop 'ERROR: Argument ortho of tpw_orthogonalize only accepts 1 or 2.'
    end if

    !==== Calculate e_ortho = e_proj X = e_proj d_ortho ====!
    call tools_zgemm('n', 'n', nmo, nk, northo, cone, e_proj, d_ortho(:,:northo), czero, &
         e_ortho)
    
  end subroutine tpw_orthogonalize
  !=======================================================
  
end module mod_ftpw


