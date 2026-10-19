module mod_tpw

  use, intrinsic :: iso_c_binding
  implicit none

contains

  !=======================================================
  subroutine tools_zgemm(ta, tb, m, k, n, alpha, a, b, beta, c)

    ! DESCRIPTION:
    !   C = alpha*op( A )*op( B ) + beta*C,
    ! where  op( X ) is one of
    !   op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H.
    implicit none
    character*1, intent(in) :: ta        ! 'n', 't', or 'c'.
    character*1, intent(in) :: tb        ! 'n', 't', or 'c'.
    integer(c_int), intent(in) :: m      ! the number of rows of op(A) and of C
    integer(c_int), intent(in) :: k      ! the number of columns of op(A) and the number of rows of op(B)
    integer(c_int), intent(in) :: n      ! the number of columns of op(B) and of C
    complex(c_double_complex), intent(in) :: alpha
    complex(c_double_complex), intent(in) :: a(*)
    complex(c_double_complex), intent(in) :: b(*)
    complex(c_double_complex), intent(in) :: beta
    complex(c_double_complex), intent(inout) :: c(*)

    integer(c_int) :: lda, ldb, ldc

    lda = k
    if (ta == 'n') lda = m
    ldb = n
    if (tb == 'n') ldb = k
    ldc = m
    
    call zgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    
  end subroutine tools_zgemm
  !=======================================================


  !=======================================================
  subroutine tools_dsyev(n, a, w)
    !DESCRIPTION:
    ! A wrapper to LAPACK's dsyev routine.
    implicit none
    integer(c_int), intent(in) :: n                 ! [input] order of matrix a 
    real(c_double), intent(inout) :: a(n,n)   ! [in/output] On input, the matrix to be diagonalized. On output, the eigenvectors arranged in columns.
    real(c_double), intent(out) :: w(n)     ! [output] eigenvalues

    integer(c_int) :: lwork, info
    real(c_double) :: work0(1)
    real(c_double), allocatable :: work(:)
    

    !==== Workspace query ====      
    call dsyev('v', 'u', n, a, n, w, work0, -1, info)
    lwork = int(work0(1))

    !==== Do the diagonalization ====
    allocate(work(lwork), source=0.0d0)
    call dsyev('v', 'u', n, a, n, w, work, lwork, info)
    deallocate(work)

    if (info /= 0) stop 'Error when solving for the solution using tools_dsyev.'
    
  end subroutine tools_dsyev
  !=======================================================


  !=======================================================
  subroutine tools_zheev(n, a, w)
    !DESCRIPTION:
    ! A wrapper to LAPACK's dsyev routine.
    use mod_numc, only : czero
    implicit none
    integer(c_int), intent(in) :: n                 ! [input] order of matrix a 
    complex(c_double_complex), intent(inout) :: a(n,n)   ! [in/output] On input, the matrix to be diagonalized. On output, the eigenvectors arranged in columns.
    real(c_double), intent(out) :: w(n)     ! [output] eigenvalues

    integer(c_int) :: lda, lwork, info
    complex(c_double_complex) :: work0(1)
    complex(c_double_complex), allocatable :: work(:)
    real(c_double), allocatable :: rwork(:)
    

    if (n <= 0) stop 'tools_zheev: n must be larger than or equal to 1.'
    lda = n
    allocate(rwork(3*n-2), source=0.0d0)
    
    !==== Workspace query ====      
    call zheev('v', 'u', n, a, lda, w, work0, -1, rwork, info)
    lwork = int(work0(1))

    !==== Do the diagonalization ====
    allocate(work(lwork), source=czero)
    call zheev('v', 'u', n, a, lda, w, work, lwork, rwork, info)
    deallocate(work, rwork)

    if (info /= 0) stop 'Error when solving for the solution using tools_dsyev.'
    
  end subroutine tools_zheev
  !=======================================================
  

  !=======================================================
  subroutine tools_mat_polnom(n, order, E, c, X)
    ! Calculates the polynomials of a matrix using Horner's scheme.
    use mod_numc, only : czero, cone
    implicit none
    integer(c_int), intent(in) :: n, order
    complex(c_double_complex), intent(in)  :: E(n,n)
    complex(c_double_complex), intent(in)  :: c(0:order)
    complex(c_double_complex), intent(out) :: X(n,n)

    complex(c_double_complex), allocatable :: temp(:,:)
    integer(c_int) :: i, k


    stop 'Test tools_mat_polnom before using it in your code.'
    
    allocate(temp(n,n))

    ! Initialize X = c_order * I
    X = czero
    do i = 1, n
       X(i,i) = c(order)
    end do

    ! Horner's scheme
    do k = order-1, 0, -1
       ! temp = E * X
       call tools_zgemm('n', 'n', n, n, n, cone, E, X, czero, temp)
       X = temp

       do i = 1, n
          X(i,i) = X(i,i) + c(k)
       end do
    end do

    deallocate(temp)

  end subroutine tools_mat_polnom
  !=======================================================


  !=======================================================
  subroutine tools_mat_series(n, max_order, tol, E, c, X, used_order, isconv)

    ! Calculates X = c_0 + c_1 E + c_2 E^2 + ...
    ! The series terminates whenever one of these conditions is met first:
    !   1) The frobenius norm of c_n E^n is less than tol,
    !   2) n is equal to max_order.
    use mod_numc, only : czero, cone
    implicit none
    integer(c_int), intent(in) :: n
    integer(c_int), intent(in) :: max_order
    real(c_double), intent(in) :: tol
    complex(c_double_complex), intent(in) :: E(n,n)
    complex(c_double_complex), intent(in) :: c(0:max_order)
    complex(c_double_complex), intent(out) :: X(n,n)
    integer(c_int), intent(out) :: used_order
    logical(c_bool), intent(out) :: isconv

    complex(c_double_complex), allocatable, target :: P(:,:), Pnext(:,:)
    complex(c_double_complex), pointer :: cur(:,:), nxt(:,:), tmp(:,:)
    real(c_double) :: err_series
    real(c_double), external :: dznrm2
    integer(c_int) :: i, k


    allocate(P(n,n), Pnext(n,n))
    cur => P
    nxt => Pnext

    ! Initialize X = I
    X = czero
    do i = 1, n
       X(i,i) = c(0)
    end do

    ! First power: E^1
    cur = E
    used_order = 0

    do k = 1, max_order
       ! Frobenius norm of the current term c(k)*E^k
       err_series = abs(c(k)) * dznrm2(n*n, cur(1,1), 1)
       if (err_series < tol) exit

       ! Accumulate c(k) * E^k into X
       X = X + c(k) * cur
       used_order = k

       if (k == max_order) exit

       ! Calculate E^(k+1) = E^k * E
       call tools_zgemm('n', 'n', n, n, n, cone, cur, E, czero, nxt)

       ! Swap working buffers without copying matrix elements
       tmp => cur
       cur => nxt
       nxt => tmp
    end do

    deallocate(P, Pnext)
    isconv = err_series < tol

  end subroutine tools_mat_series
  !=======================================================


  !=======================================================
  subroutine tools_check_ortho(n, m, A, S, tol, is_ortho, max_err)

    use mod_numc, only : czero, cone
    implicit none
    integer(c_int), intent(in) :: n, m
    complex(c_double_complex), intent(in) :: A(n,m), S(n,n)
    real(c_double), intent(in) :: tol
    logical, intent(out) :: is_ortho
    real(c_double), intent(out) :: max_err

    complex(c_double_complex), allocatable :: SA(:,:), G(:,:)
    complex(c_double_complex) :: one, zero
    real(c_double) :: err
    integer(c_int) :: i, j

    
    allocate(SA(n,m), G(m,m), source=czero)

    ! Calculate SA = S * A
    call tools_zgemm('n', 'n', n, n, m, cone, S, A, czero, SA)

    ! Calculate G = A^H * S * A
    call tools_zgemm('c', 'n', m, n, m, cone, A, SA, czero, G)

    ! Find max absolute elementwise deviation from identity
    max_err = 0.0d0
    do j = 1, m
       do i = 1, m
          if (i == j) then
              err = abs(G(i,j) - 1.0d0)
          else
              err = abs(G(i,j))
          end if
          max_err = max(max_err, err)
       end do
    end do

    ! Check the orthonormality criterion
    is_ortho = (max_err <= tol)

    deallocate(SA, G)
  end subroutine tools_check_ortho
  !=======================================================


  !=======================================================
  subroutine tools_binom(max_order, cf)
    implicit none
    integer(c_int), intent(in) :: max_order
    real(c_double), intent(in) :: cf(0:max_order)

    integer(c_int) :: i

    cf(0) = 1.0d0
    do i = 1, max_order
       cf(i) = -(2*i-1) / (2*i) * cf(i-1)
    end do
    
  end subroutine tools_binom
  !=======================================================


  
end module mod_tpw
