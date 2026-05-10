! Robertson problem integrated with the original Fortran RODAS code.
! The final state is compared against the C++ header-only RODAS example
! built from examples/robertson_rodas.cpp with the same settings.

      PROGRAM ROBERTSON_RODAS_FORTRAN
      IMPLICIT NONE

      EXTERNAL FROB, JROB, DFXDUM, MASDUM, SOLOUTDUM

      INTEGER, PARAMETER :: NEQ = 3
      INTEGER, PARAMETER :: LWORK = 2*NEQ*NEQ + 14*NEQ + 20
      INTEGER, PARAMETER :: LIWORK = NEQ + 20

      INTEGER          IFCN, ITOL, IJAC, MLJAC, MUJAC, IDFX
      INTEGER          IMAS, MLMAS, MUMAS, IOUT, IDID
      INTEGER          IWORK(LIWORK), IPAR(1)
      DOUBLE PRECISION X, XEND, H, RTOL(1), ATOL(1)
      DOUBLE PRECISION Y(NEQ), WORK(LWORK), RPAR(1)
      DOUBLE PRECISION YCPP(NEQ), DIFF(NEQ), TOTAL, MAXDIFF
      INTEGER          I

      Y(1) = 1.0D0
      Y(2) = 0.0D0
      Y(3) = 0.0D0

      X    = 0.0D0
      XEND = 40.0D0
      H    = 1.0D-6

      RTOL(1) = 1.0D-6
      ATOL(1) = 1.0D-10

      IFCN  = 0
      ITOL  = 0
      IJAC  = 1
      MLJAC = NEQ
      MUJAC = 0
      IDFX  = 0
      IMAS  = 0
      MLMAS = 0
      MUMAS = 0
      IOUT  = 0

      DO I = 1, LWORK
         WORK(I) = 0.0D0
      END DO
      DO I = 1, LIWORK
         IWORK(I) = 0
      END DO
      RPAR(1) = 0.0D0
      IPAR(1) = 0

      CALL RODAS(NEQ,FROB,IFCN,X,Y,XEND,H, &
     &           RTOL,ATOL,ITOL, &
     &           JROB,IJAC,MLJAC,MUJAC,DFXDUM,IDFX, &
     &           MASDUM,IMAS,MLMAS,MUMAS, &
     &           SOLOUTDUM,IOUT, &
     &           WORK,LWORK,IWORK,LIWORK,RPAR,IPAR,IDID)

      IF (IDID .LT. 0) THEN
         WRITE(6,'(A,I4)') 'RODAS error, IDID = ', IDID
         STOP 1
      ENDIF

      ! Reference from the C++ RODAS implementation with t=[0,40],
      ! h0=1e-6, rtol=1e-6, atol=1e-10, analytic full Jacobian.
      YCPP(1) = 0.71582703916834622D0
      YCPP(2) = 9.1855334998520728D-06
      YCPP(3) = 0.28416377529815512D0

      MAXDIFF = 0.0D0
      DO I = 1, NEQ
         DIFF(I) = ABS(Y(I) - YCPP(I))
         MAXDIFF = MAX(MAXDIFF, DIFF(I))
      END DO
      TOTAL = Y(1) + Y(2) + Y(3)

      WRITE(6,'(A)') 'Fortran RODAS Robertson final state at t=40'
      WRITE(6,'(A,3(1X,1PE24.16))') 'y =', Y(1), Y(2), Y(3)
      WRITE(6,'(A,1PE12.4)') 'conservation_error = ', ABS(TOTAL - 1.0D0)
      WRITE(6,'(A,I0)') 'steps = ', IWORK(16)
      WRITE(6,'(A,I0)') 'accepted = ', IWORK(17)
      WRITE(6,'(A,I0)') 'rejected = ', IWORK(18)
      WRITE(6,'(A,I0)') 'rhs_evals = ', IWORK(14)
      WRITE(6,'(A,I0)') 'jacobian_evals = ', IWORK(15)
      WRITE(6,'(/,A)') 'Comparison to C++ RODAS final state'
      WRITE(6,'(A)') 'component      Fortran RODAS            C++ RODAS               abs diff'
      DO I = 1, NEQ
         WRITE(6,'(I4,3(3X,1PE22.14))') I, Y(I), YCPP(I), DIFF(I)
      END DO
      WRITE(6,'(A,1PE12.4)') 'max_abs_diff = ', MAXDIFF

      IF (MAXDIFF .GT. 5.0D-5) THEN
         STOP 1
      END IF

      END PROGRAM ROBERTSON_RODAS_FORTRAN

      SUBROUTINE FROB(N, X, Y, F, RPAR, IPAR)
      IMPLICIT NONE
      INTEGER N
      DOUBLE PRECISION X, Y(N), F(N), RPAR(*)
      INTEGER IPAR(*)
      DOUBLE PRECISION K1, K2, K3
      K1 = 4.0D-2
      K2 = 1.0D4
      K3 = 3.0D7
      F(1) = -K1*Y(1) + K2*Y(2)*Y(3)
      F(2) =  K1*Y(1) - K2*Y(2)*Y(3) - K3*Y(2)*Y(2)
      F(3) =  K3*Y(2)*Y(2)
      RETURN
      END

      SUBROUTINE JROB(N, X, Y, DFY, LDFY, RPAR, IPAR)
      IMPLICIT NONE
      INTEGER N, LDFY
      DOUBLE PRECISION X, Y(N), DFY(LDFY,N), RPAR(*)
      INTEGER IPAR(*)
      DOUBLE PRECISION K1, K2, K3
      K1 = 4.0D-2
      K2 = 1.0D4
      K3 = 3.0D7
      DFY(1,1) = -K1
      DFY(1,2) =  K2*Y(3)
      DFY(1,3) =  K2*Y(2)
      DFY(2,1) =  K1
      DFY(2,2) = -K2*Y(3) - 2.0D0*K3*Y(2)
      DFY(2,3) = -K2*Y(2)
      DFY(3,1) =  0.0D0
      DFY(3,2) =  2.0D0*K3*Y(2)
      DFY(3,3) =  0.0D0
      RETURN
      END

      SUBROUTINE DFXDUM(N, X, Y, FX, RPAR, IPAR)
      IMPLICIT NONE
      INTEGER N
      DOUBLE PRECISION X, Y(N), FX(N), RPAR(*)
      INTEGER IPAR(*)
      RETURN
      END

      SUBROUTINE MASDUM(N, AM, LMAS, RPAR, IPAR)
      IMPLICIT NONE
      INTEGER N, LMAS
      DOUBLE PRECISION AM(LMAS,N), RPAR(*)
      INTEGER IPAR(*)
      RETURN
      END

      SUBROUTINE SOLOUTDUM(NR, XOLD, X, Y, CONT, LRC, N, &
     &                     RPAR, IPAR, IRTRN)
      IMPLICIT NONE
      INTEGER NR, LRC, N, IRTRN
      DOUBLE PRECISION XOLD, X, Y(N), CONT(LRC), RPAR(*)
      INTEGER IPAR(*)
      RETURN
      END

      SUBROUTINE ZGETRF(M, N, A, LDA, IPIV, INFO)
      IMPLICIT NONE
      INTEGER M, N, LDA, IPIV(*), INFO
      COMPLEX*16 A(LDA,*)
      INFO = -1
      RETURN
      END

      SUBROUTINE ZGBTRF(M, N, KL, KU, AB, LDAB, IPIV, INFO)
      IMPLICIT NONE
      INTEGER M, N, KL, KU, LDAB, IPIV(*), INFO
      COMPLEX*16 AB(LDAB,*)
      INFO = -1
      RETURN
      END

      SUBROUTINE ZGETRS(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO)
      IMPLICIT NONE
      CHARACTER TRANS
      INTEGER N, NRHS, LDA, LDB, IPIV(*), INFO
      COMPLEX*16 A(LDA,*), B(LDB,*)
      INFO = -1
      RETURN
      END

      SUBROUTINE ZGBTRS(TRANS, N, KL, KU, NRHS, AB, LDAB, IPIV, B, LDB, INFO)
      IMPLICIT NONE
      CHARACTER TRANS
      INTEGER N, KL, KU, NRHS, LDAB, LDB, IPIV(*), INFO
      COMPLEX*16 AB(LDAB,*), B(LDB,*)
      INFO = -1
      RETURN
      END
