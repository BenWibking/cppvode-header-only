! Robertson problem integrated with DVODE (Fortran)
! Logs t, last successful step size (HU), and method order per accepted step.

      PROGRAM ROBERTSON_DVODE
      IMPLICIT NONE

      EXTERNAL FROB, JROB

      INTEGER          NEQ, ITOL, ITASK, ISTATE, IOPT, LRW, LIW, MF
      INTEGER          IWORK(33)
      DOUBLE PRECISION T, TOUT, TEND, RTOL, ATOL, Y(3)
      DOUBLE PRECISION RWORK(67)
      DOUBLE PRECISION RPAR(1)
      INTEGER          IPAR(1)

      INTEGER          NSTEPS, NQU
      DOUBLE PRECISION HU

      NEQ = 3

      Y(1) = 1.0D0
      Y(2) = 0.0D0
      Y(3) = 0.0D0

      T    = 0.0D0
      TEND = 40.0D0
      TOUT = TEND

      ITOL = 1
      RTOL = 1.0D-6
      ATOL = 1.0D-10
      ITASK = 2            ! take one step only and return
      ISTATE = 1           ! first call
      IOPT = 0
      LRW = 67
      LIW = 33
      MF = 21              ! BDF, user-supplied full Jacobian

      WRITE(6, '(A)') 'DVODE Robertson per-step log (t, hu, order)'
      WRITE(6, '(A)') '================================================'

 10   CONTINUE
         CALL DVODE(FROB, NEQ, Y, T, TOUT, ITOL, RTOL, ATOL, ITASK, ISTATE, IOPT, RWORK, LRW, IWORK, LIW, JROB, MF, RPAR, IPAR)

         IF (ISTATE .LT. 0) THEN
            WRITE(6,'(A,I4)') 'DVODE error, ISTATE = ', ISTATE
            GOTO 99
         ENDIF

         HU   = RWORK(11)
         NQU  = IWORK(14)
         NSTEPS = IWORK(11)
         WRITE(6,'(A,1PD24.17,1X,A,1PD24.17,1X,A,I0)') 't = ', T, 'hu = ', HU, 'nq = ', NQU

         ! No artificial cap; run until TEND
         IF (T .GE. TEND) GOTO 20
         ! Continue stepping
         ISTATE = 2
         GOTO 10

 20   CONTINUE
      WRITE(6,'(/,A)') 'Final state at t=40:'
      WRITE(6,'(A,3(1X,1PE16.8))') 'y =', Y(1), Y(2), Y(3)
      WRITE(6,'(A,I0)') 'Total steps: ', IWORK(11)
      WRITE(6,'(A,I0)') 'Function evals: ', IWORK(12)
      WRITE(6,'(A,I0)') 'Jacobian evals: ', IWORK(13)

 99   CONTINUE
      END PROGRAM ROBERTSON_DVODE

      SUBROUTINE FROB(NEQ, T, Y, YDOT, RPAR, IPAR)
      IMPLICIT NONE
      INTEGER NEQ
      DOUBLE PRECISION T, Y(NEQ), YDOT(NEQ), RPAR(*)
      INTEGER IPAR(*)
      DOUBLE PRECISION K1, K2, K3
      K1 = 4.0D-2
      K2 = 1.0D4
      K3 = 3.0D7
      YDOT(1) = -K1*Y(1) + K2*Y(2)*Y(3)
      YDOT(2) =  K1*Y(1) - K2*Y(2)*Y(3) - K3*Y(2)*Y(2)
      YDOT(3) =  K3*Y(2)*Y(2)
      RETURN
      END

      SUBROUTINE JROB(NEQ, T, Y, ML, MU, PD, NROWPD, RPAR, IPAR)
      IMPLICIT NONE
      INTEGER NEQ, ML, MU, NROWPD
      DOUBLE PRECISION T, Y(NEQ), PD(NROWPD,NEQ), RPAR(*)
      INTEGER IPAR(*)
      DOUBLE PRECISION K1, K2, K3
      K1 = 4.0D-2
      K2 = 1.0D4
      K3 = 3.0D7
      PD(1,1) = -K1
      PD(1,2) =  K2*Y(3)
      PD(1,3) =  K2*Y(2)
      PD(2,1) =  K1
      PD(2,2) = -K2*Y(3) - 2.0D0*K3*Y(2)
      PD(2,3) = -K2*Y(2)
      PD(3,1) =  0.0D0
      PD(3,2) =  2.0D0*K3*Y(2)
      PD(3,3) =  0.0D0
      RETURN
      END
