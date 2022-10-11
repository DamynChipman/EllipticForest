C
C     file fish.f
C
C
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C     *                                                               *
C     *                  copyright (c) 2005 by UCAR                   *
C     *                                                               *
C     *       University Corporation for Atmospheric Research         *
C     *                                                               *
C     *                      all rights reserved                      *
C     *                                                               *
C     *                    FISHPACK90  version 1.1                    *
C     *                                                               *
C     *                 A Package of Fortran 77 and 90                *
C     *                                                               *
C     *                Subroutines and Example Programs               *
C     *                                                               *
C     *               for Modeling Geophysical Processes              *
C     *                                                               *
C     *                             by                                *
C     *                                                               *
C     *        John Adams, Paul Swarztrauber and Roland Sweet         *
C     *                                                               *
C     *                             of                                *
C     *                                                               *
C     *         the National Center for Atmospheric Research          *
C     *                                                               *
C     *                Boulder, Colorado  (80307)  U.S.A.             *
C     *                                                               *
C     *                   which is sponsored by                       *
C     *                                                               *
C     *              the National Science Foundation                  *
C     *                                                               *
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C

!     this module is used by all fishpack solvers to allocate
!     real and complex work space
      MODULE fish
	TYPE fishworkspace
	  REAL,POINTER,DIMENSION(:) :: rew
	  COMPLEX,POINTER,DIMENSION(:) :: cxw
	END TYPE fishworkspace
	CONTAINS
	SUBROUTINE allocatfish(irwk,icwk,wsave,ierror)
	IMPLICIT NONE
	TYPE (fishworkspace) :: wsave
!       irwk is the required real work space length
!       icwk is the required integer work space length
	INTEGER, INTENT(IN) :: irwk,icwk
!       ierror is set to 20 if the dynamic allocation is unsuccessful
!       (e.g., this would happen if m,n are too large for the computers memory
	INTEGER, INTENT(INOUT) :: ierror
	INTEGER :: istatus
!       first deallocate to avoid memory leakage
#ifdef __DEALLOC__
 	if(associated(wsave%rew))DEALLOCATE(wsave%rew)
 	if(associated(wsave%cxw))DEALLOCATE(wsave%cxw)
#endif
!       allocate irwk words of real work space
	if (irwk > 0) then
	     ALLOCATE(wsave%rew(irwk),STAT = istatus)
	end if
!       allocate icwk words of complex work space
	if (icwk > 0) then
	     ALLOCATE(wsave%cxw(icwk),STAT = istatus)
	end if
	ierror = 0
!       flag fatal error if allocation fails
c       IF (istatus /= 0) THEN
	if (istatus .ne. 0 ) then
	  ierror = 20
	END IF
	RETURN
	END SUBROUTINE allocatfish

	SUBROUTINE BLK_space(N,M,irwk,icwk)
!       this subroutine computes the real and complex work space
!       requirements (generous estimate) of blktri for N,M values
	IMPLICIT NONE
	INTEGER,INTENT(IN) :: N,M
	INTEGER,INTENT(OUT) :: irwk,icwk
	INTEGER :: L,log2n
!       compute nearest integer greater than or equal to
!       log base 2 of n+1, i.e., log2n is smallest integer
!       such that 2**log2n >= n+1
	log2n = 1
	do
	   log2n = log2n+1
	   if (n+1 <= 2**log2n) EXIT
	end do
	L = 2**(log2n+1)
	irwk = (log2n-2)*L+5+MAX0(2*N,6*M)+log2n+2*n
	icwk = ((log2n-2)*L+5+log2n)/2+3*M+N
	RETURN
	END SUBROUTINE BLK_space

	SUBROUTINE GEN_space(N,M,irwk)
!       this subroutine computes the real work space
!       requirement (generously) of genbun for the current N,M
	IMPLICIT NONE
	INTEGER,INTENT(IN) :: N,M
	INTEGER,INTENT(OUT) :: irwk
	INTEGER :: log2n
!       compute nearest integer greater than or equal to
!       log base 2 of n+1, i.e., log2n is smallest integer
!       such that 2**log2n >= n+1
	log2n = 1
	do
	   log2n = log2n+1
	   if (n+1 <= 2**log2n) EXIT
	end do
	irwk = 4*N + (10 + log2n)*M
	RETURN
	END SUBROUTINE GEN_space

	SUBROUTINE fishfin(wsave)
!       this subroutine releases allocated work space
!       fishfin should be called after a fishpack solver has finished
!       TYPE (fishworkspace) variable wsave.
	IMPLICIT NONE
	TYPE (fishworkspace) :: wsave
	INTEGER :: istatus
#ifdef __DEALLOC__
 	if(associated(wsave%rew))DEALLOCATE(wsave%rew)
 	if(associated(wsave%cxw))DEALLOCATE(wsave%cxw)
#endif
	RETURN
	END SUBROUTINE fishfin

      END MODULE fish
