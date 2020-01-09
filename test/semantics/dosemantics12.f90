!Section 11.1.7.4.3, paragraph 2 states:
!  Except for the incrementation of the DO variable that occurs in step (3), 
!  the DO variable shall neither be redefined nor become undefined while the
!  DO construct is active.

subroutine s1()

  ! Redefinition via intrinsic assignment (section 19.6.5, case (1))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    ivar = 99
  end do

  ! Redefinition in the presence of a construct association
  associate (avar => ivar)
    do ivar = 1,20
      print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
      avar = 99
    end do
  end associate

  ivar = 99

  ! Redefinition via intrinsic assignment (section 19.6.5, case (1))
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    ivar = 99
  end do

  ivar = 99

end subroutine s1

subroutine s2()

  integer :: ivar

  read '(I10)', ivar

  ! Redefinition via an input statement (section 19.6.5, case (3))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    read '(I10)', ivar
  end do

  ! Redefinition via an input statement (section 19.6.5, case (3))
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    read '(I10)', ivar
  end do

end subroutine s2

subroutine s3()

  integer :: ivar

  ! Redefinition via use as a DO variable (section 19.6.5, case (4))
  do ivar = 1,10
!ERROR: Cannot redefine DO variable 'ivar'
    do ivar = 1,20
!ERROR: Cannot redefine DO variable 'ivar'
      do ivar = 1,30
        print *, "hello"
      end do
    end do
  end do

  ! This one's OK, even though we used ivar previously as a DO variable
  ! since it's not a redefinition
  do ivar = 1,40
    print *, "hello"
  end do

  ! Redefinition via use as a DO variable (section 19.6.5, case (4))
  do concurrent (ivar = 1:10)
!ERROR: Cannot redefine DO variable 'ivar'
    do ivar = 1,20
      print *, "hello"
    end do
  end do

end subroutine s3

subroutine s4()

  integer :: ivar
  real :: x(10)

  print '(f10.5)', (x(ivar), ivar = 1, 10)

  ! Redefinition via use as a DO variable (section 19.6.5, case (5))
  do ivar = 1,20
!ERROR: Cannot redefine DO variable 'ivar'
    print '(f10.5)', (x(ivar), ivar = 1, 10)
  end do

  ! Redefinition via use as a DO variable (section 19.6.5, case (5))
  do concurrent (ivar = 1:10)
!ERROR: Cannot redefine DO variable 'ivar'
    print '(f10.5)', (x(ivar), ivar = 1, 10)
  end do

end subroutine s4

subroutine s5()

  integer :: ivar
  real :: x

  read (3, '(f10.5)', iostat = ivar) x

  ! Redefinition via use in IOSTAT specifier (section 19.6.5, case (7))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    read (3, '(f10.5)', iostat = ivar) x
  end do

  ! Redefinition via use in IOSTAT specifier (section 19.6.5, case (7))
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    read (3, '(f10.5)', iostat = ivar) x
  end do

end subroutine s5

subroutine s6()

  character (len=3) :: key
  integer :: chars
  integer :: ivar
  real :: x

  read (3, '(a3)', advance='no', size = chars) key

  ! Redefinition via use in SIZE specifier (section 19.6.5, case (9))
  do ivar = 1,20
!ERROR: Cannot redefine DO variable 'ivar'
    read (3, '(a3)', advance='no', size = ivar) key
    print *, "hello"
  end do

  ! Redefinition via use in SIZE specifier (section 19.6.5, case (9))
  do concurrent (ivar = 1:10)
!ERROR: ADVANCE specifier is not allowed in DO CONCURRENT
!ERROR: Cannot redefine DO variable 'ivar'
    read (3, '(a3)', advance='no', size = ivar) key
    print *, "hello"
  end do

end subroutine s6

subroutine s7()

  integer :: iostatVar, nextrecVar, numberVar, posVar, reclVar, sizeVar

  inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
    pos=posVar, recl=reclVar, size=sizeVar)

  ! Redefinition via use in IOSTAT specifier (section 19.6.5, case (10))
  do iostatVar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'iostatvar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in IOSTAT specifier (section 19.6.5, case (10))
  do concurrent (iostatVar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'iostatvar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in NEXTREC specifier (section 19.6.5, case (10))
  do nextrecVar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'nextrecvar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in NEXTREC specifier (section 19.6.5, case (10))
  do concurrent (nextrecVar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'nextrecvar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in NUMBER specifier (section 19.6.5, case (10))
  do numberVar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'numbervar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in NUMBER specifier (section 19.6.5, case (10))
  do concurrent (numberVar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'numbervar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in RECL specifier (section 19.6.5, case (10))
  do reclVar = 1,20
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'reclvar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in RECL specifier (section 19.6.5, case (10))
  do concurrent (reclVar = 1:10)
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'reclvar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in POS specifier (section 19.6.5, case (10))
  do posVar = 1,20
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'posvar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in POS specifier (section 19.6.5, case (10))
  do concurrent (posVar = 1:10)
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'posvar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in SIZE specifier (section 19.6.5, case (10))
  do sizeVar = 1,20
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'sizevar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in SIZE specifier (section 19.6.5, case (10))
  do concurrent (sizeVar = 1:10)
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'sizevar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

end subroutine s7

subroutine s8()

  Integer :: ivar
  integer, pointer :: ip

  allocate(ip, stat = ivar)

  ! Redefinition via a STAT= specifier (section 19.6.5, case (16))
  do ivar = 1,20
!ERROR: Cannot redefine DO variable 'ivar'
    allocate(ip, stat = ivar)
    print *, "hello"
  end do

  ! Redefinition via a STAT= specifier (section 19.6.5, case (16))
  do concurrent (ivar = 1:10)
!ERROR: Cannot redefine DO variable 'ivar'
    allocate(ip, stat = ivar)
    print *, "hello"
  end do

end subroutine s8

subroutine s9()

  Integer :: ivar

  ! OK since the DO CONCURRENT index-name exists only in the scope of the
  ! DO CONCURRENT construct
  do ivar = 1,20
    print *, "hello"
    do concurrent (ivar = 1:10)
      print *, "hello"
    end do
  end do

  ! OK since the DO CONCURRENT index-name exists only in the scope of the
  ! DO CONCURRENT construct
  do concurrent (ivar = 1:10)
    print *, "hello"
    do concurrent (ivar = 1:10)
      print *, "hello"
    end do
  end do

end subroutine s9

subroutine s10()

  Integer :: ivar
  open(file="abc", newunit=ivar)

  ! Redefinition via NEWUNIT specifier (section 19.6.5, case (29))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    open(file="abc", newunit=ivar)
  end do

  ! Redefinition via NEWUNIT specifier (section 19.6.5, case (29))
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    open(file="abc", newunit=ivar)
  end do

end subroutine s10

subroutine s11()

  Integer, allocatable :: ivar

  allocate(ivar)

  ! This look is OK
  do ivar = 1,20
    print *, "hello"
  end do

  ! Redefinition via deallocation (section 19.6.6, case (10))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    deallocate(ivar)
  end do

  ! This case is not applicable since the version of "ivar" that's inside the
  ! DO CONCURRENT has the scope of the DO CONCURRENT construct.  Within that
  ! scope, it does not have the "allocatable" attribute, so the following test
  ! fails because you can only deallocate a variable that's allocatable.
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    deallocate(ivar)
  end do

end subroutine s11
