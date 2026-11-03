!=======================================================================
! XPOCK — Fortran 90 Numerical Module
!
! Provides Fortran implementations of core wave-propagation and
! numerical operations used by the wave-structure generator:
!
!   - 1D FFT (via self-contained Cooley-Tukey, power-of-two lengths)
!   - Complex field multiplication
!   - Gaussian beam propagation (ABCD matrix)
!   - Fresnel diffraction integral
!   - Temporal envelope evaluation
!
! Compile with:
!   gfortran -O3 -fPIC -shared -o libxpock_fortran.so wave_ops.f90
!   or
!   gfortran -O3 -c wave_ops.f90 -o wave_ops.o
!
! No external dependencies beyond the Fortran standard library.
!=======================================================================

module xpock_wave_ops
    implicit none
    private
    public :: temporal_gaussian, temporal_sech, temporal_supergaussian
    public :: gaussian_beam_waist, fresnel_propagate
    public :: complex_multiply, magnitude_squared, total_power
    public :: fft1d, ifft1d
    public :: compute_stokes, skyrmion_field_f90, skyrmion_number_f90

    integer, parameter :: dp = selected_real_kind(15, 307)
    real(dp), parameter :: PI = 3.14159265358979323846_dp

contains

!-----------------------------------------------------------------------
! 1D Temporal envelopes
!-----------------------------------------------------------------------

pure real(dp) function temporal_gaussian(t, t0, dur, amp, chirp, omega)
    real(dp), intent(in) :: t
    real(dp), intent(in) :: t0, dur, amp, chirp, omega
    real(dp) :: tau, sigma, env, phase
    tau = t - t0
    sigma = dur / (2.0_dp * sqrt(2.0_dp * log(2.0_dp)))
    env = exp(-0.5_dp * (tau / sigma)**2)
    phase = 0.5_dp * chirp * tau**2 - omega * t
    temporal_gaussian = amp * env * cos(phase)
end function temporal_gaussian

pure real(dp) function temporal_sech(t, t0, dur, amp, chirp, omega)
    real(dp), intent(in) :: t, t0, dur, amp, chirp, omega
    real(dp) :: tau, env, phase
    tau = t - t0
    env = 1.0_dp / cosh(tau / dur)
    phase = 0.5_dp * chirp * tau**2 - omega * t
    temporal_sech = amp * env * cos(phase)
end function temporal_sech

pure real(dp) function temporal_supergaussian(t, t0, dur, amp, order, chirp, omega)
    real(dp), intent(in) :: t, t0, dur, amp
    integer, intent(in) :: order
    real(dp), intent(in) :: chirp, omega
    real(dp) :: tau, env, phase
    tau = t - t0
    env = exp(-((abs(tau) / dur)**(2 * order)))
    phase = 0.5_dp * chirp * tau**2 - omega * t
    temporal_supergaussian = amp * env * cos(phase)
end function temporal_supergaussian

!-----------------------------------------------------------------------
! Gaussian beam parameters
!-----------------------------------------------------------------------

pure real(dp) function gaussian_beam_waist(z, w0, wavelength)
    real(dp), intent(in) :: z, w0, wavelength
    real(dp) :: zR
    zR = PI * w0**2 / wavelength
    gaussian_beam_waist = w0 * sqrt(1.0_dp + (z / zR)**2)
end function gaussian_beam_waist

!-----------------------------------------------------------------------
! Complex array operations
!-----------------------------------------------------------------------

subroutine complex_multiply(a, b, out, n)
    complex(dp), intent(in) :: a(n), b(n)
    complex(dp), intent(out) :: out(n)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
        out(i) = a(i) * b(i)
    end do
end subroutine complex_multiply

subroutine magnitude_squared(a, out, n)
    complex(dp), intent(in) :: a(n)
    real(dp), intent(out) :: out(n)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
        out(i) = real(a(i))**2 + aimag(a(i))**2
    end do
end subroutine magnitude_squared

pure real(dp) function total_power(field, dx, dy, dt, n)
    complex(dp), intent(in) :: field(n)
    real(dp), intent(in) :: dx, dy, dt
    integer, intent(in) :: n
    real(dp) :: s
    integer :: i
    s = 0.0_dp
    do i = 1, n
        s = s + real(field(i))**2 + aimag(field(i))**2
    end do
    total_power = s * dx * dy * dt
end function total_power

!-----------------------------------------------------------------------
! Simple 1D FFT (radix-2 Cooley-Tukey, decimation-in-time)
! Works for power-of-two lengths only.
!-----------------------------------------------------------------------

recursive subroutine fft1d(a, n)
    complex(dp), intent(inout) :: a(n)
    integer, intent(in) :: n
    complex(dp) :: w, wn, t
    integer :: i, j, k, m
    real(dp) :: theta

    if (n == 1) return

    ! Bit-reversal permutation
    j = 1
    do i = 1, n
        if (i < j) then
            t = a(j)
            a(j) = a(i)
            a(i) = t
        end if
        m = n / 2
        do while (m >= 1 .and. j > m)
            j = j - m
            m = m / 2
        end do
        j = j + m
    end do

    ! Butterfly stages
    m = 2
    do while (m <= n)
        wn = cmplx(cos(2.0_dp * PI / m), sin(2.0_dp * PI / m), dp)
        do k = 1, n, m
            w = (1.0_dp, 0.0_dp)
            do j = k, k + m / 2 - 1
                t = w * a(j + m / 2)
                a(j + m / 2) = a(j) - t
                a(j) = a(j) + t
                w = w * wn
            end do
        end do
        m = m * 2
    end do
end subroutine fft1d

recursive subroutine ifft1d(a, n)
    complex(dp), intent(inout) :: a(n)
    integer, intent(in) :: n
    complex(dp) :: w, wn, t
    integer :: i, j, k, m
    real(dp) :: theta

    if (n == 1) return

    ! Conjugate, FFT, conjugate, scale
    do i = 1, n
        a(i) = conjg(a(i))
    end do
    call fft1d(a, n)
    do i = 1, n
        a(i) = conjg(a(i)) / real(n, dp)
    end do
end subroutine ifft1d

!-----------------------------------------------------------------------
! Fresnel diffraction (angular spectrum approximation, 1D)
!-----------------------------------------------------------------------

subroutine fresnel_propagate(u_in, z, wavelength, dx, n, u_out)
    complex(dp), intent(in) :: u_in(n)
    real(dp), intent(in) :: z, wavelength, dx
    integer, intent(in) :: n
    complex(dp), intent(out) :: u_out(n)

    complex(dp) :: U(n), H(n)
    real(dp) :: k, df, f, fx
    integer :: i

    k = 2.0_dp * PI / wavelength
    df = 1.0_dp / (n * dx)

    ! Angular spectrum of input
    do i = 1, n
        U(i) = u_in(i)
    end do
    call fft1d(U, n)

    ! Transfer function H(fx) = exp(i * k * z * sqrt(1 - (lambda*fx)^2))
    ! H is even in fx, so the positive-sign convention of fft1d cancels
    ! between the forward and inverse transforms; what matters is that
    ! each FFT bin k carries its TRUE frequency. fft1d output is
    ! unshifted: bin 1 (1-based) is DC, bins 2..n/2+1 are positive
    ! frequencies (i-1)*df, bins n/2+2..n are negative (i-1-n)*df.
    ! The old formula fx = (i - n/2 - 1) * df wrongly assumed an
    ! fftshifted layout and applied H to the wrong frequency bins.
    do i = 1, n
        if (i <= n / 2 + 1) then
            fx = (i - 1) * df
        else
            fx = (i - 1 - n) * df
        end if
        if (abs(wavelength * fx) <= 1.0_dp) then
            H(i) = cmplx(cos(k * z * sqrt(1.0_dp - (wavelength * fx)**2)),
                         sin(k * z * sqrt(1.0_dp - (wavelength * fx)**2)), dp)
        else
            H(i) = (0.0_dp, 0.0_dp)  ! evanescent
        end if
    end do

    ! Multiply in frequency domain
    do i = 1, n
        U(i) = U(i) * H(i)
    end do

    ! Inverse FFT
    call ifft1d(U, n)

    do i = 1, n
        u_out(i) = U(i)
    end do
end subroutine fresnel_propagate

!-----------------------------------------------------------------------
! Stokes parameters from Ex, Ey
! S0 = |Ex|^2 + |Ey|^2
! S1 = |Ex|^2 - |Ey|^2
! S2 = 2 * Re(Ex * conj(Ey))
! S3 = 2 * Im(Ex * conj(Ey))
!-----------------------------------------------------------------------
subroutine compute_stokes(Ex, Ey, Ny, Nx, S0, S1, S2, S3)
    complex(dp), intent(in)  :: Ex(Ny, Nx), Ey(Ny, Nx)
    integer, intent(in) :: Ny, Nx
    real(dp), intent(out) :: S0(Ny, Nx), S1(Ny, Nx), S2(Ny, Nx), S3(Ny, Nx)
    integer :: i, j
    complex(dp) :: prod
    do j = 1, Ny
        do i = 1, Nx
            prod = Ex(j, i) * conjg(Ey(j, i))
            S0(j, i) = real(Ex(j, i))**2 + aimag(Ex(j, i))**2 &
                     + real(Ey(j, i))**2 + aimag(Ey(j, i))**2
            S1(j, i) = real(Ex(j, i))**2 + aimag(Ex(j, i))**2 &
                     - real(Ey(j, i))**2 - aimag(Ey(j, i))**2
            S2(j, i) = 2.0_dp * real(prod)
            S3(j, i) = 2.0_dp * aimag(prod)
        end do
    end do
end subroutine compute_stokes

!-----------------------------------------------------------------------
! Skyrmion polarization texture field generation (Fortran)
! Canonical form matching the XPOCK Python core:
!
!   theta(r) = pi * min(max(r / w, 0), 1)   ! clamped: 0 at centre,
!                                           ! pi at r = w, uniform beyond
!   Ex = cos(theta/2)
!   Ey = sin(theta/2) * exp(i * q * phi)    ! pole-free Jones vector,
!                                           ! |Ex|^2 + |Ey|^2 = 1
!
! The previous LG-pair construction (l1, l2) did not reproduce the
! Python skyrmion number and has been replaced.
!-----------------------------------------------------------------------
subroutine skyrmion_field_f90(x, y, Nx, Ny, q, w, Ex, Ey)
    real(dp), intent(in) :: x(Nx), y(Ny)
    integer, intent(in) :: Nx, Ny, q
    real(dp), intent(in) :: w
    complex(dp), intent(out) :: Ex(Ny, Nx), Ey(Ny, Nx)
    real(dp) :: X, Y, r, phi, theta, half, sh
    real(dp) :: xc, yc
    integer :: i, j
    xc = 0.5_dp * (x(1) + x(Nx))
    yc = 0.5_dp * (y(1) + y(Ny))
    do j = 1, Ny
        do i = 1, Nx
            X = x(i) - xc
            Y = y(j) - yc
            r = sqrt(X**2 + Y**2)
            phi = atan2(Y, X)
            theta = PI * min(max(r / w, 0.0_dp), 1.0_dp)
            half = theta / 2.0_dp
            sh = sin(half)
            Ex(j, i) = cmplx(cos(half), 0.0_dp, dp)
            Ey(j, i) = cmplx(sh * cos(real(q, dp) * phi), &
                             sh * sin(real(q, dp) * phi), dp)
        end do
    end do
end subroutine skyrmion_field_f90

!-----------------------------------------------------------------------
! Scalar triple product u . (v x w)
!-----------------------------------------------------------------------
pure real(dp) function triple_dot(u, v, w)
    real(dp), intent(in) :: u(3), v(3), w(3)
    triple_dot = u(1) * (v(2) * w(3) - v(3) * w(2)) &
               + u(2) * (v(3) * w(1) - v(1) * w(3)) &
               + u(3) * (v(1) * w(2) - v(2) * w(1))
end function triple_dot

!-----------------------------------------------------------------------
! Skyrmion number Q from Stokes parameters (Fortran)
! Gauge-invariant lattice solid-angle (plaquette) method matching the
! Python core:
!
!   Q = (1/4pi) * sum_over_cells [ Omega(a, c, b) + Omega(a, d, c) ]
!
! with a=n(j,i), b=n(j,i+1), c=n(j+1,i+1), d=n(j+1,i),
! Omega(u,v,w) = 2 * atan2( u . (v x w), 1 + u.v + v.w + w.u ).
!
! Exactly integer Q for a lattice skyrmion. Replaces the old finite-
! difference gradient product.
!-----------------------------------------------------------------------
real(dp) function skyrmion_number_f90(S0, S1, S2, S3, x, y, Nx, Ny)
    real(dp), intent(in) :: S0(Ny, Nx), S1(Ny, Nx), S2(Ny, Nx), S3(Ny, Nx)
    real(dp), intent(in) :: x(Nx), y(Ny)
    integer, intent(in) :: Nx, Ny
    real(dp) :: Q
    real(dp), allocatable :: n(:, :, :)
    real(dp) :: s0, inv, norm, num, den
    integer :: i, j
    allocate(n(3, Nx, Ny))

    ! Normalized unit Stokes vectors per pixel (zero vector where S0 == 0,
    ! mirroring the Python core: divide by 1 when S0 == 0).
    do j = 1, Ny
        do i = 1, Nx
            s0 = S0(j, i)
            inv = 1.0_dp
            if (s0 > 0.0_dp) inv = 1.0_dp / s0
            norm = sqrt((S1(j, i) * inv)**2 + (S2(j, i) * inv)**2 & 
                        + (S3(j, i) * inv)**2)
            if (norm > 0.0_dp) then
                n(1, i, j) = S1(j, i) * inv / norm
                n(2, i, j) = S2(j, i) * inv / norm
                n(3, i, j) = S3(j, i) * inv / norm
            else
                n(1, i, j) = 0.0_dp
                n(2, i, j) = 0.0_dp
                n(3, i, j) = 0.0_dp
            end if
        end do
    end do

    Q = 0.0_dp
    do j = 1, Ny - 1
        do i = 1, Nx - 1
            ! Cell (a=n00, b=n01, c=n11, d=n10); triangles (a,c,b) and (a,d,c)
            num = triple_dot(n(:, i, j), n(:, i + 1, j + 1), n(:, i + 1, j))
            den = 1.0_dp + dot_product(n(:, i, j), n(:, i + 1, j + 1)) &
                  + dot_product(n(:, i + 1, j + 1), n(:, i + 1, j)) &
                  + dot_product(n(:, i + 1, j), n(:, i, j))
            Q = Q + 2.0_dp * atan2(num, den)

            num = triple_dot(n(:, i, j), n(:, i, j + 1), n(:, i + 1, j + 1))
            den = 1.0_dp + dot_product(n(:, i, j), n(:, i, j + 1)) &
                  + dot_product(n(:, i, j + 1), n(:, i + 1, j + 1)) &
                  + dot_product(n(:, i + 1, j + 1), n(:, i, j))
            Q = Q + 2.0_dp * atan2(num, den)
        end do
    end do

    deallocate(n)
    skyrmion_number_f90 = Q / (4.0_dp * PI)
end function skyrmion_number_f90
end module xpock_wave_ops