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
    do i = 1, n
        fx = (i - n / 2 - 1) * df
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
! |Psi> = |l1>|H> + |l2>|V>, skyrmion number Q = l1 - l2
!-----------------------------------------------------------------------
subroutine skyrmion_field_f90(x, y, Nx, Ny, l1, l2, w, Ex, Ey)
    real(dp), intent(in) :: x(Nx), y(Ny)
    integer, intent(in) :: Nx, Ny, l1, l2
    real(dp), intent(in) :: w
    complex(dp), intent(out) :: Ex(Ny, Nx), Ey(Ny, Nx)
    real(dp) :: X, Y, r, phi, env, radial1, radial2
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
            env = exp(-(r / w)**2)
            radial1 = (r / w)**abs(l1) * env
            radial2 = (r / w)**abs(l2) * env
            Ex(j, i) = radial1 * cmplx(cos(l1 * phi), sin(l1 * phi), dp)
            Ey(j, i) = radial2 * cmplx(cos(l2 * phi), sin(l2 * phi), dp)
        end do
    end do
end subroutine skyrmion_field_f90

!-----------------------------------------------------------------------
! Skyrmion number Q from Stokes parameters
! Q = (1/4pi) * integral n · (dn_x × dn_y) dx dy
! where n = (S1, S2, S3) / S0
!-----------------------------------------------------------------------
real(dp) function skyrmion_number_f90(S0, S1, S2, S3, x, y, Nx, Ny)
    real(dp), intent(in) :: S0(Ny, Nx), S1(Ny, Nx), S2(Ny, Nx), S3(Ny, Nx)
    real(dp), intent(in) :: x(Nx), y(Ny)
    integer, intent(in) :: Nx, Ny
    real(dp) :: Q, dx, dy
    real(dp) :: nx, ny, nz
    real(dp) :: dnx_dx, dny_dx, dnz_dx
    real(dp) :: dnx_dy, dny_dy, dnz_dy
    real(dp) :: cross_x, cross_y, cross_z
    real(dp) :: nx_r, ny_r, nz_r, nx_l, ny_l, nz_l
    real(dp) :: nx_d, ny_d, nz_d, nx_u, ny_u, nz_u
    real(dp) :: s0_r, s0_l, s0_d, s0_u
    integer :: i, j
    Q = 0.0_dp
    dx = x(2) - x(1)
    dy = y(2) - y(1)
    do j = 2, Ny - 1
        do i = 2, Nx - 1
            s0_r = S0(j, i+1); s0_l = S0(j, i-1)
            s0_d = S0(j+1, i); s0_u = S0(j-1, i)
            if (S0(j,i) < 1e-12_dp) cycle
            if (s0_r < 1e-12_dp .or. s0_l < 1e-12_dp) cycle
            if (s0_d < 1e-12_dp .or. s0_u < 1e-12_dp) cycle

            nx = S1(j,i) / S0(j,i)
            ny = S2(j,i) / S0(j,i)
            nz = S3(j,i) / S0(j,i)

            nx_r = S1(j,i+1) / s0_r; ny_r = S2(j,i+1) / s0_r; nz_r = S3(j,i+1) / s0_r
            nx_l = S1(j,i-1) / s0_l; ny_l = S2(j,i-1) / s0_l; nz_l = S3(j,i-1) / s0_l
            nx_d = S1(j+1,i) / s0_d; ny_d = S2(j+1,i) / s0_d; nz_d = S3(j+1,i) / s0_d
            nx_u = S1(j-1,i) / s0_u; ny_u = S2(j-1,i) / s0_u; nz_u = S3(j-1,i) / s0_u

            dnx_dx = (nx_r - nx_l) / (2.0_dp * dx)
            dny_dx = (ny_r - ny_l) / (2.0_dp * dx)
            dnz_dx = (nz_r - nz_l) / (2.0_dp * dx)

            dnx_dy = (nx_d - nx_u) / (2.0_dp * dy)
            dny_dy = (ny_d - ny_u) / (2.0_dp * dy)
            dnz_dy = (nz_d - nz_u) / (2.0_dp * dy)

            cross_x = dny_dx * dnz_dy - dnz_dx * dny_dy
            cross_y = dnz_dx * dnx_dy - dnx_dx * dnz_dy
            cross_z = dnx_dx * dny_dy - dny_dx * dnx_dy

            Q = Q + nx * cross_x + ny * cross_y + nz * cross_z
        end do
    end do
    skyrmion_number_f90 = Q * dx * dy / (4.0_dp * PI)
end function skyrmion_number_f90
end module xpock_wave_ops