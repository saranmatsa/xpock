/*
 * XPOCK — C++ Wave Kernels (implementation)
 *
 * Compile with:
 *   g++ -O3 -std=c++17 -shared -fPIC wave_kernels.cpp -o libxpock_kernels.so
 *
 * Requires no external libraries beyond the C++ standard library.
 */

#include "wave_kernels.hpp"

#include <numeric>
#include <algorithm>
#include <array>

namespace xpock {

// ---------------------------------------------------------------------
// Helper: factorial (for HG/LG normalization)
// ---------------------------------------------------------------------

static Real factorial_int(int n) {
    Real r = 1.0;
    for (int i = 2; i <= n; ++i) r *= static_cast<Real>(i);
    return r;
}

static Real hermite_poly(int n, Real x) {
    // Physicists' Hermite polynomials via recurrence
    if (n == 0) return 1.0;
    if (n == 1) return 2.0 * x;
    Real h_prev = 1.0;
    Real h_curr = 2.0 * x;
    for (int i = 2; i <= n; ++i) {
        Real h_next = 2.0 * x * h_curr - 2.0 * static_cast<Real>(i - 1) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    return h_curr;
}

static Real gen_laguerre(int p, Real alpha, Real x) {
    // Generalized Laguerre polynomial L_p^alpha(x) via recurrence
    if (p == 0) return 1.0;
    if (p == 1) return -x + alpha + 1.0;
    Real L_prev = 1.0;
    Real L_curr = -x + alpha + 1.0;
    for (int k = 1; k < p; ++k) {
        Real L_next = ((2 * k + 1 + alpha - x) * L_curr - (k + alpha) * L_prev) /
                      static_cast<Real>(k + 1);
        L_prev = L_curr;
        L_curr = L_next;
    }
    return L_curr;
}

// ---------------------------------------------------------------------
// Temporal envelope
// ---------------------------------------------------------------------

CArray temporal_envelope(const RArray& t, const TemporalParams& params) {
    size_t N = t.size();
    CArray env(N);
    Real amp = params.amplitude;
    Real tau0 = params.center;
    Real dur = params.duration;

    for (size_t i = 0; i < N; ++i) {
        Real tau = t[i] - tau0;
        Real env_real = 0.0;

        switch (params.shape) {
            case PulseShape::Gaussian: {
                Real sigma = dur / (2.0 * std::sqrt(2.0 * std::log(2.0)));
                env_real = std::exp(-0.5 * (tau / sigma) * (tau / sigma));
                break;
            }
            case PulseShape::Sech: {
                env_real = 1.0 / std::cosh(tau / dur);
                break;
            }
            case PulseShape::SuperGaussian: {
                int order = 4;
                env_real = std::exp(-std::pow(std::fabs(tau) / dur, 2 * order));
                break;
            }
            case PulseShape::Rectangular: {
                env_real = (std::fabs(tau) <= dur / 2.0) ? 1.0 : 0.0;
                break;
            }
            case PulseShape::Exponential: {
                env_real = (tau >= 0.0) ? std::exp(-tau / dur) : 0.0;
                break;
            }
            case PulseShape::ChirpedGaussian: {
                Real sigma = dur / (2.0 * std::sqrt(2.0 * std::log(2.0)));
                env_real = std::exp(-0.5 * (tau / sigma) * (tau / sigma));
                break;
            }
            case PulseShape::Bisectorial: {
                Real w = dur / 4.0;
                Real e1 = std::exp(-((tau - dur / 2.0) / w) * ((tau - dur / 2.0) / w));
                Real e2 = std::exp(-((tau + dur / 2.0) / w) * ((tau + dur / 2.0) / w));
                env_real = e1 + e2;
                break;
            }
            default:
                env_real = 1.0;
        }

        Complex val(env_real * amp, 0.0);

        // Quadratic chirp
        if (params.chirp_rate != 0.0) {
            val *= std::exp(Complex(0.0, 0.5 * params.chirp_rate * tau * tau));
        }

        // Carrier
        if (params.carrier_freq != 0.0) {
            val *= std::exp(Complex(0.0, -params.carrier_freq * t[i]));
        }

        env[i] = val;
    }
    return env;
}

// ---------------------------------------------------------------------
// Spatial modes
// ---------------------------------------------------------------------

CArray spatial_mode(const RArray& x, const RArray& y, const SpatialParams& params) {
    size_t Nx = x.size();
    size_t Ny = y.size();
    CArray field(Ny * Nx);

    Real k = params.k;
    if (k == 0.0) k = 2.0 * M_PI / params.wavelength;

    Real w0 = params.waist;
    Real zR = M_PI * w0 * w0 / params.wavelength;
    Real Z = params.z - params.z0;

    Real w = w0 * std::sqrt(1.0 + (Z / zR) * (Z / zR));
    Real R = (Z != 0.0) ? Z * (1.0 + (zR / Z) * (zR / Z)) : std::numeric_limits<Real>::infinity();
    Real gouy = (params.n + params.m + 1) * std::atan(Z / zR);

    for (size_t j = 0; j < Ny; ++j) {
        Real Yc = y[j] - params.y0;
        for (size_t i = 0; i < Nx; ++i) {
            Real Xc = x[i] - params.x0;
            Real r2 = Xc * Xc + Yc * Yc;
            Real r = std::sqrt(r2);
            Real phi = std::atan2(Yc, Xc);

            Complex psi(0.0, 0.0);

            switch (params.mode) {
                case SpatialMode::Gaussian: {
                    Real amp = std::exp(-r2 / (w * w));
                    Real phase = (std::isfinite(R)) ? k * r2 / (2.0 * R) : 0.0;
                    phase -= gouy;
                    Real norm = 1.0 / (w * std::sqrt(M_PI));
                    psi = Complex(norm * amp * std::cos(phase), norm * amp * std::sin(phase));
                    break;
                }
                case SpatialMode::HermiteGaussian: {
                    Real Hn = hermite_poly(params.n, std::sqrt(2.0) * Xc / w);
                    Real Hm = hermite_poly(params.m, std::sqrt(2.0) * Yc / w);
                    Real amp = std::exp(-r2 / (w * w)) * Hn * Hm;
                    Real phase = k * r2 / (2.0 * R) - gouy;
                    Real norm = 1.0 / (w * std::sqrt(M_PI *
                        std::pow(2.0, params.n + params.m) *
                        factorial_int(params.n) * factorial_int(params.m)));
                    psi = Complex(norm * amp * std::cos(phase), norm * amp * std::sin(phase));
                    break;
                }
                case SpatialMode::LaguerreGaussian: {
                    Real w_l = w * std::sqrt(2.0);
                    Real L = gen_laguerre(params.p, std::fabs(params.l), 2.0 * r2 / (w_l * w_l));
                    Real amp = std::pow(std::sqrt(2.0) * r / w_l, std::fabs(params.l)) *
                               std::exp(-r2 / (w_l * w_l)) * L;
                    Real phase = params.l * phi + k * r2 / (2.0 * R) -
                                 (2 * params.p + std::fabs(params.l) + 1) * std::atan(Z / zR);
                    Real norm = 1.0 / (w_l * std::sqrt(M_PI)) *
                                std::sqrt(factorial_int(params.p) /
                                          factorial_int(params.p + static_cast<int>(std::fabs(params.l))));
                    psi = Complex(norm * amp * std::cos(phase), norm * amp * std::sin(phase));
                    break;
                }
                case SpatialMode::Bessel: {
                    // J0 approximation via series (simplified)
                    Real k_r = k * 0.1;
                    Real k_z = std::sqrt(k * k - k_r * k_r);
                    // J0(kr) approximation
                    Real xj = k_r * r;
                    Real J0 = 1.0 - xj*xj/4.0 + xj*xj*xj*xj/64.0; // rough
                    psi = Complex(J0 * std::cos(k_z * Z), J0 * std::sin(k_z * Z));
                    break;
                }
                case SpatialMode::SuperGaussian: {
                    int order = 4;
                    Real amp = std::exp(-std::pow(r2 / (w * w), order));
                    psi = Complex(amp, 0.0);
                    break;
                }
                case SpatialMode::Spherical: {
                    Real rr = std::sqrt(r2 + Z * Z);
                    if (rr < 1e-12) rr = 1e-12;
                    psi = Complex(std::cos(k * rr) / rr, std::sin(k * rr) / rr);
                    break;
                }
                case SpatialMode::PlaneWave: {
                    psi = Complex(std::cos(k * Z), std::sin(k * Z));
                    break;
                }
            }

            // Hard aperture
            if (params.aperture_radius > 0.0 && r > params.aperture_radius) {
                psi = Complex(0.0, 0.0);
            }

            field[j * Nx + i] = psi;
        }
    }
    return field;
}

// ---------------------------------------------------------------------
// Spatio-temporal combination
// ---------------------------------------------------------------------

CArray combine_spacetime(const CArray& spatial, const CArray& temporal,
                         int Ny, int Nx, int Nt) {
    size_t total = static_cast<size_t>(Ny) * Nx * Nt;
    CArray out(total);
    size_t idx = 0;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            Complex sp = spatial[j * Nx + i];
            for (int k = 0; k < Nt; ++k) {
                out[idx++] = sp * temporal[k];
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------
// Topological phase singularity
// ---------------------------------------------------------------------

void imprint_vortex(CArray& field, const RArray& x, const RArray& y, int l) {
    size_t Nx = x.size();
    size_t Ny = y.size();
    for (size_t j = 0; j < Ny; ++j) {
        Real Yc = y[j];
        for (size_t i = 0; i < Nx; ++i) {
            Real Xc = x[i];
            Real phi = std::atan2(Yc, Xc);
            Complex factor(std::cos(l * phi), std::sin(l * phi));
            field[j * Nx + i] *= factor;
        }
    }
}

// ---------------------------------------------------------------------
// Jones vectors
// ---------------------------------------------------------------------

void jones_vector(const RArray& x, const RArray& y, const JonesParams& p,
                  CArray& Ex, CArray& Ey) {
    size_t Nx = x.size();
    size_t Ny = y.size();
    Ex.resize(Ny * Nx);
    Ey.resize(Ny * Nx);

    for (size_t j = 0; j < Ny; ++j) {
        Real Yc = y[j];
        for (size_t i = 0; i < Nx; ++i) {
            Real Xc = x[i];
            size_t idx = j * Nx + i;
            Complex ex(0.0), ey(0.0);

            switch (p.state) {
                case 0: { // linear
                    Real c = std::cos(p.angle);
                    Real s = std::sin(p.angle);
                    ex = Complex(c, 0.0);
                    ey = Complex(s, 0.0);
                    break;
                }
                case 1: { // circular
                    Real sgn = (p.handedness > 0) ? 1.0 : -1.0;
                    ex = Complex(1.0 / std::sqrt(2.0), 0.0);
                    ey = Complex(0.0, sgn / std::sqrt(2.0));
                    break;
                }
                case 2: { // elliptical
                    Real theta = p.angle;
                    Real eps = 0.5;
                    ex = Complex(std::cos(theta) * std::cos(eps), 0.0);
                    ey = Complex(std::sin(theta) * std::cos(eps), std::sin(eps));
                    Real norm = std::sqrt(std::norm(ex) + std::norm(ey));
                    if (norm > 0) { ex /= norm; ey /= norm; }
                    break;
                }
                case 3: { // radial
                    Real phi = std::atan2(Yc, Xc);
                    ex = Complex(std::cos(phi), 0.0);
                    ey = Complex(std::sin(phi), 0.0);
                    break;
                }
                case 4: { // azimuthal
                    Real phi = std::atan2(Yc, Xc);
                    ex = Complex(-std::sin(phi), 0.0);
                    ey = Complex(std::cos(phi), 0.0);
                    break;
                }
                case 5: { // skyrmion
                    Real R = p.skyrmion_radius;
                    int q = p.skyrmion_charge;
                    Real r = std::sqrt(Xc * Xc + Yc * Yc);
                    Real phi = std::atan2(Yc, Xc);
                    Real theta = M_PI * r / R;
                    Real nx = std::sin(theta) * std::cos(q * phi);
                    Real ny = std::sin(theta) * std::sin(q * phi);
                    Real nz = std::cos(theta);
                    Real ex_r = std::sqrt(std::max(0.0, (1.0 + nz) / 2.0));
                    Real ey_r = (nx) / (2.0 * ex_r + 1e-15);
                    Real ey_i = (ny) / (2.0 * ex_r + 1e-15);
                    ex = Complex(ex_r, 0.0);
                    ey = Complex(ey_r, ey_i);
                    break;
                }
                default:
                    ex = Complex(1.0, 0.0);
                    ey = Complex(0.0, 0.0);
            }
            Ex[idx] = ex;
            Ey[idx] = ey;
        }
    }
}

// ---------------------------------------------------------------------
// Stokes parameters
// ---------------------------------------------------------------------

Stokes compute_stokes(const CArray& Ex, const CArray& Ey, int Ny, int Nx) {
    Stokes st;
    size_t total = static_cast<size_t>(Ny) * Nx;
    st.S0.resize(total);
    st.S1.resize(total);
    st.S2.resize(total);
    st.S3.resize(total);

    for (size_t i = 0; i < total; ++i) {
        Real s0 = std::norm(Ex[i]) + std::norm(Ey[i]);
        Real s1 = std::norm(Ex[i]) - std::norm(Ey[i]);
        Real s2 = 2.0 * std::real(Ex[i] * std::conj(Ey[i]));
        Real s3 = 2.0 * std::imag(Ex[i] * std::conj(Ey[i]));
        st.S0[i] = s0;
        st.S1[i] = s1;
        st.S2[i] = s2;
        st.S3[i] = s3;
    }
    return st;
}

// ---------------------------------------------------------------------
// Winding number
// ---------------------------------------------------------------------

int winding_number(const RArray& phase, const RArray& x, const RArray& y) {
    size_t Nx = x.size();
    size_t Ny = y.size();
    // Simple approach: unwrap phase along a circular contour and count wraps
    // Use the sum of phase differences around the perimeter as proxy.
    Real total_wrap = 0.0;
    // Traverse the boundary of the grid
    for (size_t i = 0; i < Nx - 1; ++i) {
        size_t idx1 = 0 * Nx + i;
        size_t idx2 = 0 * Nx + (i + 1);
        Real d = phase[idx2] - phase[idx1];
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        total_wrap += d;
    }
    for (size_t j = 0; j < Ny - 1; ++j) {
        size_t idx1 = j * Nx + (Nx - 1);
        size_t idx2 = (j + 1) * Nx + (Nx - 1);
        Real d = phase[idx2] - phase[idx1];
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        total_wrap += d;
    }
    for (size_t i = Nx - 1; i > 0; --i) {
        size_t idx1 = (Ny - 1) * Nx + i;
        size_t idx2 = (Ny - 1) * Nx + (i - 1);
        Real d = phase[idx2] - phase[idx1];
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        total_wrap += d;
    }
    for (size_t j = Ny - 1; j > 0; --j) {
        size_t idx1 = j * Nx + 0;
        size_t idx2 = (j - 1) * Nx + 0;
        Real d = phase[idx2] - phase[idx1];
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        total_wrap += d;
    }
    return static_cast<int>(std::round(total_wrap / (2.0 * M_PI)));
}

// ---------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------

CArray complex_multiply(const CArray& a, const CArray& b) {
    size_t N = a.size();
    CArray out(N);
    for (size_t i = 0; i < N; ++i) out[i] = a[i] * b[i];
    return out;
}

RArray magnitude_squared(const CArray& a) {
    size_t N = a.size();
    RArray out(N);
    for (size_t i = 0; i < N; ++i) out[i] = std::norm(a[i]);
    return out;
}

Real total_power(const CArray& field, Real dx, Real dy, Real dt) {
    Real sum = 0.0;
    for (const auto& v : field) sum += std::norm(v);
    return sum * dx * dy * dt;
}

// ---------------------------------------------------------------------
// Skyrmion polarization texture (canonical: matches Python core)
// ---------------------------------------------------------------------

void skyrmion_field(const RArray& x, const RArray& y,
                    int q, Real w,
                    CArray& Ex, CArray& Ey) {
    size_t Nx = x.size();
    size_t Ny = y.size();
    size_t N = Nx * Ny;
    Ex.resize(N);
    Ey.resize(N);

    Real xc = 0.5 * (x.front() + x.back());
    Real yc = 0.5 * (y.front() + y.back());

    // Clamp theta to [0, pi]: texture confined to radius w, uniform
    // boundary polarization beyond w. Without this clamp theta grows
    // without bound across the grid and the computed skyrmion number is
    // neither integer nor equal to the imposed charge.
    for (size_t j = 0; j < Ny; ++j) {
        for (size_t i = 0; i < Nx; ++i) {
            size_t idx = j * Nx + i;
            Real X = x[i] - xc;
            Real Y = y[j] - yc;
            Real r = std::sqrt(X * X + Y * Y);
            Real phi = std::atan2(Y, X);

            Real theta = M_PI * std::clamp(r / w, 0.0, 1.0);
            Real half = theta / 2.0;
            Real sh = std::sin(half);

            // Pole-free Jones vector: |Ex|^2 + |Ey|^2 == 1 everywhere,
            // no division by Ex at the Bloch-sphere poles.
            Ex[idx] = Complex(std::cos(half), 0.0);
            Ey[idx] = Complex(sh * std::cos(q * phi),
                              sh * std::sin(q * phi));
        }
    }
}

// Solid angle of the triangle (a, b, c) of unit vectors on the sphere:
//   Omega = 2 * atan2( a . (b x c), 1 + a.b + b.c + c.a )
static Real solid_angle_triangle(const std::array<Real, 3>& a,
                                 const std::array<Real, 3>& b,
                                 const std::array<Real, 3>& c) {
    Real num = a[0] * (b[1] * c[2] - b[2] * c[1])
             + a[1] * (b[2] * c[0] - b[0] * c[2])
             + a[2] * (b[0] * c[1] - b[1] * c[0]);
    Real den = 1.0
             + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
             + b[0] * c[0] + b[1] * c[1] + b[2] * c[2]
             + c[0] * a[0] + c[1] * a[1] + c[2] * a[2];
    return 2.0 * std::atan2(num, den);
}

Real skyrmion_number(const RArray& S0, const RArray& S1,
                     const RArray& S2, const RArray& S3,
                     const RArray& x, const RArray& y) {
    size_t Nx = x.size();
    size_t Ny = y.size();

    // Normalized Stokes (unit) vectors, one per pixel. Mirrors the Python
    // core: n = (S1,S2,S3)/S0 then re-normalized; zero-intensity pixels
    // become the zero vector (S0==0 -> divide by 1).
    std::vector<std::array<Real, 3>> n(Nx * Ny);
    for (size_t j = 0; j < Ny; ++j) {
        for (size_t i = 0; i < Nx; ++i) {
            size_t idx = j * Nx + i;
            Real s0 = S0[idx];
            Real inv = (s0 > 0.0) ? 1.0 / s0 : 1.0;
            Real nx = S1[idx] * inv;
            Real ny = S2[idx] * inv;
            Real nz = S3[idx] * inv;
            Real norm = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (norm > 0.0) {
                n[idx] = {nx / norm, ny / norm, nz / norm};
            } else {
                n[idx] = {0.0, 0.0, 0.0};
            }
        }
    }

    // Gauge-invariant lattice solid-angle (plaquette) method. Each
    // elementary cell (a=n00, b=n01, c=n11, d=n10) is split into
    // triangles (a, c, b) and (a, d, c), matching the Python core's
    // vertex ordering so Q == +q for a skyrmion of charge q.
    Real Q = 0.0;
    for (size_t j = 0; j < Ny - 1; ++j) {
        for (size_t i = 0; i < Nx - 1; ++i) {
            const std::array<Real, 3>& a = n[j * Nx + i];
            const std::array<Real, 3>& b = n[j * Nx + i + 1];
            const std::array<Real, 3>& c = n[(j + 1) * Nx + i + 1];
            const std::array<Real, 3>& d = n[(j + 1) * Nx + i];
            Q += solid_angle_triangle(a, c, b);
            Q += solid_angle_triangle(a, d, c);
        }
    }
    return Q / (4.0 * M_PI);
}

} // namespace xpock