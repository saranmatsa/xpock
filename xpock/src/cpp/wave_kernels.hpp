#ifndef XPOCK_WAVE_KERNELS_HPP
#define XPOCK_WAVE_KERNELS_HPP

/*
 * XPOCK — C++ Wave Kernels
 *
 * Performance-critical numerical kernels for wave-structure generation.
 * These are header-declared here and implemented in wave_kernels.cpp.
 *
 * Intended to be compiled into a shared library and called from Python
 * (via pybind11 or ctypes) or used standalone. Provides:
 *   - 1D temporal envelope evaluation
 *   - 2D spatial mode evaluation (Hermite-Gaussian, Laguerre-Gaussian)
 *   - Complex field multiplication (spatio-temporal envelope)
 *   - Topological phase singularity imprinting
 *   - Stokes parameter computation
 *   - Winding number estimation
 *
 * All functions use double precision and std::complex<double>.
 * No external dependencies beyond the C++ standard library and
 * optional Eigen3 (used if available; otherwise plain loops).
 */

#include <complex>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

namespace xpock {

using Real = double;
using Complex = std::complex<Real>;
using CArray = std::vector<Complex>;
using RArray = std::vector<Real>;

// ---------------------------------------------------------------------
// 1D Temporal Envelopes
// ---------------------------------------------------------------------

enum class PulseShape {
    Gaussian,
    Sech,
    SuperGaussian,
    Rectangular,
    Exponential,
    ChirpedGaussian,
    Bisectorial,
    Custom,
};

struct TemporalParams {
    PulseShape shape = PulseShape::Gaussian;
    Real amplitude = 1.0;
    Real center = 0.0;
    Real duration = 1.0;
    Real chirp_rate = 0.0;
    Real carrier_freq = 0.0;
};

/**
 * Evaluate a 1D temporal envelope on a time grid.
 *
 * @param t       input time array (size N)
 * @param params  temporal parameters
 * @return        complex envelope including carrier and chirp (size N)
 */
CArray temporal_envelope(const RArray& t, const TemporalParams& params);

// ---------------------------------------------------------------------
// 2D Spatial Modes
// ---------------------------------------------------------------------

enum class SpatialMode {
    Gaussian,
    HermiteGaussian,
    LaguerreGaussian,
    Bessel,
    SuperGaussian,
    Spherical,
    PlaneWave,
};

struct SpatialParams {
    SpatialMode mode = SpatialMode::Gaussian;
    int n = 0;      // HG x-order
    int m = 0;      // HG y-order
    int p = 0;      // LG radial index
    int l = 0;      // LG azimuthal index (OAM)
    Real waist = 1.0;
    Real wavelength = 1.0;
    Real k = 0.0;   // wavenumber, computed from wavelength if 0
    Real x0 = 0.0;  // beam center
    Real y0 = 0.0;
    Real z0 = 0.0;
    Real z = 0.0;   // axial position
    Real aperture_radius = -1.0; // <0 means no aperture
};

/**
 * Evaluate a 2D spatial mode on a Cartesian grid.
 *
 * @param x       x coordinates (size Nx)
 * @param y       y coordinates (size Ny)
 * @param params  spatial mode parameters
 * @return        complex field psi(y, x) stored row-major (Ny x Nx)
 */
CArray spatial_mode(const RArray& x, const RArray& y, const SpatialParams& params);

// ---------------------------------------------------------------------
// Spatio-temporal combination
// ---------------------------------------------------------------------

/**
 * Multiply a 2D spatial field (Ny x Nx) by a 1D temporal envelope (Nt)
 * to produce a 3D field (Ny x Nx x Nt) stored flat.
 *
 * @param spatial  complex spatial field, size Ny*Nx
 * @param temporal complex temporal envelope, size Nt
 * @param Ny, Nx, Nt dimensions
 * @return         complex field, size Ny*Nx*Nt
 */
CArray combine_spacetime(const CArray& spatial, const CArray& temporal,
                         int Ny, int Nx, int Nt);

// ---------------------------------------------------------------------
// Topological phase singularity
// ---------------------------------------------------------------------

/**
 * Imprint a vortex of charge l on a complex field:
 *   field *= exp(i * l * phi), phi = atan2(y, x)
 *
 * @param field  complex field (modified in place), size Ny*Nx
 * @param x      x coordinates (size Nx)
 * @param y      y coordinates (size Ny)
 * @param l      topological charge
 */
void imprint_vortex(CArray& field, const RArray& x, const RArray& y, int l);

// ---------------------------------------------------------------------
// Polarization (Jones vectors)
// ---------------------------------------------------------------------

struct JonesParams {
    // 0=linear, 1=circular, 2=elliptical, 3=radial, 4=azimuthal, 5=skyrmion
    int state = 0;
    Real angle = 0.0;           // linear polarization angle (rad)
    int handedness = 1;         // +1 LCP, -1 RCP
    Real skyrmion_radius = 1.0;
    int skyrmion_charge = 1;
};

/**
 * Evaluate Jones vector components Ex, Ey on a 2D grid.
 *
 * @param x   x coordinates (size Nx)
 * @param y   y coordinates (size Ny)
 * @param p   Jones parameters
 * @param Ex  output (size Ny*Nx)
 * @param Ey  output (size Ny*Nx)
 */
void jones_vector(const RArray& x, const RArray& y, const JonesParams& p,
                  CArray& Ex, CArray& Ey);

// ---------------------------------------------------------------------
// Stokes parameters
// ---------------------------------------------------------------------

struct Stokes {
    RArray S0, S1, S2, S3; // each size Ny*Nx
};

/**
 * Compute Stokes parameters from Ex, Ey fields.
 */
Stokes compute_stokes(const CArray& Ex, const CArray& Ey, int Ny, int Nx);

// ---------------------------------------------------------------------
// Winding number
// ---------------------------------------------------------------------

/**
 * Estimate the winding number of a phase field on a 2D grid.
 *
 * @param phase  phase array (size Ny*Nx), in [-pi, pi]
 * @param x      x coordinates
 * @param y      y coordinates
 * @return       estimated integer winding number
 */
int winding_number(const RArray& phase, const RArray& x, const RArray& y);

// ---------------------------------------------------------------------
// Skyrmion polarization texture
// ---------------------------------------------------------------------

/**
 * Generate a skyrmionic polarization texture on a 2D grid.
 *
 * The field is |Psi> = |l1>|H> + |l2>|V> where skyrmion number
 * Q = l1 - l2. Ex and Ey components encode the polarization texture.
 *
 * @param x   x coordinates (size Nx)
 * @param y   y coordinates (size Ny)
 * @param l1  OAM of horizontal component
 * @param l2  OAM of vertical component
 * @param w   skyrmion texture radius (m)
 * @param Ex  output Ex field (size Ny*Nx)
 * @param Ey  output Ey field (size Ny*Nx)
 */
void skyrmion_field(const RArray& x, const RArray& y,
                    int l1, int l2, Real w,
                    CArray& Ex, CArray& Ey);

/**
 * Compute skyrmion number Q from Stokes parameters:
 *   Q = (1/4pi) * integral n · (dn_x × dn_y) dx dy
 * where n = (S1, S2, S3) / S0
 *
 * @param S0  intensity (size Ny*Nx)
 * @param S1  Stokes Q (size Ny*Nx)
 * @param S2  Stokes U (size Ny*Nx)
 * @param S3  Stokes V (size Ny*Nx)
 * @param x   x coordinates (size Nx)
 * @param y   y coordinates (size Ny)
 * @return    skyrmion number Q (should be near an integer)
 */
Real skyrmion_number(const RArray& S0, const RArray& S1,
                     const RArray& S2, const RArray& S3,
                     const RArray& x, const RArray& y);

// ---------------------------------------------------------------------
// Complex field utilities
// ---------------------------------------------------------------------

/**
 * Element-wise complex multiply: out[i] = a[i] * b[i].
 */
CArray complex_multiply(const CArray& a, const CArray& b);

/**
 * Element-wise complex magnitude squared: out[i] = |a[i]|^2.
 */
RArray magnitude_squared(const CArray& a);

/**
 * Total integrated power (sum of |E|^2 * dx * dy * dt).
 */
Real total_power(const CArray& field, Real dx, Real dy, Real dt);

} // namespace xpock

#endif // XPOCK_WAVE_KERNELS_HPP