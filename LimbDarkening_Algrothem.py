import numpy as np
import matplotlib.pyplot as plt
import batman

KEPLER_LONG_CADENCE_DAYS = 29.4 / 60.0 / 24.0 
def _occulted_fraction(r, z, rp):
    """Fraction of the ring at radius r that the planet is covering.

    r  : ring radii, shape (n_points, n_ring)
    z  : planet-centre distances, shape (n_points, 1), broadcast against r
    rp : planet radius

    Three cases, in the order they are applied below:
      r + z <= rp        the whole ring is behind the planet     -> 1
      |r - z| >= rp      the ring misses the planet entirely     -> 0
      otherwise          partial overlap -> arccos(...) / pi
    """
    f = np.zeros_like(r)
    f[r + z <= rp] = 1.0

    partial = (np.abs(r - z) < rp) & (r + z > rp)
    # np.maximum guards r*z == 0 at the very centre of the disc
    cos_phi = np.clip((r**2 + z**2 - rp**2) / np.maximum(2.0 * r * z, 1e-12),
                      -1.0, 1.0)
    f[partial] = np.arccos(cos_phi[partial]) / np.pi
    return f
def _blocked_fraction(z, rp, u1, u2, n_ring=200):
    """Fraction of the star's TOTAL flux hidden, for each separation in z."""
    blocked = np.zeros_like(z)

    # Only points that are actually in transit need the integral at all.
    # On a folded light curve that is a few percent of the data, so this
    # one line is most of the reason the model is fast enough to fit with.
    in_transit = z < 1.0 + rp
    if not np.any(in_transit):
        return blocked
    zs = z[in_transit]

    # Integrate only over rings the planet actually touches, [z-rp, z+rp],
    # clipped at the stellar limb. No point summing rings it never reaches.
    lo = np.clip(zs - rp, 0.0, 1.0)
    hi = np.clip(zs + rp, 0.0, 1.0)

    # Cosine-clustered nodes instead of evenly spaced ones. The covered
    # fraction has square-root-shaped kinks at both ends of [lo, hi], and
    # uniform spacing converges painfully slowly across them. This packs
    # more points where the integrand is badly behaved.
    s = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_ring)))
    r = np.maximum(lo[:, None] + (hi - lo)[:, None] * s[None, :], 1e-12)
    zz = zs[:, None]

    mu = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))      # cos(angle from centre)
    I = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2  # quadratic limb-darkening law

    # I x circumference x covered fraction, summed over rings
    num = np.trapezoid(I * 2.0 * np.pi * r * _occulted_fraction(r, zz, rp),
                       r, axis=1)

    # When the planet covers the disc centre (z < rp), every ring inside
    # radius (rp - z) is fully hidden and sits below our integration range.
    # Add those separately or you lose flux near mid-transit.
    core = zs < rp
    if np.any(core):
        rc = np.maximum((rp - zs[core])[:, None] * s[None, :], 1e-12)
        muc = np.sqrt(np.clip(1.0 - rc * rc, 0.0, 1.0))
        Ic = 1.0 - u1 * (1.0 - muc) - u2 * (1.0 - muc) ** 2
        num[core] += np.trapezoid(Ic * 2.0 * np.pi * rc, rc, axis=1)

    # Total flux of the unobscured disc has a closed form:
    #   integral of I(r) 2 pi r dr from 0 to 1  =  pi (1 - u1/3 - u2/6)
    # Dividing by it makes "blocked" a fraction of the whole star, so the
    # out-of-transit model is exactly H.
    blocked[in_transit] = num / (np.pi * (1.0 - u1 / 3.0 - u2 / 6.0))
    return blocked
def limbdark_model(phase, rp, a, inc, u1, u2, H, real_period_days,
                   n_ring=200, n_sub=11,
                   exp_time_days=KEPLER_LONG_CADENCE_DAYS):
    """Quadratic limb-darkened transit on a circular orbit.

    phase            : orbital phase. Works on [0,1) or [-0.5,0.5) -- the
                       wrap below handles either, same as trapezoid_model.
    real_period_days : needed ONLY to convert the exposure time into phase
                       units for the smearing correction. Pass the period
                       your BLS search found.
    n_sub            : number of sub-exposures averaged per data point.

    DO NOT SET n_sub = 1 ON KEPLER LONG-CADENCE DATA. Each exposure is
    29.4 minutes long, which is comparable to the ingress duration, so the
    data is a smeared version of the true shape. Ignoring that is a
    ~2300 ppm error on a ~11600 ppm transit -- concentrated entirely in
    ingress and egress, which is where the shape information lives.
    Measured worst-case error against a converged reference:
        n_sub =  1  ->  2320 ppm      n_sub =  7  ->  39 ppm
        n_sub =  5  ->    71 ppm      n_sub = 11  ->  21 ppm
    """
    phase = np.atleast_1d(np.asarray(phase, dtype=float))
    inc_rad = np.radians(inc)

    # Spread each data point into n_sub samples across its exposure, then
    # average at the end -- this is what reproduces the smearing.
    if n_sub > 1 and real_period_days and exp_time_days:
        offsets = ((np.arange(n_sub) + 0.5) / n_sub - 0.5) * (exp_time_days / real_period_days)
        ph = phase[:, None] + offsets[None, :]
    else:
        ph = phase[:, None]

    # Same wrapping convention as trapezoid_model, so the two models can be
    # overplotted on the same axis without any conversion.
    theta = 2.0 * np.pi * ((ph + 0.5) % 1.0 - 0.5)

    # Sky-projected planet-star separation, in stellar radii
    z = a * np.sqrt(np.sin(theta) ** 2 + (np.cos(inc_rad) * np.cos(theta)) ** 2)
    z = np.where(np.cos(theta) > 0, z, 1e3)   # planet behind the star -> no transit

    return H * (1.0 - _blocked_fraction(z, rp, u1, u2, n_ring).mean(axis=1))
def validate_against_batman(rp=0.10, a=12.0, inc=88.5, u1=0.35, u2=0.22,
                            P=3.2, n_sub=11):
    """Sanity check. Run this once and put the number in your write-up."""
    if batman is None:
        print("batman not installed -- skipping validation")
        return None
    phase = np.linspace(-0.5, 0.5, 3000)
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, 1.0, rp, a, inc
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [u1, u2]
    ref = batman.TransitModel(p, phase, supersample_factor=201,
                              exp_time=KEPLER_LONG_CADENCE_DAYS / P).light_curve(p)
    mine = limbdark_model(phase, rp, a, inc, u1, u2, 1.0, P, n_sub=n_sub)
    worst = np.max(np.abs(mine - ref))
    print(f"worst disagreement with batman: {worst * 1e6:.1f} ppm")
    return worst
def log_likelihood_ld(params, phase, flux, flux_err, real_period_days):
    rp, a, inc, u1, u2, H = params # Unpack the parameters for the batman model
    if rp <= 0 or rp > 0.5 or a <= 1.0 or inc <= 0 or inc > 90 or u1 < 0 or u2 < 0 or (u1 + u2) >= 1.0 or H <= 0:
        return -np.inf  # Return negative infinity for invalid values
 
    model_flux = limbdark_model(phase, rp, a, inc, u1, u2, H, real_period_days)
    residuals = flux - model_flux
    chi_squared = np.sum((residuals / flux_err) ** 2)
    log_likelihood_value = -0.5 * chi_squared
    return log_likelihood_value

 
def log_likelihood_ld(params, phase, flux, flux_err, real_period_days):
    """[NEW] Same chi-squared, limb-darkened model."""
    rp, a, inc, u1, u2, H = params
    if real_period_days is None:
        raise ValueError("limb-darkened fit needs real_period_days "
                         "(for the exposure-smearing correction)")
    model_flux = limbdark_model(phase, rp, a, inc, u1, u2, H, real_period_days)
    residuals = flux - model_flux
    return -0.5 * np.sum((residuals / flux_err) ** 2) 
    
def log_prior_ld(params):
    """[NEW] Uniform priors plus the two constraints that actually matter."""
    rp, a, inc, u1, u2, H = params

    if not (0.0 < rp < 0.5):
        return -np.inf
    if not (1.0 < a < 100.0):
        return -np.inf
    if not (0.0 < inc <= 90.0):
        return -np.inf
    if not (0.5 < H < 1.5):
        return -np.inf

    # GEOMETRY. With 'a' free, most of the prior volume describes orbits
    # where the planet never crosses the disc at all. There the likelihood
    # is perfectly flat, so a random walk that wanders in has nothing
    # pulling it back out. Requiring an actual transit is what keeps the
    # chain in the region where the data has something to say.
    b = a * np.cos(np.radians(inc))
    if b >= 1.0 + rp:
        return -np.inf

    # PHYSICALITY. u1, u2 must keep the intensity positive and decreasing
    # outward across the whole disc. This triangle is the standard
    # condition for the quadratic law.
    if not (u1 >= 0.0 and u1 + u2 < 1.0 and u1 + 2.0 * u2 > 0.0):
        return -np.inf

    return 0.0
    
def plot_posterior_ld(chain, burn_in=2000):
    chain_burned = chain[burn_in:] # Discard the burn-in samples
    fig, axes = plt.subplots(2, 3, figsize=(14, 8)) # Create a 2x3 grid of subplots
    param_names = ['rp', 'a', 'inc', 'u1', 'u2', 'H']
    for i, (ax, name) in enumerate(zip(axes.flatten(), param_names)):
        ax.hist(chain_burned[:, i], bins=30, density=True, alpha=0.7)
        median = np.median(chain_burned[:, i])
        lower = np.percentile(chain_burned[:, i], 16)
        upper = np.percentile(chain_burned[:, i], 84)
 
        ax.axvline(median, color='r', linestyle='--', lw=2, label=f'Median: {median:.4f}')
        ax.axvline(lower, color='orange', linestyle='--', lw=1.5, alpha=0.7)
        ax.axvline(upper, color='orange', linestyle='--', lw=1.5, alpha=0.7)
 
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Posterior: {name}')
        ax.legend()
    plt.tight_layout()
    plt.savefig("ld_mcmc_posterior.png", dpi=130)
    plt.show()
    print("Posterior Summary:")
    print("\n" + "=" * 50)
    print("LIMB-DARKENING MCMC RESULTS (after burn-in)")
    print("=" * 50)
    for i, name in enumerate(param_names):
        median = np.median(chain_burned[:, i])
        lower = np.percentile(chain_burned[:, i], 16)
        upper = np.percentile(chain_burned[:, i], 84)
        print(f"{name:8s} = {median:.6f} + {upper - median:.6f} - {median - lower:.6f}")

def plot_limbdark_fit(phase, flux_data, chain, real_period_days, burn_in=2000, trapezoid_chain=None):
    # Named plot_limbdark_fit rather than plot_trapezoid, since this plots
    # the limb-darkened (batman) fit, not a trapezoid -- see note below.
    chain_burned = chain[burn_in:]
    rp, a, inc, u1, u2, H = np.median(chain_burned, axis=0)
    model = limbdark_model(phase, rp, a, inc, u1, u2, H, real_period_days)
    residuals = flux_data - model
    order = np.argsort(phase)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
 
    axes[0].scatter(phase[order], flux_data[order], s=4, alpha=0.4, label='Data')
    axes[0].plot(phase[order], model[order], 'r-', lw=2, label='Limb-darkened model (batman)')
 
    if trapezoid_chain is not None:
        import Trapeziod_Algrothem  # local import: avoids a circular import, since
                                     # Trapeziod_Algrothem now also imports this module
        trap_burned = trapezoid_chain[burn_in:]
        Ht, depth, T_dur, T_flat = np.median(trap_burned, axis=0)
        trap_model = Trapeziod_Algrothem.trapezoid_model(phase, Ht, depth, T_dur, T_flat)
        axes[0].plot(phase[order], trap_model[order], 'g--', lw=1.5, alpha=0.8, label='Trapezoid model')
 
    axes[0].set_ylabel('Flux')
    axes[0].set_title(f'Best-Fit Limb-Darkened Model (rp={rp:.4f}, a={a:.2f}, '
                       f'inc={inc:.2f}, u1={u1:.3f}, u2={u2:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
 
    axes[1].scatter(phase[order], residuals[order], s=4, alpha=0.4, color='gray')
    axes[1].axhline(0, color='k', lw=1)
    axes[1].set_xlabel('Phase')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals: ingress/egress bowl shape should be reduced vs. the trapezoid fit')
    axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("ld_best_fit.png", dpi=130)
    plt.show()
 
