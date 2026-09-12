import numpy as np
import matplotlib.pyplot as plt
import batman

KEPLER_LONG_CADENCE_DAYS = 29.4 / 60.0 / 24.0 

def limbdark_model(phase, rp, a, inc, u1, u2, H, real_period_days,
                    ecc=0.0, w=90.0, exp_time_days=KEPLER_LONG_CADENCE_DAYS,
                    supersample_factor=7):
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = 1.0
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.limb_dark = "quadratic"
    params.u = [u1, u2]
 
    exp_time_phase = exp_time_days / real_period_days
    m = batman.TransitModel(params, phase,
                             supersample_factor=supersample_factor,
                             exp_time=exp_time_phase)
    return H * m.light_curve(params)
 
  
def log_likelihood_ld(params, phase, flux, flux_err, real_period_days):
    rp, a, inc, u1, u2, H = params # Unpack the parameters for the batman model
    if rp <= 0 or rp > 0.5 or a <= 1.0 or inc <= 0 or inc > 90 or u1 < 0 or u2 < 0 or (u1 + u2) >= 1.0 or H <= 0:
        return -np.inf  # Return negative infinity for invalid values
 
    model_flux = limbdark_model(phase, rp, a, inc, u1, u2, H, real_period_days)
    residuals = flux - model_flux
    chi_squared = np.sum((residuals / flux_err) ** 2)
    log_likelihood_value = -0.5 * chi_squared
    return log_likelihood_value

 
def log_prior_ld(params):
    rp, a, inc, u1, u2, H = params
    if (0 < rp < 0.5 and 1.0 < a < 100.0 and 60.0 < inc <= 90.0
            and 0 <= u1 <= 1 and 0 <= u2 <= 1 and (u1 + u2) < 1.0
            and 0.5 < H < 1.5):
        return 0.0  # Uniform prior (log(1) = 0)
    else:
        return -np.inf  
    
def plot_trace_ld(chain, param_names=['rp', 'a', 'inc', 'u1', 'u2', 'H']):
    fig, axes = plt.subplots(6, 1, figsize=(10, 12))
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(chain[:, i], lw=0.5, alpha=0.7)
        ax.set_ylabel(name)
        ax.set_xlabel("Iteration")
        ax.set_title(f"Trace: {name}")
 
    plt.tight_layout()
    plt.savefig("ld_mcmc_trace.png", dpi=130)
    plt.show()
    
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
 