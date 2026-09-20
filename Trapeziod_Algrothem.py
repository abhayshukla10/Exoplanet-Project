from unicodedata import name

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from Transit_Project_project.Transit_Project import LimbDarkening_Algrothem
from Transit_Project_project.Transit_Project import LimbDarkening_Algrothem
import batman
KEPLER_LONG_CADENCE_DAYS = 29.4 / 60.0 / 24.0  # ~0.0204 d, one Kepler long-cadence exposure
def trapezoid_model(phase, H, depth, T_dur, T_flat,t0=0.0):
    ingress_duration = (T_dur - T_flat) / 2.0 # downward slope duration
    engress_duration = (T_dur - T_flat) / 2.0 # upward slope duration
    x = np.abs(phase - t0 + 0.5) % 1 - 0.5 # Calculate the absolute phase difference from the transit center
    
    half_duration = T_dur / 2.0 # Half of the total transit duration
    half_flat = T_flat / 2.0 # Half of the flat bottom duration
    ingress = half_duration - half_flat # Start of the ingress phase
    
    if ingress <= 0:
        return np.where(x < half_duration, H - depth, H) # Return a flat model if ingress is non-positive
    return H - depth * np.clip((half_duration - x) / ingress, 0.0, 1.0)

def log_likelihood_ta(params, phase, flux, flux_err):
    H, depth, T_dur, T_flat = params # Unpack the parameters
    if depth <= 0 or depth > H or T_flat < 0 or T_flat > T_dur or T_dur <= 0 or T_dur > 1 or H <= 0:
        return -np.inf  # Return negative infinity for invalid  values
 
    model_flux = trapezoid_model(phase, H, depth, T_dur, T_flat, t0=0.0) # Generate the model flux using the trapezoid model
    residuals = flux - model_flux # Calculate the residuals between the observed flux and the model flux
    chi_squared = np.sum((residuals / flux_err) ** 2)
    log_likelihood_value = -0.5 * chi_squared
    return log_likelihood_value
# Make sure bace line flux is positive and less than 1.2, depth is less than 0.5, T_flat is less than T_dur, and T_dur is positive and less than 0.5
def log_prior_ta(params):
    H, depth, T_dur, T_flat = params # Unpack the parameters
    if 0 < depth < 0.5 and 0 <= T_flat <= T_dur and 0.01 < T_dur < 0.5 and 0 < H < 1.20:
        return 0.0  # Uniform prior (log(1) = 0)
    else:
        return -np.inf  # Return negative infinity for invalid values
def log_posterior(params, phase, flux, flux_err, real_period_days=None):
    n_params = len(params)
    if n_params == 4:
        lp = log_prior_ta(params) # Calculate the log prior
        if not np.isfinite(lp): # Check if the log prior is finite
            return -np.inf  # Return negative infinity for invalid values
        ll = log_likelihood_ta(params, phase, flux, flux_err) # Calculate the log likelihood
    elif n_params == 6:
        lp = LimbDarkening_Algrothem.log_prior_ld(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = LimbDarkening_Algrothem.log_likelihood_ld(params, phase, flux, flux_err, real_period_days)
    else:
        raise ValueError(f"Unexpected number of parameters: {n_params}")
    return lp + ll  # Return the sum of the log prior and log likelihood
def metropolis_hastings(initial_params, phase, flux, flux_err, step_sizes, n_sizes, n_steps=10000, verbose = True, real_peroid_days=None):
    n_params = len(initial_params) # Number of parameters
    chain = np.zeros((n_steps, n_params)) # Array with columns for each parameter and rows for each step
    current_params = initial_params.copy() # Set the current parameters to the initial parameters
    current_log_posterior = log_posterior(current_params, phase, flux, flux_err) # Calculate the log posterior for the current parameters
    n_accepted = 0 # Initialize the number of accepted steps
    for i in range(n_steps): # Loop over the number of steps
        proposal = current_params + np.random.normal(0, step_sizes) # add randon value to each parameter based on the step size
        proposal_log_posterior = log_posterior(proposal, phase, flux, flux_err) # Calculate the log posterior for the proposed parameters
        log_acceptance_ratio = proposal_log_posterior - current_log_posterior # Calculate the log acceptance ratio
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current_params = proposal  # ✓ New guess becomes current
            current_log_posterior = proposal_log_posterior
            n_accepted += 1  # Increment the number of accepted steps
        chain[i] = current_params  # Store the current parameters in the chain
        if (i+1)% 1000 == 0: # Print the progress every 1000 steps
            print(f"Step {i+1}/{n_steps}, Acceptance Rate: {n_accepted/(i+1)*100:.1f}%")
    acceptance_rate = n_accepted / n_steps # Calculate the acceptance rate
    return chain, acceptance_rate # Return the chain and the acceptance rate
def plot_trace_ta(chain, param_names=['H', 'depth', 'T_dur', 'T_flat']):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(chain[:, i], lw=0.5, alpha=0.7)
        ax.set_ylabel(name)
        ax.set_xlabel("Iteration")
        ax.set_title(f"Trace: {name}")
 
    plt.tight_layout()
    plt.savefig("mcmc_trace.png", dpi=130)
    plt.show()
def plot_posterior_ta(chain, burn_in=2000):
    chain_burned = chain[burn_in:] # Discard the burn-in samples
    fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # Create a 2x2 grid of subplots
    param_names = ['H', 'depth', 'T_dur', 'T_flat'] # Parameter names
    for i, (ax, name) in enumerate(zip(axes.flatten(), param_names)):
        ax.hist(chain_burned[:, i], bins=30, density=True, alpha=0.7)
        median = np.median(chain_burned[:, i]) # Calculate the median of the parameter
        lower = np.percentile(chain_burned[:, i], 16) # Calculate the 16th percentile
        upper = np.percentile(chain_burned[:, i], 84) # Calculate the 84th percentile
 
        ax.axvline(median, color='r', linestyle='--', lw=2, label=f'Median: {median:.4f}')
        ax.axvline(lower, color='orange', linestyle='--', lw=1.5, alpha=0.7)
        ax.axvline(upper, color='orange', linestyle='--', lw=1.5, alpha=0.7)
 
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Posterior: {name}')
        ax.legend()
    plt.tight_layout()
    plt.savefig("mcmc_posterior.png", dpi=130)
    plt.show()
    print("Posterior Summary:")
    print("\n" + "=" * 50)
    print("MCMC RESULTS (after burn-in)")
    print("=" * 50)
    for i, name in enumerate(param_names):
        median = np.median(chain_burned[:, i])
        lower = np.percentile(chain_burned[:, i], 16)
        upper = np.percentile(chain_burned[:, i], 84)
        print(f"{name:8s} = {median:.6f} + {upper - median:.6f} - {median - lower:.6f}")
def plot_trapezoid(phase, flux_data, chain, burn_in=2000):
    chain_burned = chain[burn_in:]
    best_params = np.median(chain_burned, axis=0)
    H, depth, T_dur, T_flat = best_params
    model = trapezoid_model(phase, H, depth, T_dur, T_flat)
    residuals = flux_data - model
    order = np.argsort(phase)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
 
    axes[0].scatter(phase[order], flux_data[order], s=4, alpha=0.4, label='Data')
    axes[0].plot(phase[order], model[order], 'r-', lw=2, label='Trapezoidal Model')
    axes[0].set_ylabel('Flux')
    axes[0].set_title(f'Best-Fit Trapezoidal Model (H={H:.4f}, depth={depth:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
 
    axes[1].scatter(phase[order], residuals[order], s=4, alpha=0.4, color='gray')
    axes[1].axhline(0, color='k', lw=1)
    axes[1].set_xlabel('Phase')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals: Look for systematic structure (e.g., bowl shape from limb darkening)')
    axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("mcmc_best_fit.png", dpi=130)
    plt.show()
