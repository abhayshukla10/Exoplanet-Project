from unicodedata import name

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import batman
KEPLER_LONG_CADENCE_DAYS = 29.4 / 60.0 / 24.0  # ~0.0204 d, one Kepler long-cadence exposure
def trapezoid_model(phase, H, depth, T_dur, T_flat,t0=0.0):
    ingress_duration = (T_dur - T_flat) / 2.0 # downward slope duration
    engress_duration = (T_dur - T_flat) / 2.0 # upward slope duration
    
    t1 = t0 - T_dur / 2.0 # Transit start time
    t2 = t1 + ingress_duration #Ingress end time, Flat start time
    t3 = t2 + T_flat # Flat end time, Egress start time 
    t4 = t0 + T_dur / 2.0 - engress_duration #Transit end time
    
    phase_wrapped = phase % 1.0
    
    flux_model = np.full_like(phase_wrapped, H, dtype=float)  
    # ingress phase
    if ingress_duration > 0:
        ingress_mask = (phase_wrapped >= t1) & (phase_wrapped < t2)
        flux_model[ingress_mask] = H - depth * (phase_wrapped[ingress_mask] - t1) / ingress_duration
    # flat phase
    flat_mask = (phase_wrapped >= t2) & (phase_wrapped < t3)
    flux_model[flat_mask] = H - depth
    #
    if engress_duration > 0:
        engress_mask = (phase_wrapped >= t3) & (phase_wrapped < t4)
        flux_model[engress_mask] = H - depth * (t4 - phase_wrapped[engress_mask]) / engress_duration

    return flux_model

def log_likelihood(params, phase, flux, flux_err):
    
    if len(params) == 4:
        H, depth, T_dur, T_flat = params # Unpack the parameters
        if depth <=0 or depth > H or T_flat < 0 or T_flat > T_dur or T_dur <= 0 or T_dur > 1 or H <= 0:
            return -np.inf  # Return negative infinity for invalid  values
        model_flux = trapezoid_model(phase, H, depth, T_dur, T_flat, t0=0.0) # Generate the model flux using the trapezoid model
    elif len(params) == 6:
        rp, a, inc, u1, u2, H = params # Unpack the parameters for the batman model
        if rp <= 0 or rp > 0.5 or a <= 1.0 or inc <= 0 or inc > 90 or u1 < 0 or (u1+u2) > 1 or u2 < 0 or H <= 0:
            return -np.inf  # Return negative infinity for invalid values
    residuals = flux - model_flux # Calculate the residuals between the observed flux and the model flux
    chi_squared = np.sum((residuals / flux_err) ** 2)
    log_likelihood_value = -0.5 * chi_squared
    return log_likelihood_value
# Make sure bace line flux is positive and less than 1.2, depth is less than 0.5, T_flat is less than T_dur, and T_dur is positive and less than 0.5
def log_prior(params):
    if len(params) == 4:
        H, depth, T_dur, T_flat = params # Unpack the parameters
        if 0 < depth < 0.5 and 0 <= T_flat <= T_dur and 0.01 < T_dur < 0.5 and 0 < H < 1.20:
            return 0.0  # Uniform prior (log(1) = 0)
        else:
            return -np.inf  # Return negative infinity for invalid values
    elif len(params == 6):
        rp, a, inc, u1, u2, H = params
        if(0 < rp < 0.5 and 1.0 < a < 100.0 and 60.0 < inc <= 90.0 and 0 <= u1 <= 1 and 0 <= u2 <= 1 and (u1 + u2) < 1.0 and 0.5 < H < 1.5):
            return 0.0
        else:
            return -np.inf
    else:
        raise ValueError(f"Unexpected number of parameters: {len(params)}")
def log_posterior(params, phase, flux, flux_err):
    lp = log_prior(params) # Calculate the log prior
    if not np.isfinite(lp): # Check if the log prior is finite
        return -np.inf  # Return negative infinity for invalid values
    ll = log_likelihood(params, phase, flux, flux_err) # Calculate the log likelihood
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
def plot_trace(chain, param_names=None):
    n_params = chain.shape[1]
    if param_names is None:
        if n_params == 4:
            param_names = ['H', 'depth', 'T_dur', 'T_flat']
        elif n_params == 6:
            param_names = ['rp', 'a', 'inc', 'u1', 'u2', 'H']
        else:
            raise ValueError(f"Unexpected number of parameters: {n_params}")
    fig, axes = plt.subplots(n_params,1, figsize=(10, 2*n_params)) 
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(chain[:, i], lw=0.5, alpha=0.7)
        ax.set_ylabel(name)
        ax.set_xlabel("Iteration")
        ax.set_title(f"Trace: {name}")
    
    plt.tight_layout()
    filename = "mcmc_trace"
    plt.savefig("mcmc_trace.png", dpi=130)
    plt.show()
def plot_posterior(chain, burn_in=2000):
    chain_burned = chain[burn_in:] # Discard the burn-in samples
    n_param = chain.shape[1]
    if param_names is None:
        if n_param == 4:
            param_names = ['H', 'depth', 'T_dur', 'T_flat']
        elif n_param == 6:
            param_names = ['rp', 'a', 'inc', 'u1', 'u2', 'H']
        else:
            raise ValueError(f"Unexpected number of parameters: {n_param}")
 
    rows, cols = (2, 2) if n_param == 4 else (2, 3)
    fig, axes = plt.subplots(2, 2, figsize=(10 if n_param == 4 else 14,8)) # Create a 2x2 grid of subplots
    for i, (ax, name) in enumerate(zip(axes.flatten(), param_names)):
        ax.hist(chain_burned[:, i], bins=30, density=True, alpha=0.7)
        median = np.median(chain_burned[:, i]) # Calculate the median of the parameter
        lower = np.percentile(chain_burned[:, i], 16) # Calculate the 16th percentile
        upper = np.percentile(chain_burned[:, i], 84) # Calculate the 84th percentile
        
        ax.axvline(median, color='r', linestyle='--',lw=2, label=f'Median: {median:.4f}')
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
    print("\n"+"="*50)
    print("MCMC RESULTS (after burn-in)")
    print("="*50)
    for i, name in enumerate(param_names):
        median = np.median(chain_burned[:, i])
        lower = np.percentile(chain_burned[:, i], 16)
        upper = np.percentile(chain_burned[:, i], 84)
        print(f"{name:8s} = {median:.6f} + {upper - median:.6f} - {median - lower:.6f}")
def plot_trapezoid(phase, flux_data, chain, burn_in=2000, real_period_days = None, trapezoid_chain = None):
    n_params = chain.shape[1]
    chain_burned = chain[burn_in:]
    best_params = np.median(chain_burned, axis=0)
    order = np.argsort(phase)
 
    if n_params == 4:
        H, depth, T_dur, T_flat = best_params
        model = trapezoid_model(phase, H, depth, T_dur, T_flat)
        title = f'Best-Fit Trapezoidal Model (H={H:.4f}, depth={depth:.4f})'
        model_label = 'Trapezoidal Model'
        filename = "mcmc_best_fit.png"
    elif n_params == 6:
        if real_period_days is None:
            raise ValueError("real_period_days is required for the limb-darkening (6-param) case")
        rp, a, inc, u1, u2, H = best_params
        model = limbdark_model(phase, rp, a, inc, u1, u2, H, real_period_days)
        title = (f'Best-Fit Limb-Darkened Model (rp={rp:.4f}, a={a:.2f}, '
                 f'inc={inc:.2f}, u1={u1:.3f}, u2={u2:.3f})')
        model_label = 'Limb-darkened model (batman)'
        filename = "ld_best_fit.png"
    else:
        raise ValueError(f"Unexpected number of parameters: {n_params}")
    residuals = flux_data - model
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].scatter(phase[order], flux_data[order], s=4, alpha=0.4, label='Data')
    axes[0].plot(phase[order], model[order], 'r-', lw=2, label='Trapezoidal Model')
    
    if n_params == 6 and trapezoid_chain is not None:
        trap_burned = trapezoid_chain[burn_in:]
        Ht, depth, T_dur, T_flat = np.median(trap_burned, axis=0)
        trap_model = trapezoid_model(phase, Ht, depth, T_dur, T_flat)
        axes[0].plot(phase[order], trap_model[order], 'g--', lw=1.5, alpha=0.8, label='Trapezoid model')

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
def limbdark_model(phase, rp, a, inc, u1, u2, H, real_period_days,ecc=0.0, w=90.0, exp_time_days=KEPLER_LONG_CADENCE_DAYS, supersample_factor=7):
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