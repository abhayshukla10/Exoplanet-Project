# TODO: Update the main function to your needs or remove it.
import BLS_Algrothem
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import name
import Trapeziod_Algrothem
import batman 

def main() -> None:
    print("Start coding in Python today!")


if __name__ == "__main__":
    # This is where I actually plot the trapezoid model and the residuals. The top panel shows the observed flux data along with the best-fit trapezoidal model, while the bottom panel displays the residuals, which should ideally be randomly distributed around zero if the model is a good fit.
    # Download and clean data
    search_result = lk.search_lightcurve('Kepler-186')
    lc = search_result[0].download()
    lc= lc.remove_nans()
    lc= lc.remove_outliers(sigma_upper = 5, sigma_lower=100)
    lc= lc.flatten()
    # Obtain parameters for the trapezoid model
    t = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    weights = 1.0 / lc.flux_err**2

    print(f"Downloaded {len(t)} points,within these {t.max()-t.min():.1f} days,")
    n_bins = 200 # number of bins for the BLS algorithm
    period_grid = np.arange(0.5, 10, 0.001) # period grid for the BLS algorithm
    duration_grid = np.arange(0.01, 0.5, 0.005) # duration grid for the BLS algorithm
    #gets array fobest signal to noise ratio, best period, and the indices of the best duration in the grid
    sr_values, best_period, best_i1, best_i2 = BLS_Algrothem.find_best_period(t, flux, period_grid, duration_grid, n_bins)

    print(f"Your BLS: best_period={best_period:.6f} d, window=bins {best_i1}-{best_i2}")#
    # getting phase, flux mean -weight to center around 0, the box model, the resedual which is the difference between the flux and the box model, the low and high flux values, and the low and high flux values of the box model 
    phase, flux0, box_model, residual, (lo, hi), (L_hat, H_hat) = Trapeziod_Algrothem.reconstruct_box_model(t, flux, weights, best_period, best_i1, best_i2, n_bins)
    # Getting transit duration and flat duration guesses based on the best indices from the BLS algorithm. The transit duration is estimated as the difference between the best indices divided by the number of bins, and the flat duration is assumed to be 60% of the transit duration. The baseline flux (H) and depth are also estimated from the low and high flux values.
    T_dur_guess = (best_i2 - best_i1) / n_bins
    T_flat_guess = T_dur_guess * 0.6
    # checking if the depth guess is non-positive, and if so, setting it to a small positive value (0.001) to avoid issues in the model fitting process. The phase is also centered around zero for better visualization and analysis.
    H_guess = H_hat
    depth_guess = H_hat - L_hat
    if depth_guess <= 0:
        depth_guess = 0.001  # Set a small positive value for depth if it's non-positive 
    #find center of dip    
    center = 0.5 * (lo + hi)
    #Shifts mid point to center and not where bls found it
    phase_centered = (phase - center + 0.5) % 1.0 - 0.5
    # Array of initial parameters for the trapezoid model fitting, including the baseline flux (H), depth, transit duration (T_dur), and flat duration (T_flat). 
    initial_params = [H_guess, depth_guess, T_dur_guess, T_flat_guess]
    # The step sizes for the Metropolis-Hastings algorithm are also defined for each parameter to control the proposal distribution during the sampling process.
    step_sizes = [0.0005, 0.0005, 0.002, 0.002]
    
    print("Initial guess for trapezoid fit:")
    print(f"  H={H_guess:.6f}, depth={depth_guess:.6f}, "
          f"T_dur={T_dur_guess:.6f}, T_flat={T_flat_guess:.6f}")
        
    print(f"Final Acceptance rate: {acceptance_rate*100:.1f}%")
    
    Trapeziod_Algrothem.plot_trace(chain)
    Trapeziod_Algrothem.plot_posterior(chain,burn_in=2000)
    Trapeziod_Algrothem.plot_trapezoid(phase_centered, flux0, flux_err, chain, burn_in=2000)
    
    trap_burned = chain[2000:] 
    H_med, depth_med, T_dur_med, T_flat_med = np.median(trap_burned, axis=0)

    rp_guess = np.sqrt(depth_med)  
    
    a_guess = 15.0
    
    ing_guess = 89.0
    
    u1_guess, u2_guess = 0.3, 0.2
    H_guess_id = 1.0
    
    chain, acceptance_rate = Trapeziod_Algrothem.metropolis_hastings(initial_params,phase_centered,flux0,flux_err,step_sizes,n_sizes=len(initial_params), n_steps=10000,verbose=True)

    
    print(f"Final Acceptance rate: {acceptance_rate*100:.1f}%")
    
    Trapeziod_Algrothem.plot_trace(chain)
    Trapeziod_Algrothem.plot_posterior(chain, burn_in=2000)
    Trapeziod_Algrothem.plot_trapezoid(phase_centered, flux0, chain, burn_in=2000) 
    
    trap_burned = chain[2000:]
    H_med, depth_med, T_dur_med, T_flat_med = np.median(trap_burned, axis=0)

    rp_guess = np.sqrt(depth_med) 
    a_guess = 15.0 
    inc_guess = 89.0 
    u1_guess, u2_guess = 0.3, 0.2
    
    H_guess_ld = 1.0 
    
    initial_params_ld = [rp_guess, a_guess, inc_guess, u1_guess, u2_guess, H_guess_ld]
    step_sizes_ld = [0.001, 0.5, 0.2, 0.02, 0.02, 0.0005]
    
    print("\nInitial guess for limb-darkening fit:")
    print(f"  rp={rp_guess:.6f}, a={a_guess:.2f}, inc={inc_guess:.2f}, "
          f"u1={u1_guess:.3f}, u2={u2_guess:.3f}, H={H_guess_ld:.6f}")
    
    chain_ld, acceptance_rate_ld = Trapeziod_Algrothem.metropolis_hastings(initial_params_ld, phase_centered, flux, flux_err, step_sizes_ld, n_sizes=len(initial_params_ld), real_period_days=best_period, n_steps=10000, verbose=True)
    
    print(f"Limb-darkening fit acceptance rate: {acceptance_rate_ld*100:.1f}%")
    
    Trapeziod_Algrothem.plot_trace(chain_ld)
    Trapeziod_Algrothem.plot_posterior(chain_ld, burn_in=2000)
    Trapeziod_Algrothem.plot_trapezoid(phase_centered, flux, chain_ld, burn_in=2000,real_period_days=best_period,trapezoid_chain=chain)