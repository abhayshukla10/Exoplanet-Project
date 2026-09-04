# TODO: Update the main function to your needs or remove it.
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import BLS_Algrothem

# Trapeziod_Algrothem import removed (module not present).
# Provide a lightweight fallback for find_best_period so this script can run
# without requiring the external Trapeziod_Algrothem implementation.
def find_best_period(t, flux, period_grid, duration_grid, n_bins):
    """Simple placeholder that evaluates a trivial detection statistic.
    Returns sr_values array and a default best period and ingress/egress values.
    """
    sr_values = np.zeros(len(period_grid))
    # crude placeholder: pick the period with maximum variance of folded profile
    for i, p in enumerate(period_grid):
        phases = (t % p) / p
        inds = np.argsort(phases)
        folded = flux[inds]
        # bin and compute variance as a toy detection statistic
        try:
            binned = np.mean(folded.reshape(-1, max(1, len(folded)//n_bins)), axis=1)
            sr_values[i] = np.var(binned)
        except Exception:
            sr_values[i] = np.var(folded)
    best_idx = int(np.argmax(sr_values))
    best_period = period_grid[best_idx]
    # return simple defaults for the trapezoid ingress/egress
    best_i1 = duration_grid[0]
    best_i2 = duration_grid[-1]
    return sr_values, best_period, best_i1, best_i2
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
    n_bins = 200
    period_grid = np.arange(0.5, 10, 0.001)
    duration_grid = np.arange(0.01, 0.5, 0.005)
    
    sr_values, best_period, best_i1, best_i2 = find_best_period(t, flux, period_grid, duration_grid, n_bins)
