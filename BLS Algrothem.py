import numpy as np
from abc import ABC, abstractmethod
import lightkurve as lk


class lc(ABC):
    """Light-curve wrapper with the usual cleaning and modeling operations."""

    def __init__(self, time, flux, flux_err=None):
        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if flux_err is None:
            flux_err = np.full_like(flux, np.nanstd(flux), dtype=float)
        else:
            flux_err = np.asarray(flux_err, dtype=float)
        if time.shape != flux.shape or flux.shape != flux_err.shape:
            raise ValueError("time, flux, and flux_err must have matching shapes")
        self.time = time
        self.flux = flux
        self.flux_err = flux_err

    @abstractmethod
    def remove_nans(self):
        """Return a cleaned copy with non-finite samples removed."""
        pass

    @abstractmethod
    def flatten(self):
        """Return a detrended light curve with long-term variations removed."""
        pass

    @abstractmethod
    def to_periodogram(self, method="bls", **kwargs):
        """Return a periodogram for the light curve."""
        pass


class LightCurveAdapter(lc):
    """Concrete light-curve adapter that uses Lightkurve when available."""

    def remove_nans(self):
        valid = np.isfinite(self.time) & np.isfinite(self.flux)
        if np.any(np.asarray(self.flux_err) is not None):
            valid &= np.isfinite(self.flux_err)
        return LightCurveAdapter(
            self.time[valid],
            self.flux[valid],
            self.flux_err[valid],
        )

    def flatten(self, window_length=101, polyorder=1):
        import scipy.signal as signal

        if len(self.flux) < window_length:
            window_length = max(5, len(self.flux) // 5)
            if window_length % 2 == 0:
                window_length += 1
        trend = signal.savgol_filter(self.flux, window_length=window_length, polyorder=polyorder, mode="interp")
        detrended = self.flux / trend
        return LightCurveAdapter(self.time, detrended, self.flux_err / trend)

    def to_periodogram(self, method="bls", **kwargs):
        if method.lower() != "bls":
            raise ValueError(f"Unsupported periodogram method: {method!r}")
        try:
            from lightkurve import LightCurve
        except ImportError as exc:  # pragma: no cover - only used if Lightkurve is absent.
            raise ImportError("lightkurve is required to build a BLS periodogram.") from exc

        lc_obj = LightCurve(time=self.time, flux=self.flux, flux_err=self.flux_err)
        return lc_obj.to_periodogram(method=method, **kwargs)


    #1.Define the Parameter Search Space:
    #       Before you can fit a model, you need to define the boundaries of your search. A simple "box" transit requires testing three distinct physical parameters.
    #       Identify what these three dimensions are, and set up arrays with reasonable physical bounds and step sizes for each.
period_grid = np.arange(0.5, 10, 0.001)# days bewtween dips
duration_grid = np.arange(0.01, 0.5, 0.005)#Fraction where dip lasts
phase_grid = np.arange(0, 1, 0.01)# Where does the dip start?
print(f"Period grid: {len(period_grid)} periods")
print(f"Duration grid: {len(duration_grid)} durations")
print(f"Phase grid: {len(phase_grid)} phases")
print(f"Total grid combinations: {len(period_grid)*len(duration_grid)*len(phase_grid)}")
#2.Phase-Folding and Dimensionality Reduction:
#Write a function that phase-folds your time-series data for any given test configuration.
#Running a least-squares optimization on hundreds of thousands of raw data points for every single grid combination will stall your computer. How can you mathematically 
# compressor group your phase-folded data to speed up the fitting process without losing the physical transit signal? Implement this reduction.
# Hint: think about the resolution of your optimization here
def reduce_data(t, flux, period, n_bins):# declacre a function that takes in time, flux, period, and number of bins
    Phase = ((t-t[0] )% period) / period # Each time becomes a phase between 0 and 1
    bin_indices = np.floor(Phase * n_bins).astype(int) # Assign each phase to a bin like there 100 values between 0 and 1 label them 
    bin_indices[bin_indices == n_bins] = n_bins - 1 # Make sure the last bin is included
    bin_means = np.zeros(n_bins)# array of Zeros that are just placeholders for the mean of each bin
    bin_counts = np.zeros(n_bins) # array of Zeros that are just placeholders for the number of points in each bin
    for i in range(n_bins):
        points_in_bin = flux[bin_indices == i]# Get the flux values that fall into the current bin
        bin_counts[i] = len(points_in_bin) # Count how many points are in the current bin
        if len(points_in_bin) > 0:# Check if there are points in the bin
            bin_means[i] = np.mean(points_in_bin)#Return the mean of each bin
        else:
            bin_means[i] = np.nan # Handle empty bins
            
    return bin_means, bin_counts#Return the mean of each bin and the number of points in each bin
#3.The Core Optimization Engine:
#Write the core loop. For a given folded and compressed configuration, construct a two-state "box" (upside down tophat) model representing the out-of-transit baseline and the in-transit dip.
#Define a statistical metric (like a goodness-of-fit or power spectrum calculation) to evaluate exactly how well this theoretical box fits your compressed data.

def find_best_box_indices(bin_means,bin_counts, min_width, max_width):# Function that takes in the binned means, counts, and min/max widths for the box model
    n_bins = len(bin_means) # Get the number of bins
    
    # ignore empty bins
    valid = ~np.isnan(bin_means) # Create a boolean array to identify valid bins
    x = np.where(valid,bin_means,0.0) # Replace NaN values with 0.0 for fitting
    w = np.where(valid,bin_counts,0.0) # Replace NaN values with 0.0 for fitting
    total_weight = np.sum(w) # Calculate the total weight of valid bins
    best_sr = -np.inf # Initialize the best signal-to-noise ratio
    best_i1, best_i2 = 0, min_width # Initialize the best indices for the box model
    for width in range(min_width, max_width + 1): # Loop through possible widths for the box model
        for i1 in range(0,n_bins - width): # Loop through possible starting indices for the box model
            i2 = i1 + width # Calculate the ending index for the box model
            s = np.sum(x[i1:i2]*w[i1:i2]) # sum of the weighted flux values within the box
            r = np.sum(w[i1:i2]) / total_weight # fraction of the total weight within the box
            r = min(max(r, 1e-6), 1-1e-6) # ensure the signal is within a reasonable range to avoid numerical issues
            sr = np.sqrt((s**2)/(r*(1-r))) # Calculate the signal-to-noise ratio (SNR) using the formula derived from the binomial distribution
            if sr > best_sr: # Check if this SNR is better than the best found so far
                best_sr = sr # Update the best SNR make sure to make sure s in negative for best SNR
                best_i1, best_i2 = i1, i2 # Update the best indices for the box model
                
    return best_sr, best_i1, best_i2 # Return the best SNR and the corresponding indices for the box model




def find_best_period(t,flux,period_grid,duration_grid,n_bins):# Function that takes in time, flux, period grid, duration grid, and number of bins
    flux = flux - np.mean(flux) # Normalize the flux by subtracting the mean
    min_width = max(1, int(duration_grid[0] * n_bins)) # Calculate the minimum width of the box model in bins
    max_width = max(min_width, int(duration_grid[-1] * n_bins)) # Calculate the maximum width of the box model in bins
    sr_values = np.zeros(len(period_grid)) # Initialize an array to store the SNR values for each period
    windows = []#
    for k, period in enumerate(period_grid): # Loop through each period in the grid
        bin_means, bin_counts = reduce_data(t, flux, period, n_bins) # pyright: ignore[reportUndefinedVariable] # Phase-fold and bin the data for the current period
        sr, i1, i2 = find_best_box_indices(bin_means, bin_counts, min_width, max_width) # Find the best-fitting box model for the binned data
        sr_values[k] = sr # Store the SNR value for the current period
        windows.append((i1,i2)) # Store the indices of the best-fitting box model for later analysis
        best_index = np.argmax(sr_values)
    best_index = np.argmax(sr_values)
    best_period = period_grid[best_index]
    best_i1, best_i2 = windows[best_index]
 
    return sr_values, best_period, best_i1, best_i2 #return the SNR values, best period, and indices of the best-fitting box model

def reconstruct_box_model(t, flux, weights, period, i1, i2, n_bins):
    """Fold the ORIGINAL (uncompressed) data on best_period and build
    the best-fit L (in-transit) / H (out-of-transit) box model on top
    of it, so we can see how well the box matches the real data."""
    flux0 = flux - np.average(flux, weights=weights)# center the flux around zero using weighted average
    phase = ((t - t[0]) / period) % 1.0# Converting it to a fraction how far along the data am I
 
    lo, hi = i1 / n_bins, i2 / n_bins#convet to fractions showing how far on the orbit am I
    in_transit = (phase >= lo) & (phase < hi)# split into transit and non transit data
 
    L_hat = np.average(flux0[in_transit], weights=weights[in_transit])# find average of the flux during transit using weighted average
    H_hat = np.average(flux0[~in_transit], weights=weights[~in_transit])# find average of the flux outside transit using weighted average
 
    model = np.where(in_transit, L_hat, H_hat)# create a model that is L_hat during transit and H_hat outside transit
    residual = flux0 - model# ADD the model to the original flux to see how well it fits
 
    return phase, flux0, model, residual, (lo, hi), (L_hat, H_hat)
#Download the data for KIC 8462852 (Tabby's Star) using the Lightkurve library. This star is known for its unusual light curve, which has been the subject of much study and speculation.
search_result = lk.search_lightcurve('KIC 8462852')
lc = search_result[0].download()
#clean the data by removing NaN values and normalizing the flux. This step is crucial for accurate analysis, as NaN values can skew results and normalization helps in comparing different datasets.
lc = lc.remove_nans()
remove
lc = lc.flatten()
#Gather the time, flux, and weights from the light curve object. These arrays will be used in the subsequent analysis steps.
t = lc.time.value
flux = lc.flux.value
flux_err = lc.flux_err.value
weights = 1.0 / lc.flux_err**2
n_bins = 200
test_grid = np.arange(0.01, 10, 0.005)
# Light kurve BLS algorithm
print(f"Downloaded {len(t)} points , spanning {t.max()-t.min():.1f} days")
# lightkurve Bls algrothem
print("Running lightkurve's built-in BLS...")
bls = lc.to_periodogram(method='bls')
best_period_lk = bls.period_at_max_power.value
print(f"lightkurve's period: {best_period_lk:.6f} d\n")

# My BLS algrothem

# 1) reduce_data -- fold + bin on ONE trial period, just to prove it runs
bin_means, bin_counts = reduce_data(t, flux, period=test_grid[0], n_bins=n_bins)
print(f"reduce_data output: bin_means shape={bin_means.shape}, "
      f"bin_counts shape={bin_counts.shape}")
# 2) find_best_box_indices -- fit a box to that one folded/binned result
min_width = max(1, int(duration_grid[0] * n_bins))
max_width = max(min_width, int(duration_grid[-1] * n_bins))
sound_to_noise, i1, i2 = find_best_box_indices(bin_means, bin_counts, min_width, max_width)
print(f"find_best_box_indices output: SR={sound_to_noise:.4f}, window=bins {i1}-{i2}")
# 3) find_best_period -- run the full grid search over all trial periods
sr_values, best_period, best_i1, best_i2 = find_best_period(t, flux, period_grid, duration_grid, n_bins)
print(f"find_best_period output: best_period={best_period:.6f} d, window=bins {best_i1}-{best_i2}")
# 4) reconstruct_box_model -- fold the original data on the best period and overlay the best-fit box model
phase, flux0, model, residual, (lo, hi), (L_hat, H_hat) = reconstruct_box_model(t, flux, weights, best_period, best_i1, best_i2, n_bins)    
print(f"reconstruct_box_model output: L_hat={L_hat:.6f}, H_hat={H_hat:.6f}, "
      f"mean|residual|={np.mean(np.abs(residual)):.6f}")
#compare the results from my BLS algorithm with those from Lightkurve's built-in BLS implementation. 
bls = lc.to_periodogram(method='bls')
print(f"\nMy engine's period:  {best_period:.6f} d")
print(f"lightkurve's period: {bls.period_at_max_power.value:.6f} d")
print(f"Difference: {abs(best_period - bls.period_at_max_power.value):.6f} d" )
# ============================================================
# EXAMINE THE RESIDUALS: WHERE and WHY does the box model fail?
# ============================================================
import matplotlib.pyplot as plt
 
order = np.argsort(phase)
 
fig, axes = plt.subplots(2, 1, figsize=(9, 8))
 
axes[0].scatter(phase[order], flux0[order], s=4, alpha=0.4, label="raw data")
axes[0].plot(phase[order], model[order], color="r", lw=2, label="box model")
axes[0].set_xlabel("Phase")
axes[0].set_ylabel("Flux (mean-subtracted)")
axes[0].set_title(f"Folded light curve + box model (P = {best_period:.4f} d)")
axes[0].legend()
 
axes[1].scatter(phase[order], residual[order], s=4, alpha=0.4, color="gray")
axes[1].axhline(0, color="k", lw=1)
axes[1].axvline(lo, color="r", ls="--", lw=1, label="transit window edges")
axes[1].axvline(hi, color="r", ls="--", lw=1)
axes[1].set_xlabel("Phase")
axes[1].set_ylabel("Residual (data - model)")
axes[1].set_title("Residuals -- look for structure right at the transit edges")
axes[1].legend()
 
plt.tight_layout()
plt.savefig("bls_residuals.png", dpi=130)
plt.show()
 
# quantify WHERE the mismatch is worst: near the transit edges, or elsewhere?
edge_width = 0.01  # phase units, a narrow band around each edge
near_edges = ((phase >= lo - edge_width) & (phase <= lo + edge_width)) | \
             ((phase >= hi - edge_width) & (phase <= hi + edge_width))
 
print(f"\nMean |residual| near ingress/egress (transit edges): "
      f"{np.mean(np.abs(residual[near_edges])):.6f}")
print(f"Mean |residual| everywhere else:                      "
      f"{np.mean(np.abs(residual[~near_edges])):.6f}")
print("Lindarkening is the evil culprit!")