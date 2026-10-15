from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import emcee

def trapezoid_model(phase, H, depth, T_dur, T_flat, t0=0.0):
    x = np.abs((phase - t0 + 0.5) % 1.0 - 0.5);

    half_dur, half_flat = T_dur / 2.0, T_flat / 2.0
    ingress = half_dur - half_flat

    # T_flat == T_dur: top-hat transit
    if ingress <= 0:
        return np.where(x <= half_dur, H - depth, H)

    return H - depth * np.clip((half_dur - x) / ingress, 0.0, 1.0)

def transit_model(t, t0, P, H, depth, T_dur, T_flat):

    t = np.asarray(t, dtype=float)

    if P <= 0 or T_dur <= 0 or T_flat < 0 or T_flat > T_dur:
        return np.full_like(t, np.nan)

    n = np.round((t - t0) / P)
    transit_center = t0 + n * P

    dt = t - transit_center
    x = np.abs(dt)

    half_dur = T_dur / 2.0
    half_flat = T_flat / 2.0
    ingress = half_dur - half_flat

    # T_flat == T_dur: top-hat transit
    if ingress <= 0:
        return np.where(x <= half_dur, H - depth, H)

    return H - depth * np.clip((half_dur - x) / ingress, 0.0, 1.0)

def log_likelihood_ta(params, t, flux, flux_err):
    t0, P, H, depth, T_dur, T_flat = params
    
    model_flux = transit_model(t,t0,P,H,depth,T_dur,T_flat)
    if not np.all(np.isfinite(model_flux)):
        return -np.inf
    residuals = flux - model_flux
    chi_squared = np.sum((residuals / flux_err) ** 2)
    return -0.5 * chi_squared


def log_prior_ta(params, t):
    t0, P, H, depth, T_dur, T_flat = params
    if (np.min(t) <= t0 <= np.max(t) and P > 0 and 0.5 < H < 1.5 and 0 < depth < H and 0 < T_dur < 0.5 and 0 <= T_flat <= T_dur):
        return 0.0;
    if P <= 0 or T_dur <= 0 or T_flat < 0 or T_flat > T_dur or depth <= 0 or depth > H or H <= 0:
            return -np.inf  # Return negative infinity for invalid values
    return -np.inf


def log_posterior(params, t, flux, flux_err):
    lp = log_prior_ta(params, t)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ta(params,t,flux,flux_err)

def run_mcmc(initial_params,t,flux,flux_err,n_walkers=32,n_steps=5000):
    initial_params = np.asarray(initial_params, dtype=float)
    ndim = len(initial_params)
    if n_walkers < 2 * ndim:
        raise ValueError(f"n_walkers must be at least {2 * ndim} for {ndim} parameters.")
    parameter_scales = np.array([
        1e-3,   # t0
        1e-4,   # P
        1e-4,   # H
        1e-5,   # depth
        1e-3,   # T_dur (days)
        1e-3    # T_flat (days)
    ])

    pos = initial_params + (parameter_scales * np.random.randn(n_walkers, ndim))

    for i in range(n_walkers):
        attempts = 0

        while not np.isfinite(
            log_posterior(pos[i], t, flux, flux_err)
        ):
            pos[i] = initial_params + (
                parameter_scales
                * np.random.randn(ndim)
            )
            attempts += 1
            if attempts > 1000:
                raise ValueError("Could not create valid initial walker positions.")

    sampler = emcee.EnsembleSampler( n_walkers,ndim,log_posterior,args=(t, flux, flux_err))
    sampler.run_mcmc(pos,n_steps,progress=True)
    return sampler

def plot_trace_ta(sampler,param_names=None):
    
    if param_names is None:
        param_names = ["t0","P","H","depth","T_dur","T_flat"]
    chain = sampler.get_chain()
    n_params = chain.shape[2]
    fig, axes = plt.subplots(n_params,1,figsize=(10, 12),sharex=True)

    if n_params == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(
        zip(axes, param_names)
    ):
        ax.plot(
            chain[:, :, i],
            alpha=0.3,
            lw=0.5
        )

        ax.set_ylabel(name)
        ax.set_title(f"Trace: {name}")

    axes[-1].set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()

def plot_posterior_ta(sampler,burn_in=1000,param_names=None):

    if param_names is None:
        param_names = ["t0","P","H","depth","T_dur","T_flat"]

    flat_chain = sampler.get_chain(discard=burn_in,flat=True)
    n_params = flat_chain.shape[1]
    fig, axes = plt.subplots(2,3,figsize=(14, 8))

    axes = axes.flatten()

    for i, name in enumerate(param_names):
        axes[i].hist(flat_chain[:, i],bins=30,density=True,alpha=0.7)

        median = np.median(flat_chain[:, i])
        lower = np.percentile(flat_chain[:, i],16)
        upper = np.percentile(flat_chain[:, i],84)

        axes[i].axvline(median, linestyle="--",lw=2,label=f"Median: {median:.6f}")
        axes[i].axvline(lower,linestyle="--",lw=1.5,alpha=0.7)

        axes[i].axvline(upper,linestyle="--",lw=1.5,alpha=0.7)

        axes[i].set_xlabel(name)
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"Posterior: {name}")
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_trapezoid(t,flux_data,chain,burn_in=1000):
    chain_burned = chain[burn_in:]
    t0, P, H, depth, T_dur, T_flat = np.median(chain_burned,axis=0)

    model = transit_model(t,t0,P,H,depth,T_dur,T_flat)
    residuals = flux_data - model
    order = np.argsort(t)
    fig, axes = plt.subplots(2,1,figsize=(12, 8))

    axes[0].scatter(t[order],flux_data[order],s=4,alpha=0.4,label="Data")

    axes[0].plot(t[order],model[order],lw=2,label="Time-domain trapezoid")

    axes[0].set_ylabel("Flux")
    axes[0].set_title(f"Time-Domain Trapezoid Fit "
                      f"(P={P:.6f} d, T_dur={T_dur:.6f} d)"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(t[order],residuals[order],s=4,alpha=0.4)

    axes[1].axhline(0,lw=1)

    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals: data - time-domain model")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def cut_transit_windows(t, flux, flux_err, P, t0, T_dur):
    windows = []
    cadence = np.median(np.diff(t))
    for n in range(int((t.min()-t0)/P), int((t.max()-t0)/P)+1):
        tc = t0 + n*P
        mask = (t >= tc-2*T_dur) & (t <= tc+2*T_dur)

        tw, fw, ew = t[mask], flux[mask], flux_err[mask]

        # Coverage rule: keep only complete windows with no gap > 3 cadences
        if len(tw) < 5 or tw[0] > tc-2*T_dur+cadence or tw[-1] < tc+2*T_dur-cadence:
            continue
        if np.max(np.diff(tw)) > 3*cadence:
            continue

        # Mask transit + margin, then fit a simple quadratic baseline
        x = tw - tc
        oot = np.abs(x) > 0.75*T_dur
        baseline = np.polyval(np.polyfit(x[oot], fw[oot], 2), x)

        windows.append({"epoch": n, "t": tw, "flux": fw/baseline, "flux_err": ew/baseline, "predicted_tc": tc})

    # Plot windows in a grid
    fig, axes = plt.subplots(1, len(windows), figsize=(4*len(windows), 3))
    axes = np.atleast_1d(axes)

    for ax, w in zip(axes, windows):
        ax.plot(w["t"]-w["predicted_tc"], w["flux"], ".")
        ax.set_title(f"Epoch {w['epoch']}")

    plt.tight_layout()
    plt.show()

    return windows

def global_planet_fit(windows, P, t0, depth, T_dur, T_flat):
    for w in windows:
        oot = np.abs(w["t"] - w["predicted_tc"]) > 0.75*T_dur
        w["sigma"] = np.std(w["flux"][oot])

    theta0 = [P, t0, depth, T_dur, T_flat]
    ndim, walkers = 5, 32
    pos = theta0 + 1e-4*np.random.randn(walkers, ndim)

    sampler = emcee.EnsembleSampler(
        walkers, ndim, log_posterior, args=(windows,)
    )

    sampler.run_mcmc(pos, 500, progress=True)

    tau = sampler.get_autocorr_time()
    sampler.reset()
    sampler.run_mcmc(
        sampler.get_last_sample(),
        int(50*np.max(tau)),
        progress=True
    )

    return sampler

def fit_transits(windows, P, t0, depth, T_dur, T_flat):
    results = []

    for w in windows:
        t, flux, err = w["t"], w["flux"], w["flux_err"]
        tc0, epoch = w["predicted_tc"], w["epoch"]

        def log_prob(p):
            tc, dur, H = p
            if dur <= 0 or dur > 0.5 or H <= 0:
                return -np.inf
            model = transit_model(t, tc, P, H, depth, dur, T_flat)
            if not np.all(np.isfinite(model)):
                return -np.inf
            return -0.5 * np.sum(((flux - model) / err) ** 2)

        start = [tc0, T_dur, 1.0]
        scale = [1e-4, 1e-3, 1e-4]
        pos = start + np.array(scale) * np.random.randn(16, 3)

        sampler = emcee.EnsembleSampler(16, 3, log_prob)
        sampler.run_mcmc(pos, 1000, progress=False)
        chain = sampler.get_chain(discard=500, flat=True)

        tc = np.median(chain[:, 0])
        dur = np.median(chain[:, 1])
        tc_err = np.std(chain[:, 0])
        dur_err = np.std(chain[:, 1])

        results.append({
            "epoch": epoch, "tc": tc, "tc_err": tc_err,
            "T_dur": dur, "T_dur_err": dur_err,
            "TTV": tc - (t0 + epoch * P),
            "TDV": dur - T_dur
        })

    epochs = np.array([r["epoch"] for r in results])
    ttv = np.array([r["TTV"] for r in results])
    tdv = np.array([r["TDV"] for r in results])
    ttv_err = np.array([r["tc_err"] for r in results])
    tdv_err = np.array([r["T_dur_err"] for r in results])

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ax[0].errorbar(epochs, ttv, yerr=ttv_err, fmt="o")
    ax[0].axhline(0, linestyle="--")
    ax[0].set_ylabel("TTV (days)")
    ax[1].errorbar(epochs, tdv, yerr=tdv_err, fmt="o")
    ax[1].axhline(0, linestyle="--")
    ax[1].set_ylabel("TDV (days)")
    ax[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.show()

    return results
def periodogram(results):
    from astropy.timeseries import LombScargle

    epoch = np.array([r["epoch"] for r in results])
    ttv = np.array([r["TTV"] for r in results])
    tdv = np.array([r["TDV"] for r in results])
    ttv_err = np.array([r["tc_err"] for r in results])
    tdv_err = np.array([r["T_dur_err"] for r in results])

    freq = np.linspace(0.01, 0.5, 5000)

    ls1 = LombScargle(epoch, ttv, ttv_err)
    ls2 = LombScargle(epoch, tdv, tdv_err)

    power1 = ls1.power(freq)
    power2 = ls2.power(freq)

    p1 = 1 / freq[np.argmax(power1)]
    p2 = 1 / freq[np.argmax(power2)]

    fap1 = ls1.false_alarm_probability(np.max(power1))
    fap2 = ls2.false_alarm_probability(np.max(power2))

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ax[0].plot(1 / freq, power1)
    ax[0].set_ylabel("TTV Power")
    ax[1].plot(1 / freq, power2)
    ax[1].set_ylabel("TDV Power")
    ax[1].set_xlabel("Period (epochs)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(ttv, tdv)
    plt.xlabel("TTV (days)")
    plt.ylabel("TDV (days)")
    plt.title("TTV vs TDV")
    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.show()

    print("TTV period:", p1, "FAP:", fap1)
    print("TDV period:", p2, "FAP:", fap2)

    return {"TTV_period": p1, "TDV_period": p2,
            "TTV_FAP": fap1, "TDV_FAP": fap2}
    
def injection_recovery(t, flux, flux_err, P, t0, depth,
                             T_dur, moon_masses, moon_periods,
                             run_pipeline, n_trials=5):

    recovery = np.zeros((len(moon_masses), len(moon_periods)))

    epochs = np.unique(np.round((t - t0) / P).astype(int))

    for i, mass in enumerate(moon_masses):
        for j, pm in enumerate(moon_periods):
            detected = 0

            for _ in range(n_trials):
                phase = np.random.uniform(0, 2 * np.pi)
                injected = flux.copy()

                for n in epochs:
                    angle = 2 * np.pi * n * P / pm + phase
                    ttv = mass * np.sin(angle) * 1e-4
                    tdv = mass * np.cos(angle) * T_dur * 1e-3

                    tc = t0 + n * P
                    mask = np.abs(t - tc) < 2 * T_dur

                    old = transit_model(
                        t[mask], tc, P, 1, depth, T_dur, T_dur
                    )
                    new = transit_model(
                        t[mask], tc + ttv, P, 1,
                        depth, T_dur + tdv, T_dur
                    )

                    injected[mask] += new - old

                if run_pipeline(t, injected, flux_err):
                    detected += 1

            recovery[i, j] = detected / n_trials

    plt.figure(figsize=(8, 6))
    plt.imshow(
        recovery,
        origin="lower",
        aspect="auto",
        extent=[moon_periods[0], moon_periods[-1],
                moon_masses[0], moon_masses[-1]]
    )
    plt.colorbar(label="Recovery Fraction")
    plt.xlabel("Moon Period")
    plt.ylabel("Moon Mass")
    plt.title("Injection-Recovery Map")
    plt.show()

    return recovery
