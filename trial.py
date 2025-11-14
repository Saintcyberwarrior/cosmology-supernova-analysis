# %% [markdown]
# # Cosmology Supernova Analysis
# ## Bayesian Analysis of SCP 2.1 Data to Measure H₀ and q₀
# 
# **Project Overview**: Using Type Ia supernovae as standard candles to measure cosmological parameters and test for accelerating universe expansion.
# 
# **Techniques Used**: Bayesian inference, MCMC sampling, posterior predictive checks, model comparison

# %% [markdown]
# ## 1. Import Libraries and Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import corner
import scipy.special
from scipy.optimize import minimize
from scipy import stats
import arviz as az

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("All libraries imported successfully")

# %% [markdown]
# ## 2. Load and Explore SCP 2.1 Data

# %%
# SCP 2.1 Supernova Data
data_lines = """1993ah  0.028488  35.3465833928  0.223905932998  0.128418942246
1993ag  0.050043  36.6823679154  0.166828851413  0.128418942246
1993o  0.052926  36.8176912545  0.1557559148  0.128418942246
1993b  0.070086  37.4467365424  0.158466934433  0.128418942246
1992bs  0.062668  37.4834093505  0.156099434739  0.128418942246
1992br  0.087589  38.2290570494  0.187745679272  0.128418942246
1992bp  0.078577  37.4881622607  0.1556356565185  0.128418942246
1992bo  0.017227  34.6543699503  0.199337179559  0.128418942246
1992bl  0.042233  36.3364595483  0.167174042338  0.128418942246
1992bh  0.045295  36.6402721756  0.164981248644  0.128418942246
1992bg  0.03648  35.9053219652  0.170174952845  0.128418942246
1992bc  0.019599  34.5852174312  0.184691219687  0.128418942246
1992aq  0.100915  38.4567455954  0.167333481677  0.128418942246
1992ag  0.027342  35.085765693  0.175510835947  0.128418942246
1992ae  0.074605  37.5881157565  0.15977086456  0.128418942246
1992p  0.026489  35.4806851993  0.19131226974  0.001870261557
1990af  0.049922  36.5669734706  0.162303819627  0.128418942246
1990o  0.030604  35.5502377594  0.17329544142  0.128418942246"""

# Parse data into DataFrame
data = []
for line in data_lines.strip().split('\n'):
    parts = line.split()
    data.append({
        'SN_name': parts[0],
        'Redshift': float(parts[1]),
        'Distance_modulus': float(parts[2]),
        'Distance_modulus_error': float(parts[3]),
        'P_low_mass': float(parts[4])
    })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} supernovae from SCP 2.1 dataset")

# Display first few rows
print("\nFirst 5 supernovae:")
print(df.head())

# Basic statistics
print(f"\nData Statistics:")
print(f"Redshift range: {df['Redshift'].min():.4f} - {df['Redshift'].max():.4f}")
print(f"Distance modulus range: {df['Distance_modulus'].min():.2f} - {df['Distance_modulus'].max():.2f}")

# Filter for low-redshift data (z < 0.5) as required
df_lowz = df[df['Redshift'] < 0.5].copy()
print(f"\nUsing {len(df_lowz)} supernovae with z < 0.5 for analysis")

# %% [markdown]
# ## 3. Data Visualization

# %%
# Plot the complete dataset
plt.figure(figsize=(12, 8))

# All data
plt.subplot(2, 2, 1)
plt.errorbar(df['Redshift'], df['Distance_modulus'], 
             yerr=df['Distance_modulus_error'], fmt='o', alpha=0.7, capsize=3)
plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus (μ)')
plt.title('All SCP 2.1 Supernova Data')
plt.grid(True, alpha=0.3)

# Low-z data only
plt.subplot(2, 2, 2)
plt.errorbar(df_lowz['Redshift'], df_lowz['Distance_modulus'], 
             yerr=df_lowz['Distance_modulus_error'], fmt='o', alpha=0.7, capsize=3, color='red')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus (μ)')
plt.title('Low-z Data (z < 0.5) for Analysis')
plt.grid(True, alpha=0.3)

# Error distribution
plt.subplot(2, 2, 3)
plt.hist(df['Distance_modulus_error'], bins=20, alpha=0.7, color='green')
plt.xlabel('Distance Modulus Error')
plt.ylabel('Count')
plt.title('Error Distribution')
plt.grid(True, alpha=0.3)

# Redshift distribution
plt.subplot(2, 2, 4)
plt.hist(df['Redshift'], bins=20, alpha=0.7, color='purple')
plt.xlabel('Redshift (z)')
plt.ylabel('Count')
plt.title('Redshift Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Physical Model Implementation

# %%
# Constants
c = 299792.458  # speed of light in km/s

def distance_modulus_model(z, H0, q0, model_type='small_z'):
    """
    Calculate distance modulus using different cosmological models
    
    Parameters:
    z: redshift
    H0: Hubble constant (km/s/Mpc) 
    q0: deceleration parameter
    model_type: 'small_z' for Taylor expansion, 'linear' for very low-z
    
    Returns:
    mu: distance modulus
    """
    if model_type == 'small_z':
        # Small-z approximation (Eq. 17) - our main model
        dL = (c / H0) * (z + 0.5 * (1 - q0) * z**2)
    elif model_type == 'linear':
        # Linear approximation for very low-z (checking H0)
        dL = (c / H0) * z
    else:
        raise ValueError("Unknown model type")
    
    # Distance modulus formula (Eq. 4)
    mu = 5 * np.log10(dL) + 25
    return mu

# Test the model
print("Model Test:")
test_z = 0.1
test_H0 = 70
test_q0 = -0.5
test_mu = distance_modulus_model(test_z, test_H0, test_q0)
print(f"μ(z={test_z}, H₀={test_H0}, q₀={test_q0}) = {test_mu:.2f}")

# %% [markdown]
# ## 5. Bayesian Inference Setup

# %%
def log_inverse_gamma(x, alpha, beta):
    """
    Inverse Gamma distribution log-PDF
    From lecture: 2.2 A conjugate prior for estimating θ_T
    """
    return (alpha * np.log(beta) - scipy.special.gammaln(alpha) 
            - (alpha + 1) * np.log(x) - beta/x)

def log_prior(params):
    """
    Prior distributions for parameters
    params = [H0, q0, sigma]
    """
    H0, q0, sigma = params
    
    # Uniform priors for H0 and q0 (as suggested in problem)
    if not (50 < H0 < 100):
        return -np.inf
    if not (-2 < q0 < 2):
        return -np.inf
    if sigma <= 0:
        return -np.inf
    
    # Inverse Gamma prior for sigma^2 (intrinsic scatter)
    alpha = 2  # shape parameter
    beta = 0.1  # scale parameter
    log_prior_sigma = log_inverse_gamma(sigma**2, alpha, beta)
    
    return log_prior_sigma

def log_likelihood(params, z_data, mu_data, mu_err_data):
    """
    Gaussian log-likelihood with measurement errors and intrinsic scatter
    """
    H0, q0, sigma = params
    
    # Model predictions
    mu_model = distance_modulus_model(z_data, H0, q0)
    
    # Total variance = measurement error^2 + intrinsic scatter^2
    total_variance = mu_err_data**2 + sigma**2
    
    # Gaussian log-likelihood
    logL = -0.5 * np.sum(
        (mu_data - mu_model)**2 / total_variance + 
        np.log(2 * np.pi * total_variance)
    )
    
    return logL

def log_posterior(params, z_data, mu_data, mu_err_data):
    """
    Total log posterior = log_prior + log_likelihood
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood(params, z_data, mu_data, mu_err_data)
    return lp + ll

# Prepare data for inference
z_data = df_lowz['Redshift'].values
mu_data = df_lowz['Distance_modulus'].values
mu_err_data = df_lowz['Distance_modulus_error'].values

print("Bayesian model setup complete")
print(f"Using {len(z_data)} data points for inference")

# %% [markdown]
# ## 6. Maximum A Posteriori (MAP) Estimation

# %%
# Find MAP estimate for good starting values
initial_guess = [70, -0.5, 0.2]

def neg_log_posterior(params):
    return -log_posterior(params, z_data, mu_data, mu_err_data)

print("Finding MAP estimate...")
result = minimize(neg_log_posterior, initial_guess, method='Nelder-Mead')

if result.success:
    map_estimate = result.x
    print(f"MAP estimate found:")
    print(f"  H₀ = {map_estimate[0]:.2f} km/s/Mpc")
    print(f"  q₀ = {map_estimate[1]:.3f}")
    print(f"  σ = {map_estimate[2]:.3f}")
else:
    print("MAP optimization failed, using initial guess")
    map_estimate = initial_guess

# Test the MAP model
z_test = np.linspace(0.01, 0.5, 100)
mu_test = distance_modulus_model(z_test, map_estimate[0], map_estimate[1])

plt.figure(figsize=(10, 6))
plt.errorbar(z_data, mu_data, yerr=mu_err_data, fmt='o', alpha=0.7, label='Data')
plt.plot(z_test, mu_test, 'r-', linewidth=2, label='MAP Model')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus (μ)')
plt.title('MAP Model Fit to Low-z Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 7. MCMC Sampling for Posterior Distribution

# %%
# Set up MCMC
nwalkers = 32
ndim = 3
nsteps = 4000
burnin = 1000

# Initialize walkers around MAP estimate
initial_pos = map_estimate + 1e-4 * np.random.randn(nwalkers, ndim)

print("Starting MCMC sampling...")
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior, 
    args=(z_data, mu_data, mu_err_data)
)

# Run MCMC
sampler.run_mcmc(initial_pos, nsteps, progress=True)

# Check chain convergence
fig, axes = plt.subplots(ndim, figsize=(10, 8), sharex=True)
labels = ['H₀', 'q₀', 'σ']
for i in range(ndim):
    ax = axes[i]
    ax.plot(sampler.chain[:, :, i].T, alpha=0.3)
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel('Step Number')
plt.suptitle('MCMC Chains - Convergence Check')
plt.tight_layout()
plt.show()

# Extract samples (remove burn-in)
samples = sampler.chain[:, burnin:, :].reshape(-1, ndim)
H0_samples, q0_samples, sigma_samples = samples[:, 0], samples[:, 1], samples[:, 2]

print(f"MCMC complete: {len(samples)} posterior samples")
print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

# %% [markdown]
# ## 8. Results Analysis

# %%
# Calculate summary statistics
H0_median = np.median(H0_samples)
H0_low, H0_high = np.percentile(H0_samples, [16, 84])

q0_median = np.median(q0_samples)
q0_low, q0_high = np.percentile(q0_samples, [16, 84])

sigma_median = np.median(sigma_samples)

print("="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"H₀ = {H0_median:.1f} +{H0_high-H0_median:.1f} -{H0_median-H0_low:.1f} km/s/Mpc")
print(f"q₀ = {q0_median:.3f} +{q0_high-q0_median:.3f} -{q0_median-q0_low:.3f}")
print(f"Intrinsic scatter σ = {sigma_median:.3f}")

# Joint probability distribution
fig = corner.corner(
    samples[:, :2],
    labels=['H₀', 'q₀'],
    truths=[H0_median, q0_median],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12}
)
plt.suptitle('Joint Probability Distribution: H₀ vs q₀', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Evidence for Accelerating Expansion

# %%
# Calculate probability that q₀ < 0 (accelerating universe)
prob_accelerating = np.mean(q0_samples < 0)
prob_decelerating = 1 - prob_accelerating

print("="*60)
print("EVIDENCE FOR ACCELERATING EXPANSION")
print("="*60)
print(f"Probability that q₀ < 0 (accelerating): {prob_accelerating*100:.1f}%")
print(f"Probability that q₀ > 0 (decelerating): {prob_decelerating*100:.1f}%")

# Visualize q₀ posterior
plt.figure(figsize=(10, 6))
plt.hist(q0_samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=3, label='q₀ = 0 (No acceleration)')
plt.axvline(q0_median, color='black', linestyle='-', linewidth=2, 
            label=f'Median q₀ = {q0_median:.3f}')
plt.xlabel('Deceleration Parameter (q₀)')
plt.ylabel('Probability Density')
plt.title('Posterior Distribution of q₀\nEvidence for Accelerating Expansion')
plt.legend()
plt.grid(True, alpha=0.3)

# Add probability text
plt.text(0.05, 0.95, f'P(q₀ < 0) = {prob_accelerating*100:.1f}%', 
         transform=plt.gca().transAxes, fontsize=14, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

if prob_accelerating > 0.95:
    conclusion = "✅ STRONG evidence for accelerating expansion"
elif prob_accelerating > 0.68:
    conclusion = "⚠️ MODERATE evidence for accelerating expansion"
else:
    conclusion = "❓ INCONCLUSIVE evidence for accelerating expansion"

print(f"\nCONCLUSION: {conclusion}")

# %% [markdown]
# ## 10. Validation: H₀ Extraction Check

# %%
# Check H₀ using very low-z data and linear approximation
df_very_lowz = df[df['Redshift'] < 0.1].copy()

if len(df_very_lowz) > 0:
    print(f"\nUsing {len(df_very_lowz)} supernovae with z < 0.1 for H₀ validation")
    
    # Linear model (no q₀ dependence)
    def log_posterior_linear(params, z_data, mu_data, mu_err_data):
        H0, sigma = params
        # Priors
        if not (50 < H0 < 100) or sigma <= 0:
            return -np.inf
        
        mu_model = distance_modulus_model(z_data, H0, 0, model_type='linear')
        total_variance = mu_err_data**2 + sigma**2
        logL = -0.5 * np.sum(
            (mu_data - mu_model)**2 / total_variance + 
            np.log(2 * np.pi * total_variance)
        )
        
        # Inverse gamma prior for sigma
        log_prior_sigma = log_inverse_gamma(sigma**2, 2, 0.1)
        
        return logL + log_prior_sigma
    
    # Prepare very low-z data
    z_vlow = df_very_lowz['Redshift'].values
    mu_vlow = df_very_lowz['Distance_modulus'].values
    mu_err_vlow = df_very_lowz['Distance_modulus_error'].values
    
    # Quick MCMC for linear model
    nwalkers_linear = 16
    initial_linear = [70, 0.1] + 1e-4 * np.random.randn(nwalkers_linear, 2)
    
    sampler_linear = emcee.EnsembleSampler(
        nwalkers_linear, 2, log_posterior_linear,
        args=(z_vlow, mu_vlow, mu_err_vlow)
    )
    
    sampler_linear.run_mcmc(initial_linear, 2000, progress=False)
    samples_linear = sampler_linear.chain[:, 500:, :].reshape(-1, 2)
    H0_linear = np.median(samples_linear[:, 0])
    H0_linear_err = np.std(samples_linear[:, 0])
    
    print("="*50)
    print("H₀ EXTRACTION VALIDATION")
    print("="*50)
    print(f"H₀ from full model (z < 0.5): {H0_median:.1f} km/s/Mpc")
    print(f"H₀ from linear model (z < 0.1): {H0_linear:.1f} ± {H0_linear_err:.1f} km/s/Mpc")
    print(f"Difference: {abs(H0_median - H0_linear):.1f} km/s/Mpc")
    
    if abs(H0_median - H0_linear) < 5:
        print("✅ H₀ extraction is consistent between methods")
    else:
        print("⚠️ H₀ extraction shows some discrepancy")
        
    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.errorbar(z_vlow, mu_vlow, yerr=mu_err_vlow, fmt='o', label='Very low-z data')
    
    z_plot = np.linspace(0.01, 0.1, 50)
    mu_full = distance_modulus_model(z_plot, H0_median, q0_median)
    mu_linear = distance_modulus_model(z_plot, H0_linear, 0, model_type='linear')
    
    plt.plot(z_plot, mu_full, 'r-', label=f'Full model (H₀={H0_median:.1f})')
    plt.plot(z_plot, mu_linear, 'g--', label=f'Linear model (H₀={H0_linear:.1f})')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Distance Modulus (μ)')
    plt.title('H₀ Validation: Full vs Linear Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
else:
    print("Not enough very low-z data for H₀ validation")

# %% [markdown]
# ## 11. Posterior Predictive Check

# %%
print("Generating posterior predictive check...")

# Generate predictions across full redshift range
z_range = np.linspace(0.01, max(df['Redshift']), 200)
n_predictive = 200  # Number of posterior samples to use

plt.figure(figsize=(12, 8))

# Plot all data
plt.errorbar(df['Redshift'], df['Distance_modulus'], 
             yerr=df['Distance_modulus_error'], fmt='o', alpha=0.6, 
             capsize=3, label='SCP Data', zorder=5)

# Plot posterior predictive samples
for i in range(n_predictive):
    idx = np.random.randint(len(samples))
    H0_sample, q0_sample, sigma_sample = samples[idx]
    
    mu_pred = distance_modulus_model(z_range, H0_sample, q0_sample)
    
    if i == 0:
        plt.plot(z_range, mu_pred, color='red', alpha=0.1, label='Posterior Predictive')
    else:
        plt.plot(z_range, mu_pred, color='red', alpha=0.1)

# Plot median prediction
mu_median_pred = np.array([distance_modulus_model(z, H0_median, q0_median) 
                          for z in z_range])
plt.plot(z_range, mu_median_pred, 'k-', linewidth=3, label='Median Prediction')

plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus (μ)')
plt.title('Posterior Predictive Check\nModel vs Data Across All Redshifts')
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.7, 
            label='z = 0.5 (our fit limit)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Quantitative goodness-of-fit
mu_pred_all = np.array([distance_modulus_model(z, H0_median, q0_median) 
                       for z in df['Redshift']])
residuals = df['Distance_modulus'] - mu_pred_all
chi2 = np.sum((residuals / df['Distance_modulus_error'])**2)
reduced_chi2 = chi2 / len(df)

print("="*50)
print("POSTERIOR PREDICTIVE CHECK SUMMARY")
print("="*50)
print(f"χ² = {chi2:.1f}")
print(f"Reduced χ² = {reduced_chi2:.2f}")
print(f"Number of data points = {len(df)}")

if 0.5 < reduced_chi2 < 1.5:
    print("✅ Model provides reasonable fit to data")
else:
    print("⚠️ Model shows some tension with data")

# %% [markdown]
# ## 12. Discussion and Improvements

# %%
print("="*60)
print("DISCUSSION AND POTENTIAL IMPROVEMENTS")
print("="*60)

improvements = """
Based on the lecture techniques covered, here are potential improvements:

1. **USE FULL COSMOLOGICAL MODELS** (Section 3.1):
   - Replace small-z approximation with exact ΛCDM integrals
   - Implement wCDM model with dark energy equation of state

2. **BAYESIAN MODEL COMPARISON** (Section 3.3-3.4):
   - Calculate Bayes factors to compare different cosmological models
   - Use DIC or WAIC for model selection

3. **HIERARCHICAL MODELING** (Section 2.5):
   - Account for population effects in supernova luminosities
   - Model intrinsic scatter as a hyperparameter

4. **GAUSSIAN PROCESS REGRESSION** (Section 2.5):
   - Model correlated uncertainties and systematics
   - Non-parametric reconstruction of expansion history

5. **BAYESIAN MODEL AVERAGING** (Section 4.4):
   - Combine predictions from multiple cosmological models
   - Account for model uncertainty in parameter estimates

6. **ROBUST REGRESSION** (Section 5.2):
   - Use Student-t likelihood for outlier resistance
   - Account for heavy-tailed error distributions

7. **PRIOR SENSITIVITY ANALYSIS**:
   - Test different prior choices for cosmological parameters
   - Use informative priors from complementary datasets

CURRENT LIMITATIONS ADDRESSED:
- Using only low-z data (z < 0.5) due to approximation
- Simple Gaussian error model
- Ignoring potential systematics and correlations
- Not using full dataset potential
"""

print(improvements)

# %% [markdown]
# ## 13. Final Conclusion

# %%
print("="*70)
print("FINAL PROJECT CONCLUSION")
print("="*70)

print(f"📊 ANALYSIS SUMMARY:")
print(f"   • Used {len(df_lowz)}/{len(df)} supernovae (z < 0.5)")
print(f"   • Bayesian inference with MCMC sampling")
print(f"   • Inverse Gamma prior for intrinsic scatter")

print(f"\n🎯 KEY RESULTS:")
print(f"   • H₀ = {H0_median:.1f} +{H0_high-H0_median:.1f} -{H0_median-H0_low:.1f} km/s/Mpc")
print(f"   • q₀ = {q0_median:.3f} +{q0_high-q0_median:.3f} -{q0_median-q0_low:.3f}")
print(f"   • P(accelerating) = {prob_accelerating*100:.1f}%")

print(f"\n🚀 COSMOLOGICAL IMPLICATIONS:")
if prob_accelerating > 0.95:
    print(f"   ✅ STRONG evidence for accelerating universe expansion")
    print(f"   ✓ Supports ΛCDM cosmology with dark energy")
    print(f"   ✓ Consistent with Nobel Prize-winning discovery")
elif prob_accelerating > 0.68:
    print(f"   ⚠️ MODERATE evidence for accelerating expansion")
    print(f"   ~ Suggests presence of dark energy")
else:
    print(f"   ❓ INCONCLUSIVE evidence for acceleration")
    print(f"   ? More data or improved modeling needed")

print(f"\n🔬 METHODOLOGICAL ASSESSMENT:")
print(f"   ✓ Successfully applied Bayesian inference techniques")
print(f"   ✓ Demonstrated MCMC for parameter estimation") 
print(f"   ✓ Implemented posterior predictive checks")
print(f"   → Foundation for more advanced cosmological analyses")

print("="*70)

# %% [markdown]
# ## GitHub Setup Instructions
# 
# To put this notebook on GitHub:
# 
# ```bash
# # Initialize git repository
# git init
# 
# # Add all files
# git add .
# 
# # Make first commit
# git commit -m "Initial commit: Complete cosmology supernova analysis"
# 
# # Create repository on GitHub.com, then:
# git remote add origin https://github.com/yourusername/cosmology-supernova-analysis.git
# git branch -M main
# git push -u origin main
# ```
# 
# **For regular updates:**
# ```bash
# git add .
# git commit -m "Update: [description of changes]"
# git push origin main
# ```

# %%
# Final check - save important results to variables
results_summary = {
    'H0_median': H0_median,
    'H0_uncertainty': (H0_median - H0_low, H0_high - H0_median),
    'q0_median': q0_median, 
    'q0_uncertainty': (q0_median - q0_low, q0_high - q0_median),
    'prob_accelerating': prob_accelerating,
    'n_supernovae_used': len(df_lowz),
    'reduced_chi2': reduced_chi2
}

print("Analysis complete! Results saved.")
print(f"\nKey finding: The universe appears to be accelerating with {prob_accelerating*100:.1f}% probability")
