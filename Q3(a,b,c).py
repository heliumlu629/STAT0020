import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ----------------------------
# Q3(a)
# ----------------------------
# STEP1: Read the data
# Read the compressed NOAA annual details file directly
df = pd.read_csv("StormEvents_details-2020.csv.gz", compression="gzip")

# Quick check
print("Full dataset shape:", df.shape)
print(df.head())


# ----------------------------
# STEP2: Fiter the assigned state:Georgia
ga_2020 = df[df["STATE"] == "GEORGIA"].copy()

# Check Duplicates
print("Number of duplicated EVENT_ID values:", ga_2020["EVENT_ID"].duplicated().sum())

# Check result
print(ga_2020[["STATE", "YEAR", "MONTH_NAME", "EVENT_TYPE"]].head())


# ----------------------------
# STEP3: Choose date field
# Create a column 'EVENT_DATE'(i.e 2020-01-06)
ga_2020["EVENT_DATE"] = pd.to_datetime(
    ga_2020["BEGIN_YEARMONTH"].astype(str) + ga_2020["BEGIN_DAY"].astype(str).str.zfill(2),
    format="%Y%m%d",
    errors="coerce"
)

# Create a column 'EVENT_MONTH' for the convenience of monthly counts
ga_2020["EVENT_MONTH"] = ga_2020["EVENT_DATE"].dt.month


# ----------------------------
# STEP4: Keep minimum required columns
keep_cols = [ "EVENT_DATE", "EVENT_MONTH","STATE", "EVENT_TYPE", "DAMAGE_PROPERTY"]
ga_2020 = ga_2020[keep_cols].copy()

# Check result
print("Georgia 2020 shape:", ga_2020.shape)
print(ga_2020.head())

# ----------------------------
# STEP5: Clean DAMAGE_PROPERTY into numeric dollars
def parse_damage(x):
    # Missing value -> 0.0
    if pd.isna(x):
        return 0.0
    
    # Blank space -> 0.0
    if x == "":
        return 0.0
    
    # Remove commas and spaces
    x = x.replace(",","")
    
    # Match number + suffix, e.g. 10K, 2.5M, 1B
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([KMB])", x)
    if match:
        number = float(match.group(1))
        suffix = match.group(2)
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}[suffix]
        return number * multiplier

    # Unknown / malformed entries
    return 0.0

ga_2020["DAMAGE_PROPERTY"] = ga_2020["DAMAGE_PROPERTY"].apply(parse_damage)
print(ga_2020["DAMAGE_PROPERTY"])


# ----------------------------
# STEP6: Positive-damage subset
ga_pos = ga_2020[ga_2020["DAMAGE_PROPERTY"] > 0].copy()
n_positive = len(ga_pos)
# Check Minimum sample rule: >50 which is fine
print('Number of positive damage events:',n_positive)


# ----------------------------
# STEP7: Summary quantities for Q3(a)
total_rows = len(ga_2020)
top3 = (
    ga_pos.sort_values("DAMAGE_PROPERTY", ascending=False)
    .loc[:, ["EVENT_DATE", "EVENT_TYPE", "DAMAGE_PROPERTY"]]
    .head(3)
)
print("Total Georgia 2020 event records:", total_rows)
print("Number with positive property damage:", n_positive)
print("\nTop 3 property-damage events:")
print(top3)


# ----------------------------
# STEP8: Descriptive display: histogram of log positive losses
plt.figure(figsize=(8, 5))
plt.hist(np.log10(ga_pos["DAMAGE_PROPERTY"]), bins=30)
plt.xlabel("log10(Property damage in dollars)")
plt.ylabel("Frequency")
plt.title("Histogram of log10 positive property damages: Georgia 2020")
plt.tight_layout()
plt.savefig("q3a_hist_log_damage.png", dpi=300)
plt.show()




# ----------------------------
# Q3(b)
# ----------------------------
# STEP1: Step 1: Build monthly counts N_m
monthly_counts = (
    ga_pos.groupby("EVENT_MONTH")
    .size()
    .reindex(range(1, 13), fill_value=0)
)

print(monthly_counts)


# ----------------------------
# STEP2: Compute the overdispersion ratio
# If R > 1,
# that suggests overdispersion, so Negative Binomial may be better
mean_Nm = monthly_counts.mean()
var_Nm = monthly_counts.var(ddof=1)   
R = var_Nm / mean_Nm

print("Mean monthly count =", mean_Nm)
print("Variance of monthly count =", var_Nm)
print("Overdispersion ratio R =", R)


# ----------------------------
# STEP3: Fit the frequency model(Negative Binomial)
# Using MoM method to get the parameters
mu_hat = mean_Nm
v_hat = var_Nm
alpha_hat = (v_hat - mu_hat) / (mu_hat ** 2)
print("Negative Binomial mu_hat =", mu_hat)
print("Negative Binomial alpha_hat =", alpha_hat)


# ----------------------------
# STEP4: Fit the Lognormal severity model
logX = np.log(ga_pos["DAMAGE_PROPERTY"])

mu_log_hat = logX.mean()
sigma_log_hat = logX.std(ddof=0)   # MLE version

print("Lognormal mu_hat =", mu_log_hat)
print("Lognormal sigma_hat =", sigma_log_hat)


# ----------------------------
# STEP5: One diagnostic for the Lognormal fit
from scipy import stats
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
stats.probplot(logX, dist="norm", plot=plt)
plt.title("QQ plot for log positive property damage")
plt.tight_layout()
plt.savefig("q3b_qqplot_logX.png", dpi=300)
plt.show()

# ----------------------------
# STEP6: Compound Model Moments
# Negative Binomial frequency moments
EN = 12 * mu_hat
VarN = 12 * (mu_hat + alpha_hat * mu_hat**2)

# Lognormal severity moments
EX = np.exp(mu_log_hat + 0.5 * sigma_log_hat**2)
VarX = (np.exp(sigma_log_hat**2) - 1) * np.exp(2 * mu_log_hat + sigma_log_hat**2)

# Compound annual loss moments
ES = EN * EX
VarS = EN * VarX + VarN * (EX ** 2)

print("E[N] =", EN)
print("Var(N) =", VarN)
print("E[X] =", EX)
print("Var(X) =", VarX)
print("E[S] =", ES)
print("Var(S) =", VarS)







#q3(c)
# Q3(c)
# Monte Carlo simulation for annual aggregate loss
# S = sum_{i=1}^N X_i

# STEP1: Set simulation choices
M = 100000
seed = 2026
np.random.seed(seed)

# STEP2: Negative Binomial parameterisation conversion
# For monthly count: Var(N_m) = mu + alpha * mu^2  
# But numpy.random.negative_binomial(n, p) uses: mean = n(1-p)/p, var = n(1-p)/p^2 = mu + mu^2/n
# so n = 1/alpha, p = n / (n + mu)

if alpha_hat > 0:
    size_nb = 1 / alpha_hat
    p_nb = size_nb / (size_nb + mu_hat)
else:
    size_nb = None
    p_nb = None

print("Simulation settings:")
print("M =", M)
print("seed =", seed)

if alpha_hat > 0:
    print("Monthly NB size =", size_nb)
    print("Monthly NB p =", p_nb)
else:
    print("alpha_hat <= 0, NB not appropriate; would need Poisson instead.")


# STEP3: Simulate annual aggregate loss
sim_S = np.zeros(M)

for j in range(M):
    # simulate 12 monthly counts, then sum to annual count
    if alpha_hat > 0:
        monthly_N = np.random.negative_binomial(size_nb, p_nb, size=12)
    else:
        monthly_N = np.random.poisson(mu_hat, size=12)
    
    N_year = monthly_N.sum()
    
    # simulate severities and aggregate loss
    if N_year > 0:
        losses = np.random.lognormal(mean=mu_log_hat, sigma=sigma_log_hat, size=N_year)
        sim_S[j] = losses.sum()
    else:
        sim_S[j] = 0.0


# STEP4: Monte Carlo estimates
ES_sim = sim_S.mean()
VarS_sim = sim_S.var(ddof=1)

# 99% VaR
VaR_99 = np.quantile(sim_S, 0.99)

# 99% TVaR
TVaR_99 = sim_S[sim_S >= VaR_99].mean()

# standard error for Monte Carlo estimate of E[S]
SE_ES = sim_S.std(ddof=1) / np.sqrt(M)

print("\nMonte Carlo results:")
print("MC E[S] =", ES_sim)
print("MC Var(S) =", VarS_sim)
print("VaR_0.99 =", VaR_99)
print("TVaR_0.99 =", TVaR_99)
print("SE(E[S]) =", SE_ES)


# STEP5: Compare simulation results with theoretical values from Q3(b)
print("\nComparison with theoretical results:")
print("Theoretical E[S] =", ES)
print("Monte Carlo E[S] =", ES_sim)
print("Difference in E[S] =", ES_sim - ES)
print("Relative difference in E[S] =", (ES_sim - ES) / ES)

print("Theoretical Var(S) =", VarS)
print("Monte Carlo Var(S) =", VarS_sim)
print("Difference in Var(S) =", VarS_sim - VarS)
print("Relative difference in Var(S) =", (VarS_sim - VarS) / VarS)

mean_S = sim_S.mean()
median_S = np.median(sim_S)

q50 = np.quantile(sim_S, 0.50)
q90 = np.quantile(sim_S, 0.90)
q95 = np.quantile(sim_S, 0.95)
q99 = np.quantile(sim_S, 0.99)

print("\nAdditional statistics:")
print("Mean of S =", mean_S)
print("Median of S =", median_S)
print("50% quantile =", q50)
print("90% quantile =", q90)
print("95% quantile =", q95)
print("99% quantile =", q99)


# STEP6: Plot simulated aggregate loss distribution
plt.figure(figsize=(8, 5))
plt.hist(sim_S, bins=60)
plt.xlabel("Simulated annual aggregate loss S")
plt.ylabel("Frequency")
plt.title("Monte Carlo distribution of annual aggregate loss")
plt.tight_layout()
plt.savefig("q3c_simulated_aggregate_loss.png", dpi=300)
plt.show()


# STEP7: Summary table 
summary_table = pd.DataFrame({
    "Quantity": [
        "Number of simulations M",
        "Random seed",
        "Theoretical E[S]",
        "Theoretical Var(S)",
        "Monte Carlo E[S]",
        "Monte Carlo Var(S)",
        "VaR_0.99",
        "TVaR_0.99",
        "SE(E[S])" ],
    "Value": [
        M,
        seed,
        ES,
        VarS,
        ES_sim,
        VarS_sim,
        VaR_99,
        TVaR_99,
        SE_ES ]
})


print("\nResults Summary Table:")
print(summary_table)