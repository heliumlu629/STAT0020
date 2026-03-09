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
# 6. Positive-damage subset
ga_pos = ga_2020[ga_2020["DAMAGE_PROPERTY"] > 0].copy()
n_positive = len(ga_pos)
# Check Minimum sample rule: >50 which is fine
print('Number of positive damage events:',n_positive)


# ----------------------------
# 7. Summary quantities for Q3(a)
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
# 9. Descriptive display: histogram of log positive losses
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