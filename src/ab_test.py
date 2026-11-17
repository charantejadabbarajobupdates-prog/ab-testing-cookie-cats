import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower

out = []
def log(s=""):
    print(s); out.append(str(s))

df = pd.read_csv("data/raw/cookie_cats.csv")
log(f"Rows: {len(df):,} | duplicate users: {df.userid.duplicated().sum()}")

# 1. Sample ratio check: are the groups the expected 50/50 split?
counts = df.version.value_counts()
chi_srm = stats.chisquare(counts.values)
log(f"\n[1] Group sizes: {counts.to_dict()} | SRM p-value = {chi_srm.pvalue:.3f}")

# 2. Outliers in game rounds
log(f"\n[2] Max game rounds: {df.sum_gamerounds.max():,} (extreme outlier, removed for the rounds analysis)")
rounds = df[df.sum_gamerounds < df.sum_gamerounds.quantile(0.999)]

# 3. Retention tests
log("\n[3] Retention by version")
for metric in ["retention_1", "retention_7"]:
    g = df.groupby("version")[metric].agg(["sum", "count", "mean"])
    a, b = g.loc["gate_30"], g.loc["gate_40"]
    z, p = proportions_ztest([a["sum"], b["sum"]], [a["count"], b["count"]])
    diff = b["mean"] - a["mean"]
    se = np.sqrt(a["mean"]*(1-a["mean"])/a["count"] + b["mean"]*(1-b["mean"])/b["count"])
    lo, hi = diff - 1.96*se, diff + 1.96*se
    log(f"{metric}: gate_30={a['mean']:.2%} gate_40={b['mean']:.2%} | diff={diff:+.2%} "
        f"95% CI [{lo:+.2%}, {hi:+.2%}] | z-test p={p:.4f}")
    ct = pd.crosstab(df.version, df[metric])
    log(f"   chi-square p={stats.chi2_contingency(ct)[1]:.4f}")

# 4. Bootstrap for 7-day retention
rng = np.random.default_rng(42)
r30 = df.loc[df.version == "gate_30", "retention_7"].astype(int).values
r40 = df.loc[df.version == "gate_40", "retention_7"].astype(int).values
boot = np.array([rng.choice(r40, len(r40)).mean() - rng.choice(r30, len(r30)).mean()
                 for _ in range(2000)])
log(f"\n[4] Bootstrap diff (gate_40 - gate_30), 7-day retention: mean={boot.mean():+.3%}, "
    f"95% CI [{np.percentile(boot,2.5):+.3%}, {np.percentile(boot,97.5):+.3%}], "
    f"P(gate_40 worse) = {(boot < 0).mean():.1%}")

# 5. Game rounds (not normal, so use Mann-Whitney)
a = rounds.loc[rounds.version == "gate_30", "sum_gamerounds"]
b = rounds.loc[rounds.version == "gate_40", "sum_gamerounds"]
u = stats.mannwhitneyu(a, b)
log(f"\n[5] Game rounds median: gate_30={a.median()} gate_40={b.median()} | Mann-Whitney p={u.pvalue:.3f}")

# 6. Power analysis: smallest effect we could reliably detect
es = NormalIndPower().solve_power(nobs1=len(r30), alpha=0.05, power=0.8, ratio=len(r40)/len(r30))
base = r30.mean()
mde = np.sin(np.arcsin(np.sqrt(base)) + es/2)**2 - base
log(f"\n[6] With ~{len(r30):,} users per group, 80% power detects a 7-day retention change of about {mde:.2%}")

# Figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df.groupby("version")[["retention_1", "retention_7"]].mean().mul(100).plot.bar(ax=ax[0])
ax[0].set_title("Retention rate by version (%)"); ax[0].set_ylabel("%"); ax[0].tick_params(axis="x", rotation=0)
ax[1].hist(boot * 100, bins=40, color="steelblue")
ax[1].axvline(0, color="red", ls="--"); ax[1].set_title("Bootstrap: 7-day retention diff (gate_40 - gate_30, pp)")
plt.tight_layout(); plt.savefig("reports/figures/ab_results.png", dpi=150)

open("reports/results.txt", "w", encoding="utf-8").write("\n".join(out))
