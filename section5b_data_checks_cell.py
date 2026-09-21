# =============================================================================
# 5b. Data checks: what does "missing" mean in each country?
# Paste as ONE code cell after section 5. Needs LAGGED_FILE from section 2. Runtime < 1 min.
# Findings this cell documents (full data, all four countries):
#   * ~149k Italian firms have no balance sheet and a revenue exactly on a size band
#     (250k / 750k / 1.5M / 3.5M EUR = mid-points of 0-0.5M, 0.5-1M, 1-2M, 2-5M). None of them fails.
#   * The sign of the "missing balance sheet -> failure" link is reversed in IT vs FR/PT/ES.
#   * Ownership (control) is the strongest single predictor and behaves the same in every country.
#   * Sector composition differs little and does not line up with transfer losses.
# =============================================================================
BAND_VALUES = [250_000, 750_000, 1_500_000, 3_500_000]
chk = pd.read_csv(LAGGED_FILE, low_memory=False).dropna(subset=["failure"])
chk["failure"] = chk["failure"].astype(int)
chk["has_balance_sheet"] = chk["total_assets"].notna()
chk["has_income_statement"] = chk["ebitda"].notna()
chk["banded_revenue"] = chk["revenue"].isin(BAND_VALUES) & ~chk["has_balance_sheet"]

print("1) Most frequent revenue values by country (a real revenue distribution has no spikes):")
for c, g in chk.groupby("iso"):
    print(f"   {c}:", g["revenue"].value_counts().head(4).to_dict())

print("\n2) Italian firms with a banded revenue and no balance sheet")
it = chk[chk["iso"].eq("IT")]
print(f"   count = {it['banded_revenue'].sum():,} ({it['banded_revenue'].mean():.1%} of IT);"
      f" failure rate = {it.loc[it['banded_revenue'], 'failure'].mean():.2%}"
      f" vs {it.loc[~it['banded_revenue'], 'failure'].mean():.2%} for the other IT firms")

print("\n3) Failure rate with vs without a balance sheet (the sign flips in IT):")
display(chk.pivot_table(index="iso", columns="has_balance_sheet", values="failure", aggfunc="mean")
        .rename(columns={True: "balance sheet present", False: "balance sheet missing"}).style.format("{:.2%}"))

print("\n4) Filing patterns: share of firms and failure rate by pattern")
chk["pattern"] = np.select(
    [chk["has_balance_sheet"] & chk["has_income_statement"], chk["has_balance_sheet"]],
    ["balance sheet + income statement", "balance sheet only"], "no accounts")
pat = chk.groupby(["iso", "pattern"])["failure"].agg(firms="size", fail_rate="mean")
pat["share"] = pat["firms"] / pat.groupby(level=0)["firms"].transform("sum")
display(pat[["share", "fail_rate"]].unstack(0).style.format("{:.1%}"))

print("\n5) Missing rate of key variables by country")
display(chk.groupby("iso")[["employees", "revenue", "total_assets", "ebitda", "tfp_acf", "int_paid"]]
        .apply(lambda g: g.isna().mean()).T.style.format("{:.0%}"))

print("\n6) Failure rate by ownership type (same pattern in every country)")
display(chk.pivot_table(index="control", columns="iso", values="failure", aggfunc="mean").style.format("{:.2%}"))

print("\n7) Sector composition: dissimilarity vs the other three countries pooled"
      " (0 = identical mix, 1 = no overlap), and failure rate implied by sector mix alone")
shares = pd.crosstab(chk["nace_2"], chk["iso"], normalize="columns")
sector_fail = chk.groupby("nace_2")["failure"].mean()
rows = []
for c in shares.columns:
    others = chk.loc[~chk["iso"].eq(c), "nace_2"].value_counts(normalize=True).reindex(shares.index).fillna(0)
    rows.append({"country": c, "dissimilarity_vs_others": 0.5 * (shares[c] - others).abs().sum(),
                 "actual_fail_rate": chk.loc[chk["iso"].eq(c), "failure"].mean(),
                 "fail_rate_implied_by_sector_mix": (shares[c] * sector_fail).sum()})
display(pd.DataFrame(rows).round(3))
