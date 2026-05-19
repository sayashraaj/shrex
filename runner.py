"""
sensitivity_runner.py
=====================
SEPARATE FILE — the frozen model (asset_location_optimized.py) is imported
as a module and called with different parameters. Nothing in the frozen file
is modified or touched.

PIP Research Questions addressed:
  Q1. What proportion of asset location alpha is attributable to post-tax allocation?
  Q2. How does investment horizon affect alpha?
  Q3. How do tax rates affect alpha?
  Q4. How does turnover / tax inefficiency affect alpha?
  Q5. How do account balance distributions affect alpha?
  Q6. How do contributions & withdrawals affect alpha?
  Q7. What does optimal asset location look like at each horizon?

HOW TO RUN
----------
Step 1. Ensure asset_location_optimized.py is in the same directory as this file.
Step 2. Open a terminal in that directory.
Step 3. Run:   python sensitivity_runner.py
Step 4. Results print to the console and are saved to sensitivity_results_raw.py

Expected runtime: 5-20 minutes depending on CPU.
All sensitivity runs use T=5 except the horizon study which uses T=1,5,10,15.
T=1 finishes in seconds; T=15 is the longest at up to 10 minutes.
"""

import sys
import os
import copy

# ── Import the frozen model function (do NOT run its __main__ block) ──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asset_location_optimized import solve_asset_location_optimization
from gurobipy import GRB

# ── Global constants ──────────────────────────────────────────────────────────
ASSETS   = ["Stock", "Bond", "Cash"]
ACCOUNTS = ["Taxable", "TDA", "Roth"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract(model, T):
    """Pull key numbers out of a solved model. Returns a plain dict."""
    if model.status != GRB.OPTIMAL:
        return {"status": model.status, "obj": None, "alloc": {}}
    obj = model.objVal
    alloc = {}
    for k in ACCOUNTS:
        total_k = sum(model.getVarByName(f"W[{k},{a},{T}]").X for a in ASSETS)
        alloc[k] = {}
        for a in ASSETS:
            v = model.getVarByName(f"W[{k},{a},{T}]").X
            alloc[k][a] = {
                "dollars": round(v, 2),
                "pct":     round(v / total_k * 100, 1) if total_k > 0 else 0.0,
            }
    total_port = sum(model.getVarByName(f"W[{k},{a},{T}]").X
                     for k in ACCOUNTS for a in ASSETS)
    return {
        "status": "OPTIMAL",
        "obj":    round(obj, 2),
        "total":  round(total_port, 2),
        "alloc":  alloc,
    }


def print_result(label, res, T):
    print(f"  [{label}]")
    if res["obj"] is None:
        print(f"    FAILED — Gurobi status {res['status']}")
        return
    print(f"    After-tax wealth: ${res['obj']:,.2f}  |  Total portfolio: ${res['total']:,.2f}")
    for k in ACCOUNTS:
        parts = "  ".join(f"{a}: {res['alloc'][k][a]['pct']:.0f}%"
                          for a in ASSETS)
        tot_k = sum(res['alloc'][k][a]['dollars'] for a in ASSETS)
        print(f"    {k:<10}: {parts}   (${tot_k:,.0f})")


def run_case(label, params):
    """Run a single optimisation case, print results, return result dict."""
    sep = "─" * 65
    print(f"\n{sep}\n  RUNNING: {label}\n{sep}")
    model = solve_asset_location_optimization(**params)
    res   = extract(model, params["T"])
    print_result(label, res, params["T"])
    return res


def base_params(T):
    """
    Standard parameter set used as the starting point for all sensitivity runs.
    Identical to Test Case 4 (T=15) logic, scaled to any T.
    """
    r = {(a, t): 0.075 if a == "Stock" else 0.035 if a == "Bond" else 0.020
         for t in range(1, T + 1) for a in ASSETS}
    d_div = {t: 0.020 for t in range(1, T + 1)}
    d_int = {(a, t): 0.035 if a == "Bond" else 0.020
             for t in range(1, T + 1) for a in ["Bond", "Cash"]}
    phi = {(a, t): 0.08 if a == "Stock" else 0.0
           for t in range(1, T + 1) for a in ASSETS}
    # Contributions grow at 2%/yr; no taxable contributions
    bar_c = {(k, t): (0.0 if k == "Taxable" else
                       22500.0 * 1.02 ** (t - 1) if k == "TDA" else
                       6500.0  * 1.02 ** (t - 1))
             for k in ACCOUNTS for t in range(1, T + 1)}
    # Decumulation begins in the last 5 years
    cutoff = max(1, T - 5)
    w_hat = {(k, t): (5000.0 if k == "Taxable" else 0.0) if t <= cutoff else
                      (25000.0 if k == "Taxable" else 20000.0 if k == "TDA" else 10000.0)
             for k in ACCOUNTS for t in range(1, T + 1)}
    I_base = {t: 80000.0 if t <= cutoff else 45000.0 for t in range(1, T + 1)}
    W_init = {
        ("Taxable", "Stock"): 150000.0, ("Taxable", "Bond"):  80000.0, ("Taxable", "Cash"): 40000.0,
        ("TDA",     "Stock"): 200000.0, ("TDA",     "Bond"): 120000.0, ("TDA",     "Cash"): 30000.0,
        ("Roth",    "Stock"):  60000.0, ("Roth",    "Bond"):  30000.0, ("Roth",    "Cash"): 10000.0,
    }
    B_init = {"Stock": 120000.0, "Bond": 80000.0, "Cash": 40000.0}
    Theta  = {"Stock": 0.60, "Bond": 0.30, "Cash": 0.10}
    return dict(
        T=T, assets=ASSETS, accounts=ACCOUNTS,
        r=r, d_div=d_div, d_int=d_int, phi=phi,
        tau_cg=0.15, yb=[0.0, 50000.0, float("inf")], m=[0.10, 0.24],
        I_base=I_base, bar_c=bar_c, w_hat=w_hat, Theta=Theta,
        W_init=W_init, B_init=B_init,
        tau_liq_T=0.22, discount_rho=0.04, M=1e7,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 1 — INVESTMENT HORIZON
# PIP question: "Investment horizon" impact on alpha
# Runs: T = 1, 5, 10, 15
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STUDY 1: INVESTMENT HORIZON  (T = 1, 5, 10, 15)")
print("=" * 65)

horizon_results = {}

# T=1 has a single period so we set all dicts with just t=1
p1 = base_params(1)
# Override to single-period sensible values
p1["r"]      = {("Stock", 1): 0.075, ("Bond", 1): 0.035, ("Cash", 1): 0.020}
p1["d_div"]  = {1: 0.02}
p1["d_int"]  = {("Bond", 1): 0.035, ("Cash", 1): 0.02}
p1["phi"]    = {("Stock", 1): 0.08, ("Bond", 1): 0.0, ("Cash", 1): 0.0}
p1["bar_c"]  = {("Taxable", 1): 0.0, ("TDA", 1): 22500.0, ("Roth", 1): 6500.0}
p1["w_hat"]  = {("Taxable", 1): 5000.0, ("TDA", 1): 0.0, ("Roth", 1): 0.0}
p1["I_base"] = {1: 80000.0}
p1["discount_rho"] = None
horizon_results[1] = run_case("Horizon T=1", p1)

for T in [5, 10, 15]:
    horizon_results[T] = run_case(f"Horizon T={T}", base_params(T))


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 2 — CAPITAL GAINS TAX RATE
# PIP question: "Tax rates" impact on alpha
# Runs: tau_cg = 0%, 15% (base), 23.8% (top federal rate)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STUDY 2: CAPITAL GAINS TAX RATE  (T=5 baseline)")
print("=" * 65)

tax_results = {}
for tau_cg, key in [(0.0, "0pct"), (0.15, "15pct"), (0.238, "238pct")]:
    p = base_params(5)
    p["tau_cg"] = tau_cg
    tax_results[key] = run_case(f"tau_cg = {tau_cg*100:.1f}%", p)


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 3 — STOCK TURNOVER (Tax Inefficiency)
# PIP question: "Tax efficiency (unrealised gains, etc.)"
# Runs: phi_stock = 0% (buy-and-hold), 8% (base), 20% (active), 40% (hyperactive)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STUDY 3: STOCK TURNOVER  (T=5 baseline)")
print("=" * 65)

turnover_results = {}
for phi_s, key in [(0.0, "0pct"), (0.08, "8pct"), (0.20, "20pct"), (0.40, "40pct")]:
    p = base_params(5)
    p["phi"] = {(a, t): phi_s if a == "Stock" else 0.0
                for t in range(1, 6) for a in ASSETS}
    turnover_results[key] = run_case(f"phi_stock = {phi_s*100:.0f}%", p)


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 4 — ACCOUNT BALANCE DISTRIBUTION
# PIP question: "Account balance distributions"
# Total portfolio = $350k in all cases. Only the account split changes.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STUDY 4: ACCOUNT BALANCE DISTRIBUTION  (T=5 baseline, total=$350k)")
print("=" * 65)


def build_W_init(tax_pct, tda_pct, roth_pct, total=350_000):
    """Distribute total across accounts; within each account use 60/30/10 split."""
    splits = {"Stock": 0.60, "Bond": 0.30, "Cash": 0.10}
    W = {}
    for k, pct in [("Taxable", tax_pct), ("TDA", tda_pct), ("Roth", roth_pct)]:
        for a in ASSETS:
            W[(k, a)] = total * pct * splits[a]
    return W


dist_cases = [
    ("Taxable-heavy  (50/35/15)", 0.50, 0.35, 0.15),
    ("Balanced       (49/37/14)", 0.49, 0.37, 0.14),
    ("TDA-heavy      (20/65/15)", 0.20, 0.65, 0.15),
    ("Roth-heavy     (20/30/50)", 0.20, 0.30, 0.50),
]
dist_results = {}
for name, tp, tdap, rp in dist_cases:
    p = base_params(5)
    W_init = build_W_init(tp, tdap, rp)
    p["W_init"] = W_init
    # Basis: start at 80% of Taxable position (20% embedded gain)
    p["B_init"] = {a: W_init[("Taxable", a)] * 0.80 for a in ASSETS}
    dist_results[name] = run_case(f"Distribution: {name}", p)


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 5 — CONTRIBUTIONS AND WITHDRAWALS
# PIP question: "Contributions & withdrawals"
# Four cases: none / contrib only / withdrawals only / full base
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STUDY 5: CONTRIBUTIONS & WITHDRAWALS  (T=5 baseline)")
print("=" * 65)

T_cf = 5
zero_c  = {(k, t): 0.0 for k in ACCOUNTS for t in range(1, T_cf + 1)}
zero_wd = {(k, t): 0.0 for k in ACCOUNTS for t in range(1, T_cf + 1)}

cf_cases = {
    "No contrib, no withdrawals": ("bar_c", zero_c, "w_hat", zero_wd),
    "Contributions only":         ("bar_c", None,   "w_hat", zero_wd),
    "Withdrawals only":           ("bar_c", zero_c, "w_hat", None),
    "Full base":                  ("bar_c", None,   "w_hat", None),
}
cashflow_results = {}
for case_name, (ck, cv, wk, wv) in cf_cases.items():
    p = base_params(T_cf)
    if cv is not None:
        p["bar_c"] = cv
    if wv is not None:
        p["w_hat"] = wv
    cashflow_results[case_name] = run_case(f"Cash flows: {case_name}", p)


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 6 — ASSET LOCATION ALPHA vs NAIVE BENCHMARK
# PIP question: "What proportion of alpha is attributable to post-tax allocation?"
# Naive: deterministic forward simulation forcing all accounts to hold Theta[a]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STUDY 6: ASSET LOCATION ALPHA vs NAIVE BENCHMARK  (T=15)")
print("=" * 65)


def naive_forward_sim(T, W_init, bar_c, w_hat,
                      tau_cg=0.15, tau_liq_T=0.22, tau_marg=0.24,
                      d_div_rate=0.02, d_int_bond=0.035, d_int_cash=0.02,
                      phi_stock=0.08):
    """
    Deterministic forward simulation with naive equal allocation.
    Every account is forced to hold exactly Theta each period.
    Returns the after-tax terminal objective value using the same formula
    as the optimiser: term_Tax + term_TDA + term_Roth.
    """
    Theta = {"Stock": 0.60, "Bond": 0.30, "Cash": 0.10}
    r_map = {"Stock": 0.075, "Bond": 0.035, "Cash": 0.020}

    W = copy.deepcopy(W_init)
    # Basis: initial Taxable positions at 80% of market value
    B = {a: W[("Taxable", a)] * 0.80 for a in ASSETS}

    for t in range(1, T + 1):
        # Step 1: Turnover gain on Taxable Stock
        gain_stock  = phi_stock * max(0, W[("Taxable", "Stock")] - B["Stock"])
        B["Stock"] += gain_stock

        # Step 2: Returns
        for k in ACCOUNTS:
            for a in ASSETS:
                W[(k, a)] *= (1 + r_map[a])
        DIV = W[("Taxable", "Stock")] / (1 + r_map["Stock"]) * d_div_rate
        B["Stock"] += DIV
        # Bond/Cash basis = market value (no gains)
        B["Bond"] = W[("Taxable", "Bond")]
        B["Cash"] = W[("Taxable", "Cash")]

        # Steps 3-4: Annual tax + pro-rata liquidation from Taxable
        INT_t = (W[("Taxable", "Bond")] / (1 + r_map["Bond"]) * d_int_bond +
                 W[("Taxable", "Cash")] / (1 + r_map["Cash"]) * d_int_cash)
        Tax_ann = tau_cg * (gain_stock + DIV) + tau_marg * INT_t
        V_Tax = sum(W[("Taxable", a)] for a in ASSETS)
        sell_frac = Tax_ann / V_Tax if V_Tax > 0 else 0.0
        for a in ASSETS:
            sell_a = W[("Taxable", a)] * sell_frac
            basis_frac = B[a] / W[("Taxable", a)] if W[("Taxable", a)] > 0 else 1.0
            B[a] -= sell_a * basis_frac
            W[("Taxable", a)] -= sell_a

        # Step 5: Contributions split by Theta
        for k in ACCOUNTS:
            c_k = bar_c.get((k, t), 0.0)
            for a in ASSETS:
                added = c_k * Theta[a]
                W[(k, a)] += added
                if k == "Taxable":
                    B[a] += added  # full basis on new purchases

        # Step 6: Withdrawals (gross up TDA)
        gross_TDA = (w_hat.get(("TDA", t), 0.0) / (1 - tau_marg)
                     if w_hat.get(("TDA", t), 0.0) > 0 else 0.0)
        actual_wd = {
            "Taxable": w_hat.get(("Taxable", t), 0.0),
            "TDA":     gross_TDA,
            "Roth":    w_hat.get(("Roth", t), 0.0),
        }
        for k in ACCOUNTS:
            wd_k  = actual_wd[k]
            V_k   = sum(W[(k, a)] for a in ASSETS)
            if V_k > 0 and wd_k > 0:
                for a in ASSETS:
                    frac = W[(k, a)] / V_k
                    withdrawn = min(wd_k * frac, W[(k, a)])
                    if k == "Taxable" and W[("Taxable", a)] > 0:
                        B[a] -= withdrawn * (B[a] / W[("Taxable", a)])
                    W[(k, a)] = max(0.0, W[(k, a)] - withdrawn)

        # Step 7: Rebalance every account back to Theta (naive)
        for k in ACCOUNTS:
            V_k = sum(W[(k, a)] for a in ASSETS)
            for a in ASSETS:
                W[(k, a)] = V_k * Theta[a]
            if k == "Taxable":
                for a in ASSETS:
                    B[a] = min(B.get(a, 0.0), W[("Taxable", a)])

    # Terminal objective
    term_Tax  = sum(W[("Taxable", a)] - tau_cg * max(0, W[("Taxable", a)] - B[a])
                    for a in ASSETS)
    term_TDA  = sum(W[("TDA",  a)] * (1 - tau_liq_T) for a in ASSETS)
    term_Roth = sum(W[("Roth", a)] for a in ASSETS)
    return round(term_Tax + term_TDA + term_Roth, 2)


W_init_15 = {
    ("Taxable", "Stock"): 150000.0, ("Taxable", "Bond"):  80000.0, ("Taxable", "Cash"): 40000.0,
    ("TDA",     "Stock"): 200000.0, ("TDA",     "Bond"): 120000.0, ("TDA",     "Cash"): 30000.0,
    ("Roth",    "Stock"):  60000.0, ("Roth",    "Bond"):  30000.0, ("Roth",    "Cash"): 10000.0,
}
bar_c_15 = {(k, t): (0.0 if k == "Taxable" else
                      22500.0 * 1.02 ** (t - 1) if k == "TDA" else
                      6500.0  * 1.02 ** (t - 1))
            for k in ACCOUNTS for t in range(1, 16)}
w_hat_15 = {(k, t): (5000.0 if k == "Taxable" else 0.0) if t <= 10 else
                     (25000.0 if k == "Taxable" else 20000.0 if k == "TDA" else 10000.0)
            for k in ACCOUNTS for t in range(1, 16)}

print("  Computing naive benchmark (deterministic simulation)...")
naive_15 = naive_forward_sim(T=15, W_init=W_init_15, bar_c=bar_c_15, w_hat=w_hat_15)

optimal_15 = horizon_results.get(15, {}).get("obj")
if optimal_15 is None:
    print("  T=15 optimal not available from Study 1, re-running...")
    model_tmp = solve_asset_location_optimization(**base_params(15))
    optimal_15 = model_tmp.objVal if model_tmp.status == GRB.OPTIMAL else None

alpha_dollar = round(optimal_15 - naive_15, 2) if optimal_15 else None
alpha_pct    = round(alpha_dollar / naive_15 * 100, 2) if (naive_15 and alpha_dollar is not None) else None

print(f"\n  Naive benchmark (T=15, forced Theta allocation): ${naive_15:>14,.2f}")
print(f"  Optimal (T=15, asset location optimised):        ${optimal_15:>14,.2f}" if optimal_15 else "  Optimal: N/A")
print(f"  Asset Location Alpha (dollars):                  ${alpha_dollar:>14,.2f}" if alpha_dollar else "")
print(f"  Asset Location Alpha (percent):                   {alpha_pct:>12.2f}%" if alpha_pct else "")

alpha_results = {
    "naive_15":     naive_15,
    "optimal_15":   optimal_15,
    "alpha_dollar": alpha_dollar,
    "alpha_pct":    alpha_pct,
}


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RAW RESULTS
# ─────────────────────────────────────────────────────────────────────────────

all_results = {
    "horizon":   {str(k): v for k, v in horizon_results.items()},
    "tax_rate":  tax_results,
    "turnover":  turnover_results,
    "dist":      dist_results,
    "cashflow":  cashflow_results,
    "alpha":     alpha_results,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_results_raw.py")
with open(out_path, "w") as f:
    f.write("# Auto-generated by sensitivity_runner.py — do not edit manually.\n")
    f.write("# Import this file in presentation_builder.py\n\n")
    f.write(f"results = {repr(all_results)}\n")

print(f"\n\nResults saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("CLEAN SUMMARY — COPY THESE NUMBERS INTO YOUR PRESENTATION")
print("=" * 65)

print("\nSTUDY 1 — Investment Horizon:")
print(f"  {'Horizon':<10}  {'After-Tax Wealth ($)':>22}")
for T in [1, 5, 10, 15]:
    obj = horizon_results.get(T, {}).get("obj")
    print(f"  T={T:<8}  ${obj:>20,.2f}" if obj else f"  T={T}: no result")

print("\nSTUDY 2 — Capital Gains Tax Rate (T=5):")
print(f"  {'tau_cg':<12}  {'After-Tax Wealth ($)':>22}")
labels2 = {"0pct": "0%", "15pct": "15%", "238pct": "23.8%"}
for k, v in tax_results.items():
    obj = v.get("obj")
    print(f"  {labels2[k]:<12}  ${obj:>20,.2f}" if obj else f"  {labels2[k]}: no result")

print("\nSTUDY 3 — Stock Turnover (T=5):")
print(f"  {'phi_stock':<12}  {'After-Tax Wealth ($)':>22}")
labels3 = {"0pct": "0%", "8pct": "8%", "20pct": "20%", "40pct": "40%"}
for k, v in turnover_results.items():
    obj = v.get("obj")
    print(f"  {labels3[k]:<12}  ${obj:>20,.2f}" if obj else f"  {labels3[k]}: no result")

print("\nSTUDY 4 — Account Balance Distribution (T=5):")
print(f"  {'Case':<30}  {'After-Tax Wealth ($)':>22}")
for k, v in dist_results.items():
    obj = v.get("obj")
    print(f"  {k:<30}  ${obj:>20,.2f}" if obj else f"  {k}: no result")

print("\nSTUDY 5 — Contributions & Withdrawals (T=5):")
print(f"  {'Case':<35}  {'After-Tax Wealth ($)':>22}")
for k, v in cashflow_results.items():
    obj = v.get("obj")
    print(f"  {k:<35}  ${obj:>20,.2f}" if obj else f"  {k}: no result")

print("\nSTUDY 6 — Asset Location Alpha (T=15):")
print(f"  Naive (equal allocation):   ${naive_15:>14,.2f}")
print(f"  Optimal (location):         ${optimal_15:>14,.2f}" if optimal_15 else "  Optimal: N/A")
print(f"  Alpha ($):                  ${alpha_dollar:>14,.2f}" if alpha_dollar else "")
print(f"  Alpha (%):                   {alpha_pct:>12.2f}%" if alpha_pct else "")

print("\n" + "=" * 65)
print("ALL DONE. Now run: python presentation_builder.py")
print("=" * 65)
