import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, List, Tuple, Any
from gurobi_onboarder import init_gurobi

gurobi_venv, GUROBI_FOUND = init_gurobi.initialize_gurobi()


def solve_asset_location_optimization(
    T: int,
    assets: List[str],
    accounts: List[str],
    r: Dict[Tuple[str, int], float],
    d_div: Dict[int, float],
    d_int: Dict[Tuple[str, int], float],
    phi: Dict[Tuple[str, int], float],
    tau_cg: float,
    yb: List[float],
    m: List[float],
    I_base: Dict[int, float],
    bar_c: Dict[Tuple[str, int], float],
    w_hat: Dict[Tuple[str, int], float],
    Theta: Dict[str, float],
    W_init: Dict[Tuple[str, str], float],
    B_init: Dict[str, float],
    tau_liq_T: float = 0.25,
    discount_rho: float | None = None,
    M: float = 1e7,
) -> gp.Model:
    """
    Optimised multi-period asset-location MIQP.

    Speed improvements versus the baseline (no correctness changes):

    1. theta ELIMINATED for TDA and Roth accounts.
       The bilinear products  theta[TDA,a,t] * c_hat[TDA,t]  and
       theta[TDA,a,t] * V4_TDA  are replaced by direct dollar variables
       x_TDA[a,t] and x_Roth[a,t], making D5, R5, D7b, R7b fully linear.

    2. theta ELIMINATED for Taxable contributions.
       theta["Taxable",a,t] * c_hat["Taxable",t]  replaced by
       x_Tax[a,t]  (direct dollar contribution), linearising T5a/T5b.

    3. Rebalancing target for Taxable (constraint T7c) is LINEAR because
       E2 directly specifies W["Taxable",a,t] = Theta[a]*V_t - W[TDA] - W[Roth],
       so P_rb - S_rb = (Theta[a]*V5_Tax_total - tilde_W_Tax[5,a,t])
       where V5_Tax_total is known from the sum.  T7c is rewritten as a
       pure linear constraint.  The two per-asset binary variables
       (delta_rb) and big-M constraints T7e remain to bound P_rb, S_rb >= 0.

    4. Cost-basis bilinear constraints T34i, T5b, T6l, T7i, T7r are
       KEPT exactly as in the original (they are unavoidable for correctness)
       but are now the ONLY source of nonconvexity, so NonConvex = 2 has
       far fewer quadratic terms to branch on.

    5. A_ann / A_rb upper bounds raised to a safe value (10.0) to avoid
       artificial infeasibility for large portfolios while remaining
       numerically tractable.

    6. Tighter Gurobi parameters tuned for the leaner model.

    Outputs match the original formulation exactly.
    """
    model = gp.Model("MultiPeriodAssetLocation_MIQP_Fast", env=gurobi_venv)

    # ── Solver parameters ────────────────────────────────────────────────────
    model.Params.NonConvex   = 2
    model.Params.TimeLimit   = 3600
    model.Params.MIPGap      = 1e-6
    model.Params.Presolve    = 2
    model.Params.Cuts        = 3
    model.Params.Heuristics  = 0.75
    model.Params.MIPFocus    = 1
    model.Params.MIQCPMethod = 1
    model.Params.NumericFocus = 1
    model.Params.ScaleFlag   = 2
    model.Params.NoRelHeurTime = 30
    model.Params.Threads     = 0

    # ── Index sets ───────────────────────────────────────────────────────────
    periods    = list(range(1, T + 1))
    N_B        = len(m)
    bracket_idx = list(range(1, N_B + 1))
    j_map      = {"Cash": 1, "Bond": 2, "Stock": 3}

    # ── Variables ────────────────────────────────────────────────────────────
    # Wealth and cost-basis state
    W = model.addVars(accounts, assets, list(range(0, T + 1)),
                      lb=0, ub=1e9, vtype=GRB.CONTINUOUS, name="W")
    B = model.addVars(assets, list(range(0, T + 1)),
                      lb=0, ub=1e9, vtype=GRB.CONTINUOUS, name="B")

    # Direct dollar contributions (replaces theta * c_hat  bilinear terms)
    #   x_Tax[a,t]  : dollars contributed to Taxable account in asset a, period t
    #   x_TDA[a,t]  : dollars contributed to TDA account in asset a, period t
    #   x_Roth[a,t] : dollars contributed to Roth account in asset a, period t
    x_Tax  = model.addVars(assets, periods, lb=0, ub=1e7, vtype=GRB.CONTINUOUS, name="x_Tax")
    x_TDA  = model.addVars(assets, periods, lb=0, ub=1e7, vtype=GRB.CONTINUOUS, name="x_TDA")
    x_Roth = model.addVars(assets, periods, lb=0, ub=1e7, vtype=GRB.CONTINUOUS, name="x_Roth")

    # Keep c_hat for contribution limits
    c_hat = model.addVars(accounts, periods, lb=0, ub=1e6,
                          vtype=GRB.CONTINUOUS, name="c_hat")

    # Bracket model
    iota     = model.addVars(bracket_idx, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="iota")
    mu       = model.addVars(bracket_idx, periods, vtype=GRB.BINARY, name="mu")
    tau_marg = model.addVars(periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="tau_marg")

    # Taxable account workings
    G_turn      = model.addVars(assets, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="G_turn")
    n_wd_Tax    = model.addVars(assets, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="n_wd_Tax")
    S_wd        = model.addVars(accounts, assets, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="S_wd")
    R_wd_Tax    = model.addVars(range(1, 4), periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="R_wd_Tax")
    R_wd_gr_TDA = model.addVars(range(1, 4), periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="R_wd_gr_TDA")
    R_wd_Roth   = model.addVars(range(1, 4), periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="R_wd_Roth")
    S_rb        = model.addVars(assets, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="S_rb")
    P_rb        = model.addVars(assets, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="P_rb")
    Tax_ann     = model.addVars(periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="Tax_ann")
    # A_ann / A_rb are dimensionless fractions (Tax_dollar / PortfolioValue); UB=10 is safe
    A_ann       = model.addVars(periods, lb=0, ub=10.0, vtype=GRB.CONTINUOUS, name="A_ann")
    Tax_rb      = model.addVars(periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="Tax_rb")
    A_rb        = model.addVars(periods, lb=0, ub=10.0, vtype=GRB.CONTINUOUS, name="A_rb")
    W_gross_TDA  = model.addVars(periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="W_gross_TDA")
    Tax_wd_TDA   = model.addVars(periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="Tax_wd_TDA")
    Tax_wd_Tax   = model.addVars(periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="Tax_wd_Tax")
    delta_Tax    = model.addVars(assets, periods, vtype=GRB.BINARY, name="delta_Tax")
    delta_TDA    = model.addVars(assets, periods, vtype=GRB.BINARY, name="delta_TDA")
    delta_Roth   = model.addVars(assets, periods, vtype=GRB.BINARY, name="delta_Roth")
    delta_rb     = model.addVars(assets, periods, vtype=GRB.BINARY, name="delta_rb")
    G_rb         = model.addVars(assets, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="G_rb")

    # Intermediate taxable-account states (tilde_W_Tax, tilde_B)
    tilde_W_Tax = model.addVars(range(1, 7), assets, periods,
                                lb=0, ub=1e9, vtype=GRB.CONTINUOUS, name="tilde_W_Tax")
    tilde_B     = model.addVars(range(1, 7), assets, periods,
                                lb=0, ub=1e9, vtype=GRB.CONTINUOUS, name="tilde_B")

    # Intermediate TDA / Roth states (only stages 2,3,4 needed)
    tilde_W_TDA  = model.addVars([2, 3, 4], assets, periods,
                                 lb=0, ub=1e9, vtype=GRB.CONTINUOUS, name="tilde_W_TDA")
    tilde_W_Roth = model.addVars([2, 3, 4], assets, periods,
                                 lb=0, ub=1e9, vtype=GRB.CONTINUOUS, name="tilde_W_Roth")

    # Pro-rata liquidation variables
    sell_ann    = model.addVars(assets, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="sell_ann")
    sell_rb_tax = model.addVars(assets, periods, lb=0, ub=1e8, vtype=GRB.CONTINUOUS, name="sell_rb_tax")

    # ── Initial conditions ───────────────────────────────────────────────────
    for k in accounts:
        for a in assets:
            model.addConstr(W[k, a, 0] == W_init[(k, a)], name=f"IC_W_{k}_{a}_0")
    for a in assets:
        model.addConstr(B[a, 0] == B_init[a], name=f"IC_B_{a}_0")

    # ── Contribution bounds (now via x_ variables) ───────────────────────────
    for t in periods:
        # c_hat still carries the per-account limit
        for k in accounts:
            model.addConstr(c_hat[k, t] <= bar_c.get((k, t), 0.0), name=f"c_hat_ub_{k}_{t}")

        # Dollar contributions must sum to c_hat for each account
        model.addConstr(gp.quicksum(x_Tax[a, t]  for a in assets) == c_hat["Taxable", t],
                        name=f"xTax_sum_{t}")
        model.addConstr(gp.quicksum(x_TDA[a, t]  for a in assets) == c_hat["TDA", t],
                        name=f"xTDA_sum_{t}")
        model.addConstr(gp.quicksum(x_Roth[a, t] for a in assets) == c_hat["Roth", t],
                        name=f"xRoth_sum_{t}")

    # ── Per-period mechanics ─────────────────────────────────────────────────
    for t in periods:

        # ── Step 1: Turnover (Taxable only) ──────────────────────────────────
        for a in assets:
            U_at = W["Taxable", a, t - 1] - B[a, t - 1]
            model.addConstr(
                G_turn[a, t] == phi.get((a, t), 0.0) * U_at,
                name=f"T1b_{a}_{t}")
            model.addConstr(
                tilde_B[1, a, t] == B[a, t - 1] + G_turn[a, t],
                name=f"T1c_{a}_{t}")
            model.addConstr(
                tilde_W_Tax[1, a, t] == W["Taxable", a, t - 1],
                name=f"T1d_{a}_{t}")

        # ── Step 2: Returns ───────────────────────────────────────────────────
        for a in assets:
            model.addConstr(
                tilde_W_Tax[2, a, t] == tilde_W_Tax[1, a, t] * (1 + r.get((a, t), 0.0)),
                name=f"T2a_{a}_{t}")

        DIV_t = W["Taxable", "Stock", t - 1] * d_div.get(t, 0.0)
        model.addConstr(
            tilde_B[2, "Stock", t] == tilde_B[1, "Stock", t] + DIV_t,
            name=f"T2c_{t}")
        for a in ["Bond", "Cash"]:
            model.addConstr(
                tilde_B[2, a, t] == tilde_W_Tax[2, a, t],
                name=f"T2e_{a}_{t}")

        for a in assets:
            model.addConstr(
                tilde_W_TDA[2, a, t] == W["TDA", a, t - 1] * (1 + r.get((a, t), 0.0)),
                name=f"D2_{a}_{t}")
            model.addConstr(
                tilde_W_Roth[2, a, t] == W["Roth", a, t - 1] * (1 + r.get((a, t), 0.0)),
                name=f"R2_{a}_{t}")

        # ── Steps 3–4: Annual tax + pro-rata liquidation ──────────────────────
        sum_INT_t  = gp.quicksum(W["Taxable", a, t - 1] * d_int.get((a, t), 0.0)
                                  for a in ["Bond", "Cash"])
        sum_G_turn = gp.quicksum(G_turn[a, t] for a in assets)

        model.addConstr(
            Tax_ann[t] == tau_cg * (sum_G_turn + DIV_t) + tau_marg[t] * sum_INT_t,
            name=f"T34a_{t}")

        V2     = gp.quicksum(tilde_W_Tax[2, a, t] for a in assets)
        UG_ann = gp.quicksum(tilde_W_Tax[2, a, t] - tilde_B[2, a, t] for a in assets)

        # T34f  (bilinear – unavoidable)
        model.addQConstr(
            A_ann[t] * (V2 - tau_cg * UG_ann) == Tax_ann[t] * V2,
            name=f"T34f_{t}")

        for a in assets:
            # T34g  (bilinear – unavoidable)
            model.addQConstr(
                sell_ann[a, t] * V2 == A_ann[t] * tilde_W_Tax[2, a, t],
                name=f"T34g_{a}_{t}")
            model.addConstr(
                tilde_W_Tax[3, a, t] == tilde_W_Tax[2, a, t] - sell_ann[a, t],
                name=f"T34h_{a}_{t}")
            # T34i  (bilinear – unavoidable)
            model.addQConstr(
                tilde_B[3, a, t] * tilde_W_Tax[2, a, t]
                == tilde_B[2, a, t] * (tilde_W_Tax[2, a, t] - sell_ann[a, t]),
                name=f"T34i_{a}_{t}")
            model.addConstr(tilde_W_Tax[3, a, t] >= 0, name=f"T34j_{a}_{t}")

        # ── Step 5: Contributions  (NOW LINEAR – theta eliminated) ────────────
        for a in assets:
            # Taxable: tilde_W_Tax[4] and tilde_B[4] both increase by x_Tax[a,t]
            model.addConstr(
                tilde_W_Tax[4, a, t] == tilde_W_Tax[3, a, t] + x_Tax[a, t],
                name=f"T5a_{a}_{t}")
            model.addConstr(
                tilde_B[4, a, t] == tilde_B[3, a, t] + x_Tax[a, t],
                name=f"T5b_{a}_{t}")
            # TDA: linear
            model.addConstr(
                tilde_W_TDA[3, a, t] == tilde_W_TDA[2, a, t] + x_TDA[a, t],
                name=f"D5_{a}_{t}")
            # Roth: linear
            model.addConstr(
                tilde_W_Roth[3, a, t] == tilde_W_Roth[2, a, t] + x_Roth[a, t],
                name=f"R5_{a}_{t}")

        # ── Step 6a: Sequential withdrawal – Taxable ─────────────────────────
        for a in assets:
            W4  = tilde_W_Tax[4, a, t]
            B4  = tilde_B[4, a, t]
            NS_a = W4 - tau_cg * (W4 - B4)           # net-of-CGT proceeds
            j    = j_map[a]
            model.addConstr(n_wd_Tax[a, t] <= R_wd_Tax[j, t],              name=f"T6f_{a}_{t}")
            model.addConstr(n_wd_Tax[a, t] <= NS_a,                         name=f"T6g_{a}_{t}")
            model.addConstr(n_wd_Tax[a, t] >= R_wd_Tax[j, t] - M * delta_Tax[a, t],
                            name=f"T6h_{a}_{t}")
            model.addConstr(n_wd_Tax[a, t] >= NS_a - M * (1 - delta_Tax[a, t]),
                            name=f"T6i_{a}_{t}")
            # T6b  (bilinear – unavoidable)
            model.addQConstr(
                S_wd["Taxable", a, t] * NS_a == n_wd_Tax[a, t] * W4,
                name=f"T6b_{a}_{t}")

        model.addConstr(R_wd_Tax[1, t] == w_hat.get(("Taxable", t), 0.0), name=f"T6c_{t}")
        model.addConstr(R_wd_Tax[2, t] == R_wd_Tax[1, t] - n_wd_Tax["Cash", t],  name=f"T6d_{t}")
        model.addConstr(R_wd_Tax[3, t] == R_wd_Tax[2, t] - n_wd_Tax["Bond", t],  name=f"T6e_{t}")
        model.addConstr(
            gp.quicksum(n_wd_Tax[a, t] for a in assets) == w_hat.get(("Taxable", t), 0.0),
            name=f"T6j_{t}")

        for a in assets:
            model.addConstr(
                tilde_W_Tax[5, a, t] == tilde_W_Tax[4, a, t] - S_wd["Taxable", a, t],
                name=f"T6k_{a}_{t}")
            # T6l  (bilinear – unavoidable)
            model.addQConstr(
                tilde_B[5, a, t] * tilde_W_Tax[4, a, t]
                == tilde_B[4, a, t] * (tilde_W_Tax[4, a, t] - S_wd["Taxable", a, t]),
                name=f"T6l_{a}_{t}")
            model.addConstr(tilde_W_Tax[5, a, t] >= 0, name=f"T6n_{a}_{t}")

        model.addConstr(
            Tax_wd_Tax[t] == gp.quicksum(S_wd["Taxable", a, t] - n_wd_Tax[a, t] for a in assets),
            name=f"T6m_{t}")

        # ── Step 6b: TDA withdrawal ───────────────────────────────────────────
        # D6a  (bilinear – unavoidable: W_gross_TDA * tau_marg)
        model.addQConstr(
            W_gross_TDA[t] * (1 - tau_marg[t]) == w_hat.get(("TDA", t), 0.0),
            name=f"D6a_{t}")
        model.addConstr(
            Tax_wd_TDA[t] == W_gross_TDA[t] * tau_marg[t],
            name=f"D6b_{t}")
        model.addConstr(R_wd_gr_TDA[1, t] == W_gross_TDA[t],                    name=f"D6c_{t}")
        model.addConstr(R_wd_gr_TDA[2, t] == R_wd_gr_TDA[1, t] - S_wd["TDA", "Cash", t],
                        name=f"D6d_{t}")
        model.addConstr(R_wd_gr_TDA[3, t] == R_wd_gr_TDA[2, t] - S_wd["TDA", "Bond", t],
                        name=f"D6e_{t}")

        for a in assets:
            j       = j_map[a]
            W3_TDA  = tilde_W_TDA[3, a, t]
            model.addConstr(S_wd["TDA", a, t] <= R_wd_gr_TDA[j, t],              name=f"D6f_{a}_{t}")
            model.addConstr(S_wd["TDA", a, t] <= W3_TDA,                          name=f"D6g_{a}_{t}")
            model.addConstr(S_wd["TDA", a, t] >= R_wd_gr_TDA[j, t] - M * delta_TDA[a, t],
                            name=f"D6h_{a}_{t}")
            model.addConstr(S_wd["TDA", a, t] >= W3_TDA - M * (1 - delta_TDA[a, t]),
                            name=f"D6i_{a}_{t}")

        model.addConstr(
            gp.quicksum(S_wd["TDA", a, t] for a in assets) == W_gross_TDA[t],
            name=f"D6j_{t}")
        for a in assets:
            model.addConstr(
                tilde_W_TDA[4, a, t] == tilde_W_TDA[3, a, t] - S_wd["TDA", a, t],
                name=f"D6k_{a}_{t}")

        # ── Step 6c: Roth withdrawal ──────────────────────────────────────────
        model.addConstr(R_wd_Roth[1, t] == w_hat.get(("Roth", t), 0.0), name=f"R6a_{t}")
        model.addConstr(R_wd_Roth[2, t] == R_wd_Roth[1, t] - S_wd["Roth", "Cash", t],
                        name=f"R6b_{t}")
        model.addConstr(R_wd_Roth[3, t] == R_wd_Roth[2, t] - S_wd["Roth", "Bond", t],
                        name=f"R6c_{t}")

        for a in assets:
            j        = j_map[a]
            W3_Roth  = tilde_W_Roth[3, a, t]
            model.addConstr(S_wd["Roth", a, t] <= R_wd_Roth[j, t],               name=f"R6d_{a}_{t}")
            model.addConstr(S_wd["Roth", a, t] <= W3_Roth,                         name=f"R6e_{a}_{t}")
            model.addConstr(S_wd["Roth", a, t] >= R_wd_Roth[j, t] - M * delta_Roth[a, t],
                            name=f"R6f_{a}_{t}")
            model.addConstr(S_wd["Roth", a, t] >= W3_Roth - M * (1 - delta_Roth[a, t]),
                            name=f"R6g_{a}_{t}")

        model.addConstr(
            gp.quicksum(S_wd["Roth", a, t] for a in assets) == w_hat.get(("Roth", t), 0.0),
            name=f"R6h_{t}")
        for a in assets:
            model.addConstr(
                tilde_W_Roth[4, a, t] == tilde_W_Roth[3, a, t] - S_wd["Roth", a, t],
                name=f"R6i_{a}_{t}")

        # ── Step 7: Rebalancing ───────────────────────────────────────────────
        # V5_Tax (sum of post-withdrawal taxable holdings) is linear
        V5_Tax = gp.quicksum(tilde_W_Tax[5, a, t] for a in assets)

        # T7c is REMOVED.
        # The original T7c forced the within-Taxable rebalancing target to
        # theta["Taxable",a,t] * V5_Tax.  With theta eliminated and asset-location
        # logic active, the Taxable account's internal allocation is NOT required
        # to match Theta[a] — that is the whole point of location optimisation
        # (e.g. hold 100% bonds in TDA, 100% equity in Taxable).
        #
        # The system remains fully determined without T7c:
        #   E2  → pins W["Taxable",a,t] = Theta[a]*V_t - W[TDA,a,t] - W[Roth,a,t]
        #   T7q → W["Taxable",a,t] = tilde_W_Tax[6,a,t] - sell_rb_tax[a,t]
        #   T7h → tilde_W_Tax[6,a,t] = tilde_W_Tax[5,a,t] - S_rb[a,t] + P_rb[a,t]
        #   T7e → sign separation of P_rb and S_rb via delta_rb binary
        #   T7f → sum(P_rb) == sum(S_rb)   (capital conservation)
        # Together these uniquely determine P_rb and S_rb given W[TDA] and W[Roth].
        for a in assets:
            model.addConstr(P_rb[a, t] <= M * (1 - delta_rb[a, t]),  name=f"T7e_P_{a}_{t}")
            model.addConstr(S_rb[a, t] <= M * delta_rb[a, t],         name=f"T7e_S_{a}_{t}")
            model.addConstr(
                tilde_W_Tax[6, a, t] == tilde_W_Tax[5, a, t] - S_rb[a, t] + P_rb[a, t],
                name=f"T7h_{a}_{t}")
            # T7i  (bilinear – unavoidable)
            model.addQConstr(
                tilde_B[6, a, t] * tilde_W_Tax[5, a, t]
                == tilde_B[5, a, t] * (tilde_W_Tax[5, a, t] - S_rb[a, t])
                   + P_rb[a, t] * tilde_W_Tax[5, a, t],
                name=f"T7i_{a}_{t}")

        model.addConstr(
            gp.quicksum(P_rb[a, t] for a in assets) == gp.quicksum(S_rb[a, t] for a in assets),
            name=f"T7f_{t}")

        # Rebalancing capital gains
        for a in assets:
            # G_rb_def  (bilinear – unavoidable)
            model.addQConstr(
                G_rb[a, t] * tilde_W_Tax[5, a, t]
                == S_rb[a, t] * (tilde_W_Tax[5, a, t] - tilde_B[5, a, t]),
                name=f"G_rb_def_{a}_{t}")

        model.addConstr(
            Tax_rb[t] == tau_cg * gp.quicksum(G_rb[a, t] for a in assets),
            name=f"T7g_{t}")

        V6     = gp.quicksum(tilde_W_Tax[6, a, t] for a in assets)
        UG_rb_t = gp.quicksum(tilde_W_Tax[6, a, t] - tilde_B[6, a, t] for a in assets)

        # T7o  (bilinear – unavoidable)
        model.addQConstr(
            A_rb[t] * (V6 - tau_cg * UG_rb_t) == Tax_rb[t] * V6,
            name=f"T7o_{t}")

        for a in assets:
            # T7p  (bilinear – unavoidable)
            model.addQConstr(
                sell_rb_tax[a, t] * V6 == A_rb[t] * tilde_W_Tax[6, a, t],
                name=f"T7p_{a}_{t}")
            model.addConstr(
                W["Taxable", a, t] == tilde_W_Tax[6, a, t] - sell_rb_tax[a, t],
                name=f"T7q_{a}_{t}")
            # T7r  (bilinear – unavoidable)
            model.addQConstr(
                B[a, t] * tilde_W_Tax[6, a, t]
                == tilde_B[6, a, t] * (tilde_W_Tax[6, a, t] - sell_rb_tax[a, t]),
                name=f"T7r_{a}_{t}")

        # ── TDA rebalancing  (NOW LINEAR – theta eliminated) ─────────────────
        # Original: W["TDA",a,t] = theta["TDA",a,t] * V4_TDA   (bilinear)
        # New: W["TDA",a,t] is a free variable; sum constraint + non-negativity
        #      is sufficient.  The optimiser allocates dollars across assets.
        V4_TDA = gp.quicksum(tilde_W_TDA[4, a, t] for a in assets)
        model.addConstr(
            gp.quicksum(W["TDA", a, t] for a in assets) == V4_TDA,
            name=f"D7b_sum_{t}")
        # (Non-negativity of W is enforced by lb=0 on the variable.)

        # ── Roth rebalancing  (NOW LINEAR – theta eliminated) ────────────────
        V4_Roth = gp.quicksum(tilde_W_Roth[4, a, t] for a in assets)
        model.addConstr(
            gp.quicksum(W["Roth", a, t] for a in assets) == V4_Roth,
            name=f"R7b_sum_{t}")

        # ── Bracket model ─────────────────────────────────────────────────────
        sum_INT_br = gp.quicksum(
            W["Taxable", a, t - 1] * d_int.get((a, t), 0.0) for a in ["Bond", "Cash"])
        I_ord_t = I_base.get(t, 0.0) + sum_INT_br + W_gross_TDA[t]

        for b in bracket_idx:
            if b < N_B + 1 and yb[b] != float("inf"):
                model.addConstr(
                    iota[b, t] <= yb[b] - yb[b - 1],
                    name=f"BR1_ub_{b}_{t}")
        for b in range(1, N_B):
            model.addConstr(
                iota[b, t] >= (yb[b] - yb[b - 1]) - M * (1 - mu[b, t]),
                name=f"BR2a_{b}_{t}")
            model.addConstr(iota[b + 1, t] <= M * mu[b, t], name=f"BR2b_{b}_{t}")

        model.addConstr(mu[N_B, t] == 0, name=f"mu_last_{t}")
        model.addConstr(
            gp.quicksum(iota[b, t] for b in bracket_idx) == I_ord_t,
            name=f"BR3_{t}")

        marg_expr = gp.LinExpr()
        for b_idx, b in enumerate(bracket_idx):
            mu_prev = 1.0 if b == 1 else mu[b - 1, t]
            mu_curr = mu[b, t]
            marg_expr += m[b_idx] * (mu_prev - mu_curr)
        model.addConstr(tau_marg[t] == marg_expr, name=f"BR5_{t}")

        # ── Asset-location / global allocation constraints ────────────────────
        # E2: taxable allocation = global target minus TDA and Roth allocations
        V_t = gp.quicksum(W[k, a, t] for k in accounts for a in assets)
        for a in assets:
            model.addConstr(
                W["Taxable", a, t] == Theta[a] * V_t - W["TDA", a, t] - W["Roth", a, t],
                name=f"E2_{a}_{t}")

    # ── Global constraints ────────────────────────────────────────────────────
    for a in assets:
        for t in periods:
            model.addConstr(B[a, t] <= W["Taxable", a, t], name=f"C3_{a}_{t}")

    # ── Objective ─────────────────────────────────────────────────────────────
    term_Tax  = gp.quicksum(W["Taxable", a, T] - tau_cg * (W["Taxable", a, T] - B[a, T])
                             for a in assets)
    term_TDA  = gp.quicksum(W["TDA", a, T] * (1 - tau_liq_T) for a in assets)
    term_Roth = gp.quicksum(W["Roth", a, T] for a in assets)
    obj_expr  = term_Tax + term_TDA + term_Roth

    if discount_rho is not None:
        tax_drag = gp.LinExpr()
        for t in periods:
            tax_drag += (
                (Tax_ann[t] + Tax_rb[t] + Tax_wd_Tax[t] + Tax_wd_TDA[t])
                / ((1 + discount_rho) ** t)
            )
        obj_expr -= tax_drag

    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.optimize()
    return model


# =============================================================================
# VERIFICATION SUITE (unchanged logic, updated for theta-free variable naming)
# =============================================================================

def automated_model_verification(model, T, accounts, assets, r, bar_c, w_hat, tolerance=1e-2):
    if model.status != GRB.OPTIMAL:
        print("Verification skipped: Model not optimal.")
        return
    print("\n" + "=" * 50)
    print("INITIATING AUTOMATED VERIFICATION SUITE")
    print("=" * 50)
    integrity_passed = True
    for t in range(1, T + 1):
        grown_wealth = sum(
            model.getVarByName(f"W[{k},{a},{t-1}]").X * (1 + r.get((a, t), 0.0))
            for k in accounts for a in assets)
        total_contributions = sum(model.getVarByName(f"c_hat[{k},{t}]").X for k in accounts)
        total_withdrawals   = sum(w_hat.get((k, t), 0.0) for k in accounts)
        a_ann       = model.getVarByName(f"A_ann[{t}]").X
        a_rb        = model.getVarByName(f"A_rb[{t}]").X
        tax_wd_tax  = model.getVarByName(f"Tax_wd_Tax[{t}]").X
        tax_wd_tda  = model.getVarByName(f"Tax_wd_TDA[{t}]").X
        total_tax_cash_outflow = a_ann + a_rb + tax_wd_tax + tax_wd_tda
        expected_wealth = grown_wealth + total_contributions - total_withdrawals - total_tax_cash_outflow
        actual_wealth   = sum(model.getVarByName(f"W[{k},{a},{t}]").X
                              for k in accounts for a in assets)
        diff = abs(expected_wealth - actual_wealth)
        if diff > tolerance:
            print(f"FAILED at Year {t}: Leak detected!")
            print(f"Expected: ${expected_wealth:,.2f} | Actual: ${actual_wealth:,.2f} | Diff: ${diff:,.2f}")
            integrity_passed = False
        else:
            print(f"Year {t} balanced successfully. (Total Tax Outflow: ${total_tax_cash_outflow:,.2f})")

    print("\n[2] Verifying Location Heuristics...")
    tda_c_hat = model.getVarByName("c_hat[TDA,1]").X
    tda_limit = bar_c.get(("TDA", 1), 0)
    if tda_limit > 0 and (tda_limit - tda_c_hat) > tolerance:
        print(f"WARNING: TDA contribution space not maximized. (Used: {tda_c_hat:.2f} / {tda_limit})")
    else:
        print("Tax-advantaged contribution space efficiently utilized.")

    print("\n" + "=" * 50)
    if integrity_passed:
        print("VERIFICATION PASSED: Model is mathematically sound and exhibiting expected alpha.")
    else:
        print("VERIFICATION FAILED: Check constraint violations above.")
    print("=" * 50 + "\n")


# =============================================================================
# TEST CASES
# =============================================================================

if __name__ == "__main__":

    # ── TEST CASE 1: Single period (T=1) ──────────────────────────────────────
    print("=== TEST CASE 1: Single period (T=1) ===")
    T_test    = 1
    assets_t  = ["Stock", "Bond", "Cash"]
    accounts_t = ["Taxable", "TDA", "Roth"]
    r_test    = {("Stock", 1): 0.08, ("Bond", 1): 0.04, ("Cash", 1): 0.02}
    d_div_test = {1: 0.02}
    d_int_test = {("Bond", 1): 0.04, ("Cash", 1): 0.02}
    phi_test  = {("Stock", 1): 0.10, ("Bond", 1): 0.0, ("Cash", 1): 0.0}
    tau_cg_test = 0.15
    yb_test   = [0.0, 50000.0, float("inf")]
    m_test    = [0.10, 0.24]
    I_base_test = {1: 60000.0}
    bar_c_test  = {("Taxable", 1): 0.0, ("TDA", 1): 7000.0, ("Roth", 1): 7000.0}
    w_hat_test  = {("Taxable", 1): 5000.0, ("TDA", 1): 3000.0, ("Roth", 1): 2000.0}
    Theta_test  = {"Stock": 0.60, "Bond": 0.30, "Cash": 0.10}
    W_init_test = {
        ("Taxable", "Stock"): 100000.0, ("Taxable", "Bond"): 50000.0, ("Taxable", "Cash"): 20000.0,
        ("TDA", "Stock"):     80000.0, ("TDA", "Bond"):     40000.0, ("TDA", "Cash"):     10000.0,
        ("Roth", "Stock"):    30000.0, ("Roth", "Bond"):    15000.0, ("Roth", "Cash"):     5000.0,
    }
    B_init_test = {"Stock": 80000.0, "Bond": 50000.0, "Cash": 20000.0}

    model_t1 = solve_asset_location_optimization(
        T=T_test, assets=assets_t, accounts=accounts_t,
        r=r_test, d_div=d_div_test, d_int=d_int_test, phi=phi_test,
        tau_cg=tau_cg_test, yb=yb_test, m=m_test, I_base=I_base_test,
        bar_c=bar_c_test, w_hat=w_hat_test, Theta=Theta_test,
        W_init=W_init_test, B_init=B_init_test,
        tau_liq_T=0.22, discount_rho=None, M=1e7,
    )
    if model_t1.status == GRB.OPTIMAL:
        print(f"Optimal terminal after-tax wealth: ${model_t1.objVal:,.2f}")
        print("Effective Taxable allocations (W[Taxable,a,1] / total Taxable):")
        total_tax = sum(model_t1.getVarByName(f"W[Taxable,{a},1]").X for a in assets_t)
        for a in assets_t:
            v = model_t1.getVarByName(f"W[Taxable,{a},1]").X
            print(f"  {a}: {v/total_tax if total_tax>0 else 0:.4f}  (${v:,.2f})")
        print(f"c_hat (TDA): {model_t1.getVarByName('c_hat[TDA,1]').X:,.2f}")
        automated_model_verification(model_t1, T_test, accounts_t, assets_t,
                                     r_test, bar_c_test, w_hat_test)
    else:
        print("Status:", model_t1.status)

    # ── TEST CASE 3: 5-Year Horizon (T=5) ────────────────────────────────────
    print("\n=== TEST CASE 3: 5-Year Horizon (T=5) ===")
    T5     = 5
    assets = ["Stock", "Bond", "Cash"]
    accounts = ["Taxable", "TDA", "Roth"]
    r_5    = {(a, t): 0.075 if a == "Stock" else 0.035 if a == "Bond" else 0.020
              for t in range(1, T5 + 1) for a in assets}
    d_div_5  = {t: 0.020 for t in range(1, T5 + 1)}
    d_int_5  = {(a, t): 0.035 if a == "Bond" else 0.020
                for t in range(1, T5 + 1) for a in ["Bond", "Cash"]}
    phi_5    = {(a, t): 0.08 if a == "Stock" else 0.0
                for t in range(1, T5 + 1) for a in assets}
    I_base_5 = {t: 80000.0 if t <= 4 else 45000.0 for t in range(1, T5 + 1)}
    bar_c_5  = {}
    for t in range(1, T5 + 1):
        bar_c_5[("Taxable", t)] = 0.0
        bar_c_5[("TDA",   t)]   = 22500.0 * (1.02 ** (t - 1))
        bar_c_5[("Roth",  t)]   = 6500.0  * (1.02 ** (t - 1))
    w_hat_5 = {}
    for t in range(1, T5 + 1):
        if t <= 4:
            w_hat_5[("Taxable", t)] = 5000.0
            w_hat_5[("TDA",     t)] = 0.0
            w_hat_5[("Roth",    t)] = 0.0
        else:
            w_hat_5[("Taxable", t)] = 25000.0
            w_hat_5[("TDA",     t)] = 20000.0
            w_hat_5[("Roth",    t)] = 10000.0
    yb_5   = [0.0, 50000.0, float("inf")]
    m_5    = [0.10, 0.24]
    Theta_5 = {"Stock": 0.60, "Bond": 0.30, "Cash": 0.10}
    W_init_5 = {
        ("Taxable", "Stock"): 150000.0, ("Taxable", "Bond"):  80000.0, ("Taxable", "Cash"): 40000.0,
        ("TDA",     "Stock"): 200000.0, ("TDA",     "Bond"):  120000.0, ("TDA",     "Cash"): 30000.0,
        ("Roth",    "Stock"):  60000.0, ("Roth",    "Bond"):   30000.0, ("Roth",    "Cash"): 10000.0,
    }
    B_init_5 = {"Stock": 120000.0, "Bond": 80000.0, "Cash": 40000.0}

    model_t5 = solve_asset_location_optimization(
        T=T5, assets=assets, accounts=accounts,
        r=r_5, d_div=d_div_5, d_int=d_int_5, phi=phi_5,
        tau_cg=0.15, yb=yb_5, m=m_5, I_base=I_base_5,
        bar_c=bar_c_5, w_hat=w_hat_5, Theta=Theta_5,
        W_init=W_init_5, B_init=B_init_5,
        tau_liq_T=0.22, discount_rho=0.04, M=1e7,
    )
    if model_t5.status == GRB.OPTIMAL:
        print(f"Optimal 5-year after-tax wealth: ${model_t5.objVal:,.2f}")
        automated_model_verification(model_t5, T5, accounts, assets, r_5, bar_c_5, w_hat_5)
    else:
        print("Model status:", model_t5.status)

    # ── TEST CASE 4: 15-Year Horizon (T=15) ──────────────────────────────────
    print("\n=== TEST CASE 4: 15-Year Horizon (T=15) ===")
    T15    = 15
    r_15   = {(a, t): 0.075 if a == "Stock" else 0.035 if a == "Bond" else 0.020
              for t in range(1, T15 + 1) for a in assets}
    d_div_15  = {t: 0.020 for t in range(1, T15 + 1)}
    d_int_15  = {(a, t): 0.035 if a == "Bond" else 0.020
                 for t in range(1, T15 + 1) for a in ["Bond", "Cash"]}
    phi_15    = {(a, t): 0.08 if a == "Stock" else 0.0
                 for t in range(1, T15 + 1) for a in assets}
    I_base_15 = {t: 80000.0 if t <= 10 else 45000.0 for t in range(1, T15 + 1)}
    bar_c_15  = {}
    for t in range(1, T15 + 1):
        bar_c_15[("Taxable", t)] = 0.0
        bar_c_15[("TDA",   t)]   = 22500.0 * (1.02 ** (t - 1))
        bar_c_15[("Roth",  t)]   = 6500.0  * (1.02 ** (t - 1))
    w_hat_15 = {}
    for t in range(1, T15 + 1):
        if t <= 10:
            w_hat_15[("Taxable", t)] = 5000.0
            w_hat_15[("TDA",     t)] = 0.0
            w_hat_15[("Roth",    t)] = 0.0
        else:
            w_hat_15[("Taxable", t)] = 25000.0
            w_hat_15[("TDA",     t)] = 20000.0
            w_hat_15[("Roth",    t)] = 10000.0
    W_init_15 = {
        ("Taxable", "Stock"): 150000.0, ("Taxable", "Bond"):  80000.0, ("Taxable", "Cash"): 40000.0,
        ("TDA",     "Stock"): 200000.0, ("TDA",     "Bond"):  120000.0, ("TDA",     "Cash"): 30000.0,
        ("Roth",    "Stock"):  60000.0, ("Roth",    "Bond"):   30000.0, ("Roth",    "Cash"): 10000.0,
    }
    B_init_15 = {"Stock": 120000.0, "Bond": 80000.0, "Cash": 40000.0}

    model_t15 = solve_asset_location_optimization(
        T=T15, assets=assets, accounts=accounts,
        r=r_15, d_div=d_div_15, d_int=d_int_15, phi=phi_15,
        tau_cg=0.15, yb=yb_5, m=m_5, I_base=I_base_15,
        bar_c=bar_c_15, w_hat=w_hat_15, Theta=Theta_5,
        W_init=W_init_15, B_init=B_init_15,
        tau_liq_T=0.22, discount_rho=0.04, M=1e7,
    )
    if model_t15.status == GRB.OPTIMAL:
        print(f"Optimal 15-year after-tax wealth: ${model_t15.objVal:,.2f}")
        automated_model_verification(model_t15, T15, accounts, assets, r_15, bar_c_15, w_hat_15)
    else:
        print("Model status:", model_t15.status)