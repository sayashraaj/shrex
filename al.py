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

    Changes vs. original baseline — all formulation-preserving:

    [1] x_Tax / x_TDA / x_Roth: direct dollar contribution variables replace
        theta*c_hat bilinear products in T5a, T5b, D5, R5. Fully linear.

    [2] D7b / R7b linearised: W[TDA,a,t] and W[Roth,a,t] are direct dollar
        variables; a single sum constraint replaces the bilinear theta*V4 form.
        The optimizer allocates dollars across assets freely.

    [3] theta_Tax[a,t] retained ONLY for the T7c rebalancing anchor.
        It is decoupled from contributions (no bilinear cost there).
        Its [0,1] simplex domain (enforced by E1_Tax) provides the geometric
        anchor that prevents McCormick envelope explosion in the coupled
        bilinear chain: P_rb / S_rb / A_rb / sell_rb_tax / V6 / UG_rb.
        theta_Tax is FREE to differ from Theta[a], preserving asset-location
        optimality. E2 enforces the global target independently.

    [4] TIGHT_M: analytical upper bound replaces raw 1e9/1e8 on all wealth and
        flow variables. Shrinks McCormick envelopes by orders of magnitude.

    [5] Intermediate cost-basis bounds (CB3..CB6): tilde_B[s] <= tilde_W_Tax[s]
        at every intermediate stage. Prevents B > W during relaxation, which
        otherwise generates phantom negative taxes and corrupts bound propagation.

    [6] A_ann / A_rb UB = 2.0 (dimensionless fraction ceiling with margin).

    [7] IIS written on infeasibility for diagnostics.

    All originally bilinear/quadratic constraints remain so — none approximated.
    """

    periods = list(range(1, T + 1))

    # ── Tight analytical upper bound ──────────────────────────────────────────
    total_init = sum(W_init.values())
    max_contrib = max(
        (sum(bar_c.get((k, t), 0.0) for k in accounts) for t in periods),
        default=0.0,
    )
    max_r  = max((v for v in r.values()), default=0.0)
    TIGHT_M = max(3.0 * (total_init + T * max_contrib) * ((1.0 + max_r) ** T), 1e6)

    model = gp.Model("MultiPeriodAssetLocation_MIQP_v3", env=gurobi_venv)

    # ── Solver parameters ─────────────────────────────────────────────────────
    model.Params.NonConvex    = 2
    model.Params.TimeLimit    = 3600
    model.Params.MIPGap       = 1e-6
    model.Params.Presolve     = 2
    model.Params.Cuts         = 3
    model.Params.Heuristics   = 0.75
    model.Params.MIPFocus     = 1
    model.Params.MIQCPMethod  = 1
    model.Params.NumericFocus = 1
    model.Params.ScaleFlag    = 2
    model.Params.NoRelHeurTime = 30
    model.Params.Threads      = 0

    # ── Index sets ────────────────────────────────────────────────────────────
    N_B         = len(m)
    bracket_idx = list(range(1, N_B + 1))
    j_map       = {"Cash": 1, "Bond": 2, "Stock": 3}

    # ── Variables ─────────────────────────────────────────────────────────────

    W = model.addVars(accounts, assets, list(range(0, T + 1)),
                      lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="W")
    B = model.addVars(assets, list(range(0, T + 1)),
                      lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="B")

    # Direct dollar contributions (replace theta*c_hat bilinear terms)
    x_Tax  = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="x_Tax")
    x_TDA  = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="x_TDA")
    x_Roth = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="x_Roth")

    c_hat = model.addVars(accounts, periods, lb=0, ub=TIGHT_M,
                          vtype=GRB.CONTINUOUS, name="c_hat")

    # Taxable rebalancing fractions — [0,1] simplex anchor for T7c only
    theta_Tax = model.addVars(assets, periods, lb=0, ub=1.0,
                              vtype=GRB.CONTINUOUS, name="theta_Tax")

    # Bracket model
    iota     = model.addVars(bracket_idx, periods,
                             lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="iota")
    mu       = model.addVars(bracket_idx, periods, vtype=GRB.BINARY, name="mu")
    tau_marg = model.addVars(periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="tau_marg")

    # Taxable intermediate flows
    G_turn      = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="G_turn")
    n_wd_Tax    = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="n_wd_Tax")
    S_wd        = model.addVars(accounts, assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="S_wd")
    R_wd_Tax    = model.addVars(range(1, 4), periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="R_wd_Tax")
    R_wd_gr_TDA = model.addVars(range(1, 4), periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="R_wd_gr_TDA")
    R_wd_Roth   = model.addVars(range(1, 4), periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="R_wd_Roth")
    S_rb        = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="S_rb")
    P_rb        = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="P_rb")

    Tax_ann     = model.addVars(periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="Tax_ann")
    Tax_rb      = model.addVars(periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="Tax_rb")
    Tax_wd_TDA  = model.addVars(periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="Tax_wd_TDA")
    Tax_wd_Tax  = model.addVars(periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="Tax_wd_Tax")
    W_gross_TDA = model.addVars(periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="W_gross_TDA")
    G_rb        = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="G_rb")
    sell_ann    = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="sell_ann")
    sell_rb_tax = model.addVars(assets, periods, lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="sell_rb_tax")

    # Dimensionless tax-liquidation fractions; UB=2.0 gives margin, keeps envelopes tight
    A_ann = model.addVars(periods, lb=0, ub=2.0, vtype=GRB.CONTINUOUS, name="A_ann")
    A_rb  = model.addVars(periods, lb=0, ub=2.0, vtype=GRB.CONTINUOUS, name="A_rb")

    delta_Tax  = model.addVars(assets, periods, vtype=GRB.BINARY, name="delta_Tax")
    delta_TDA  = model.addVars(assets, periods, vtype=GRB.BINARY, name="delta_TDA")
    delta_Roth = model.addVars(assets, periods, vtype=GRB.BINARY, name="delta_Roth")
    delta_rb   = model.addVars(assets, periods, vtype=GRB.BINARY, name="delta_rb")

    tilde_W_Tax = model.addVars(range(1, 7), assets, periods,
                                lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="tilde_W_Tax")
    tilde_B     = model.addVars(range(1, 7), assets, periods,
                                lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="tilde_B")

    tilde_W_TDA  = model.addVars([2, 3, 4], assets, periods,
                                 lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="tilde_W_TDA")
    tilde_W_Roth = model.addVars([2, 3, 4], assets, periods,
                                 lb=0, ub=TIGHT_M, vtype=GRB.CONTINUOUS, name="tilde_W_Roth")

    # ── Initial conditions ────────────────────────────────────────────────────
    for k in accounts:
        for a in assets:
            model.addConstr(W[k, a, 0] == W_init[(k, a)], name=f"IC_W_{k}_{a}_0")
    for a in assets:
        model.addConstr(B[a, 0] == B_init[a], name=f"IC_B_{a}_0")

    # ── Contribution bounds and sum constraints ───────────────────────────────
    for t in periods:
        for k in accounts:
            model.addConstr(c_hat[k, t] <= bar_c.get((k, t), 0.0), name=f"c_hat_ub_{k}_{t}")
        model.addConstr(gp.quicksum(x_Tax[a, t]  for a in assets) == c_hat["Taxable", t],
                        name=f"xTax_sum_{t}")
        model.addConstr(gp.quicksum(x_TDA[a, t]  for a in assets) == c_hat["TDA", t],
                        name=f"xTDA_sum_{t}")
        model.addConstr(gp.quicksum(x_Roth[a, t] for a in assets) == c_hat["Roth", t],
                        name=f"xRoth_sum_{t}")
        # Simplex anchor for taxable rebalancing (theta_Tax sums to 1)
        model.addConstr(gp.quicksum(theta_Tax[a, t] for a in assets) == 1.0,
                        name=f"E1_Tax_{t}")

    # ── Per-period mechanics ──────────────────────────────────────────────────
    for t in periods:

        # ── Step 1: Turnover ──────────────────────────────────────────────────
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

        model.addQConstr(
            A_ann[t] * (V2 - tau_cg * UG_ann) == Tax_ann[t] * V2,
            name=f"T34f_{t}")
        for a in assets:
            model.addQConstr(
                sell_ann[a, t] * V2 == A_ann[t] * tilde_W_Tax[2, a, t],
                name=f"T34g_{a}_{t}")
            model.addConstr(
                tilde_W_Tax[3, a, t] == tilde_W_Tax[2, a, t] - sell_ann[a, t],
                name=f"T34h_{a}_{t}")
            model.addQConstr(
                tilde_B[3, a, t] * tilde_W_Tax[2, a, t]
                == tilde_B[2, a, t] * (tilde_W_Tax[2, a, t] - sell_ann[a, t]),
                name=f"T34i_{a}_{t}")
            model.addConstr(tilde_W_Tax[3, a, t] >= 0,                    name=f"T34j_{a}_{t}")
            model.addConstr(tilde_B[3, a, t] <= tilde_W_Tax[3, a, t],     name=f"CB3_{a}_{t}")

        # ── Step 5: Contributions (LINEAR — theta eliminated) ─────────────────
        for a in assets:
            model.addConstr(
                tilde_W_Tax[4, a, t] == tilde_W_Tax[3, a, t] + x_Tax[a, t],
                name=f"T5a_{a}_{t}")
            model.addConstr(
                tilde_B[4, a, t] == tilde_B[3, a, t] + x_Tax[a, t],
                name=f"T5b_{a}_{t}")
            model.addConstr(
                tilde_W_TDA[3, a, t] == tilde_W_TDA[2, a, t] + x_TDA[a, t],
                name=f"D5_{a}_{t}")
            model.addConstr(
                tilde_W_Roth[3, a, t] == tilde_W_Roth[2, a, t] + x_Roth[a, t],
                name=f"R5_{a}_{t}")
            model.addConstr(tilde_B[4, a, t] <= tilde_W_Tax[4, a, t],     name=f"CB4_{a}_{t}")

        # ── Step 6a: Sequential withdrawal — Taxable ──────────────────────────
        for a in assets:
            W4   = tilde_W_Tax[4, a, t]
            B4   = tilde_B[4, a, t]
            NS_a = W4 - tau_cg * (W4 - B4)
            j    = j_map[a]
            model.addConstr(n_wd_Tax[a, t] <= R_wd_Tax[j, t],
                            name=f"T6f_{a}_{t}")
            model.addConstr(n_wd_Tax[a, t] <= NS_a,
                            name=f"T6g_{a}_{t}")
            model.addConstr(n_wd_Tax[a, t] >= R_wd_Tax[j, t] - M * delta_Tax[a, t],
                            name=f"T6h_{a}_{t}")
            model.addConstr(n_wd_Tax[a, t] >= NS_a - M * (1 - delta_Tax[a, t]),
                            name=f"T6i_{a}_{t}")
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
            model.addQConstr(
                tilde_B[5, a, t] * tilde_W_Tax[4, a, t]
                == tilde_B[4, a, t] * (tilde_W_Tax[4, a, t] - S_wd["Taxable", a, t]),
                name=f"T6l_{a}_{t}")
            model.addConstr(tilde_W_Tax[5, a, t] >= 0,                    name=f"T6n_{a}_{t}")
            model.addConstr(tilde_B[5, a, t] <= tilde_W_Tax[5, a, t],     name=f"CB5_{a}_{t}")

        model.addConstr(
            Tax_wd_Tax[t] == gp.quicksum(S_wd["Taxable", a, t] - n_wd_Tax[a, t] for a in assets),
            name=f"T6m_{t}")

        # ── Step 6b: TDA withdrawal ───────────────────────────────────────────
        model.addQConstr(
            W_gross_TDA[t] * (1 - tau_marg[t]) == w_hat.get(("TDA", t), 0.0),
            name=f"D6a_{t}")
        model.addConstr(
            Tax_wd_TDA[t] == W_gross_TDA[t] * tau_marg[t],
            name=f"D6b_{t}")
        model.addConstr(R_wd_gr_TDA[1, t] == W_gross_TDA[t],                       name=f"D6c_{t}")
        model.addConstr(R_wd_gr_TDA[2, t] == R_wd_gr_TDA[1, t] - S_wd["TDA", "Cash", t],
                        name=f"D6d_{t}")
        model.addConstr(R_wd_gr_TDA[3, t] == R_wd_gr_TDA[2, t] - S_wd["TDA", "Bond", t],
                        name=f"D6e_{t}")

        for a in assets:
            j      = j_map[a]
            W3_TDA = tilde_W_TDA[3, a, t]
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
            j       = j_map[a]
            W3_Roth = tilde_W_Roth[3, a, t]
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
        V5_Tax = gp.quicksum(tilde_W_Tax[5, a, t] for a in assets)

        for a in assets:
            # T7c: theta_Tax * V5_Tax anchors the rebalancing bilinear chain.
            # theta_Tax is NOT constrained to equal Theta[a] — the optimizer
            # chooses it freely. E2 separately enforces the global allocation.
            model.addQConstr(
                P_rb[a, t] - S_rb[a, t] == theta_Tax[a, t] * V5_Tax - tilde_W_Tax[5, a, t],
                name=f"T7c_{a}_{t}")
            model.addConstr(P_rb[a, t] <= TIGHT_M * (1 - delta_rb[a, t]), name=f"T7e_P_{a}_{t}")
            model.addConstr(S_rb[a, t] <= TIGHT_M * delta_rb[a, t],        name=f"T7e_S_{a}_{t}")
            model.addConstr(
                tilde_W_Tax[6, a, t] == tilde_W_Tax[5, a, t] - S_rb[a, t] + P_rb[a, t],
                name=f"T7h_{a}_{t}")
            model.addQConstr(
                tilde_B[6, a, t] * tilde_W_Tax[5, a, t]
                == tilde_B[5, a, t] * (tilde_W_Tax[5, a, t] - S_rb[a, t])
                   + P_rb[a, t] * tilde_W_Tax[5, a, t],
                name=f"T7i_{a}_{t}")
            model.addConstr(tilde_B[6, a, t] <= tilde_W_Tax[6, a, t],     name=f"CB6_{a}_{t}")

        model.addConstr(
            gp.quicksum(P_rb[a, t] for a in assets) == gp.quicksum(S_rb[a, t] for a in assets),
            name=f"T7f_{t}")

        for a in assets:
            model.addQConstr(
                G_rb[a, t] * tilde_W_Tax[5, a, t]
                == S_rb[a, t] * (tilde_W_Tax[5, a, t] - tilde_B[5, a, t]),
                name=f"G_rb_def_{a}_{t}")

        model.addConstr(
            Tax_rb[t] == tau_cg * gp.quicksum(G_rb[a, t] for a in assets),
            name=f"T7g_{t}")

        V6      = gp.quicksum(tilde_W_Tax[6, a, t] for a in assets)
        UG_rb_t = gp.quicksum(tilde_W_Tax[6, a, t] - tilde_B[6, a, t] for a in assets)

        model.addQConstr(
            A_rb[t] * (V6 - tau_cg * UG_rb_t) == Tax_rb[t] * V6,
            name=f"T7o_{t}")
        for a in assets:
            model.addQConstr(
                sell_rb_tax[a, t] * V6 == A_rb[t] * tilde_W_Tax[6, a, t],
                name=f"T7p_{a}_{t}")
            model.addConstr(
                W["Taxable", a, t] == tilde_W_Tax[6, a, t] - sell_rb_tax[a, t],
                name=f"T7q_{a}_{t}")
            model.addQConstr(
                B[a, t] * tilde_W_Tax[6, a, t]
                == tilde_B[6, a, t] * (tilde_W_Tax[6, a, t] - sell_rb_tax[a, t]),
                name=f"T7r_{a}_{t}")

        # ── TDA rebalancing (LINEAR — theta eliminated) ───────────────────────
        V4_TDA = gp.quicksum(tilde_W_TDA[4, a, t] for a in assets)
        model.addConstr(
            gp.quicksum(W["TDA", a, t] for a in assets) == V4_TDA,
            name=f"D7b_sum_{t}")

        # ── Roth rebalancing (LINEAR — theta eliminated) ──────────────────────
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
                model.addConstr(iota[b, t] <= yb[b] - yb[b - 1], name=f"BR1_ub_{b}_{t}")
        for b in range(1, N_B):
            model.addConstr(iota[b, t] >= (yb[b] - yb[b - 1]) - M * (1 - mu[b, t]),
                            name=f"BR2a_{b}_{t}")
            model.addConstr(iota[b + 1, t] <= M * mu[b, t], name=f"BR2b_{b}_{t}")

        model.addConstr(mu[N_B, t] == 0, name=f"mu_last_{t}")
        model.addConstr(gp.quicksum(iota[b, t] for b in bracket_idx) == I_ord_t,
                        name=f"BR3_{t}")

        marg_expr = gp.LinExpr()
        for b_idx, b in enumerate(bracket_idx):
            mu_prev = 1.0 if b == 1 else mu[b - 1, t]
            marg_expr += m[b_idx] * (mu_prev - mu[b, t])
        model.addConstr(tau_marg[t] == marg_expr, name=f"BR5_{t}")

        # ── Asset-location: global allocation constraint (E2) ─────────────────
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
    term_Tax  = gp.quicksum(
        W["Taxable", a, T] - tau_cg * (W["Taxable", a, T] - B[a, T]) for a in assets)
    term_TDA  = gp.quicksum(W["TDA",  a, T] * (1 - tau_liq_T) for a in assets)
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

    if model.status == GRB.INFEASIBLE:
        print("\n[DIAGNOSTIC] Model INFEASIBLE — computing IIS...")
        model.computeIIS()
        model.write("debug.ilp")
        print("[DIAGNOSTIC] IIS written to debug.ilp")

    return model


# =============================================================================
# VERIFICATION SUITE
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
        a_ann      = model.getVarByName(f"A_ann[{t}]").X
        a_rb       = model.getVarByName(f"A_rb[{t}]").X
        tax_wd_tax = model.getVarByName(f"Tax_wd_Tax[{t}]").X
        tax_wd_tda = model.getVarByName(f"Tax_wd_TDA[{t}]").X
        total_tax_outflow = a_ann + a_rb + tax_wd_tax + tax_wd_tda
        expected = grown_wealth + total_contributions - total_withdrawals - total_tax_outflow
        actual   = sum(model.getVarByName(f"W[{k},{a},{t}]").X
                       for k in accounts for a in assets)
        diff = abs(expected - actual)
        if diff > tolerance:
            print(f"FAILED at Year {t}: Leak! Expected ${expected:,.2f} | Actual ${actual:,.2f} | Diff ${diff:,.2f}")
            integrity_passed = False
        else:
            print(f"Year {t} balanced. (Tax outflow: ${total_tax_outflow:,.2f})")

    print("\n[2] Verifying location heuristics...")
    tda_c_hat = model.getVarByName("c_hat[TDA,1]").X
    tda_limit = bar_c.get(("TDA", 1), 0)
    if tda_limit > 0 and (tda_limit - tda_c_hat) > tolerance:
        print(f"WARNING: TDA contribution space not maximized. ({tda_c_hat:.2f} / {tda_limit})")
    else:
        print("Tax-advantaged contribution space efficiently utilized.")

    print("\n" + "=" * 50)
    print("VERIFICATION PASSED." if integrity_passed else "VERIFICATION FAILED.")
    print("=" * 50 + "\n")


# =============================================================================
# TEST CASES
# =============================================================================

if __name__ == "__main__":

    assets   = ["Stock", "Bond", "Cash"]
    accounts = ["Taxable", "TDA", "Roth"]

    # ── T=1 ───────────────────────────────────────────────────────────────────
    print("=== TEST CASE 1: T=1 ===")
    W_init_t1 = {
        ("Taxable", "Stock"): 100000.0, ("Taxable", "Bond"): 50000.0, ("Taxable", "Cash"): 20000.0,
        ("TDA",     "Stock"):  80000.0, ("TDA",     "Bond"): 40000.0, ("TDA",     "Cash"): 10000.0,
        ("Roth",    "Stock"):  30000.0, ("Roth",    "Bond"): 15000.0, ("Roth",    "Cash"):  5000.0,
    }
    B_init_t1  = {"Stock": 80000.0, "Bond": 50000.0, "Cash": 20000.0}
    r_t1       = {("Stock", 1): 0.08, ("Bond", 1): 0.04, ("Cash", 1): 0.02}
    bar_c_t1   = {("Taxable", 1): 0.0, ("TDA", 1): 7000.0, ("Roth", 1): 7000.0}
    w_hat_t1   = {("Taxable", 1): 5000.0, ("TDA", 1): 3000.0, ("Roth", 1): 2000.0}

    m1 = solve_asset_location_optimization(
        T=1, assets=assets, accounts=accounts,
        r=r_t1,
        d_div={1: 0.02},
        d_int={("Bond", 1): 0.04, ("Cash", 1): 0.02},
        phi={("Stock", 1): 0.10, ("Bond", 1): 0.0, ("Cash", 1): 0.0},
        tau_cg=0.15, yb=[0.0, 50000.0, float("inf")], m=[0.10, 0.24],
        I_base={1: 60000.0}, bar_c=bar_c_t1, w_hat=w_hat_t1,
        Theta={"Stock": 0.60, "Bond": 0.30, "Cash": 0.10},
        W_init=W_init_t1, B_init=B_init_t1,
        tau_liq_T=0.22, discount_rho=None, M=1e7,
    )
    if m1.status == GRB.OPTIMAL:
        print(f"Optimal terminal after-tax wealth: ${m1.objVal:,.2f}")
        total_tax = sum(m1.getVarByName(f"W[Taxable,{a},1]").X for a in assets)
        for a in assets:
            v = m1.getVarByName(f"W[Taxable,{a},1]").X
            print(f"  Taxable {a}: {v/total_tax if total_tax > 0 else 0:.4f}  (${v:,.2f})")
        print(f"  c_hat[TDA,1]: {m1.getVarByName('c_hat[TDA,1]').X:,.2f}")
        automated_model_verification(m1, 1, accounts, assets, r_t1, bar_c_t1, w_hat_t1)
    else:
        print("Status:", m1.status)

    # ── T=5 ───────────────────────────────────────────────────────────────────
    print("\n=== TEST CASE 3: T=5 ===")
    T5      = 5
    r_5     = {(a, t): 0.075 if a == "Stock" else 0.035 if a == "Bond" else 0.020
               for t in range(1, T5 + 1) for a in assets}
    d_div_5 = {t: 0.020 for t in range(1, T5 + 1)}
    d_int_5 = {(a, t): 0.035 if a == "Bond" else 0.020
               for t in range(1, T5 + 1) for a in ["Bond", "Cash"]}
    phi_5   = {(a, t): 0.08 if a == "Stock" else 0.0
               for t in range(1, T5 + 1) for a in assets}
    bar_c_5 = {(k, t): (0.0 if k == "Taxable" else
                         22500.0 * 1.02 ** (t - 1) if k == "TDA" else
                         6500.0  * 1.02 ** (t - 1))
               for k in accounts for t in range(1, T5 + 1)}
    w_hat_5 = {(k, t): (5000.0 if k == "Taxable" else 0.0) if t <= 4 else
                        (25000.0 if k == "Taxable" else 20000.0 if k == "TDA" else 10000.0)
               for k in accounts for t in range(1, T5 + 1)}
    W_init_5 = {
        ("Taxable", "Stock"): 150000.0, ("Taxable", "Bond"):  80000.0, ("Taxable", "Cash"): 40000.0,
        ("TDA",     "Stock"): 200000.0, ("TDA",     "Bond"): 120000.0, ("TDA",     "Cash"): 30000.0,
        ("Roth",    "Stock"):  60000.0, ("Roth",    "Bond"):  30000.0, ("Roth",    "Cash"): 10000.0,
    }
    B_init_5 = {"Stock": 120000.0, "Bond": 80000.0, "Cash": 40000.0}
    Theta_5  = {"Stock": 0.60, "Bond": 0.30, "Cash": 0.10}

    m5 = solve_asset_location_optimization(
        T=T5, assets=assets, accounts=accounts,
        r=r_5, d_div=d_div_5, d_int=d_int_5, phi=phi_5,
        tau_cg=0.15, yb=[0.0, 50000.0, float("inf")], m=[0.10, 0.24],
        I_base={t: 80000.0 if t <= 4 else 45000.0 for t in range(1, T5 + 1)},
        bar_c=bar_c_5, w_hat=w_hat_5, Theta=Theta_5,
        W_init=W_init_5, B_init=B_init_5,
        tau_liq_T=0.22, discount_rho=0.04, M=1e7,
    )
    if m5.status == GRB.OPTIMAL:
        print(f"Optimal 5-year after-tax wealth: ${m5.objVal:,.2f}")
        automated_model_verification(m5, T5, accounts, assets, r_5, bar_c_5, w_hat_5)
    else:
        print("Model status:", m5.status)

    # ── T=15 ──────────────────────────────────────────────────────────────────
    print("\n=== TEST CASE 4: T=15 ===")
    T15      = 15
    r_15     = {(a, t): 0.075 if a == "Stock" else 0.035 if a == "Bond" else 0.020
                for t in range(1, T15 + 1) for a in assets}
    d_div_15 = {t: 0.020 for t in range(1, T15 + 1)}
    d_int_15 = {(a, t): 0.035 if a == "Bond" else 0.020
                for t in range(1, T15 + 1) for a in ["Bond", "Cash"]}
    phi_15   = {(a, t): 0.08 if a == "Stock" else 0.0
                for t in range(1, T15 + 1) for a in assets}
    bar_c_15 = {(k, t): (0.0 if k == "Taxable" else
                          22500.0 * 1.02 ** (t - 1) if k == "TDA" else
                          6500.0  * 1.02 ** (t - 1))
                for k in accounts for t in range(1, T15 + 1)}
    w_hat_15 = {(k, t): (5000.0 if k == "Taxable" else 0.0) if t <= 10 else
                         (25000.0 if k == "Taxable" else 20000.0 if k == "TDA" else 10000.0)
                for k in accounts for t in range(1, T15 + 1)}
    W_init_15 = {
        ("Taxable", "Stock"): 150000.0, ("Taxable", "Bond"):  80000.0, ("Taxable", "Cash"): 40000.0,
        ("TDA",     "Stock"): 200000.0, ("TDA",     "Bond"): 120000.0, ("TDA",     "Cash"): 30000.0,
        ("Roth",    "Stock"):  60000.0, ("Roth",    "Bond"):  30000.0, ("Roth",    "Cash"): 10000.0,
    }
    B_init_15 = {"Stock": 120000.0, "Bond": 80000.0, "Cash": 40000.0}

    m15 = solve_asset_location_optimization(
        T=T15, assets=assets, accounts=accounts,
        r=r_15, d_div=d_div_15, d_int=d_int_15, phi=phi_15,
        tau_cg=0.15, yb=[0.0, 50000.0, float("inf")], m=[0.10, 0.24],
        I_base={t: 80000.0 if t <= 10 else 45000.0 for t in range(1, T15 + 1)},
        bar_c=bar_c_15, w_hat=w_hat_15, Theta=Theta_5,
        W_init=W_init_15, B_init=B_init_15,
        tau_liq_T=0.22, discount_rho=0.04, M=1e7,
    )
    if m15.status == GRB.OPTIMAL:
        print(f"Optimal 15-year after-tax wealth: ${m15.objVal:,.2f}")
        automated_model_verification(m15, T15, accounts, assets, r_15, bar_c_15, w_hat_15)
    else:
        print("Model status:", m15.status)
