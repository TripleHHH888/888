import numpy as np
from dataclasses import dataclass

try:
    from scipy.optimize import least_squares
except Exception as exc:  # pragma: no cover
    least_squares = None


@dataclass
class Params:
    Qmax_Cd: float = 3.0
    Qmax_DOM: float = 5.0


# Data provided by user
# Columns: Cd_conc, DOM_conc, DOM_adsorption (mg/g)
DOM_rows = [
    [0, 10, -0.092777778],
    [2, 10, -0.264044444],
    [4, 10, -0.204433333],
    [6, 10, -0.221888889],
    [8, 10, -0.141933333],
    [10, 10, -0.027644444],
    [0, 20, 1.818755556],
    [2, 20, 0.0901],
    [4, 20, 0.321533333],
    [6, 20, 0.419444444],
    [8, 20, 0.593333333],
    [10, 20, 0.649577778],
    [0, 30, 3.026688889],
    [2, 30, 2.0552],
    [4, 30, 0.767333333],
    [6, 30, 0.8992],
    [8, 30, 1.136311111],
    [10, 30, 1.155444444],
    [0, 40, 4.190066667],
    [2, 40, 3.772488889],
    [4, 40, 2.572711111],
    [6, 40, 2.886222222],
    [8, 40, 2.649111111],
    [10, 40, 2.141333333],
    [0, 50, 5.150222222],
    [2, 50, 5.069066667],
    [4, 50, 2.970333333],
    [6, 50, 2.623622222],
    [8, 50, 2.864933333],
    [10, 50, 2.965111111],
]

# Columns: Cd_conc, DOM_conc, Cd_adsorption (mg/g)
CD_rows = [
    [0, 10, 0],
    [2, 10, 0.370733333],
    [4, 10, 0.7186],
    [6, 10, 0.9266],
    [8, 10, 1.297966667],
    [10, 10, 1.599666667],
    [0, 20, 0],
    [2, 20, 0.349933333],
    [4, 20, 0.666433333],
    [6, 20, 0.8634],
    [8, 20, 1.154433333],
    [10, 20, 1.406333333],
    [0, 30, 0],
    [2, 30, 0.313466667],
    [4, 30, 0.622133333],
    [6, 30, 0.849966667],
    [8, 30, 1.083366667],
    [10, 30, 1.2983],
    [0, 40, 0],
    [2, 40, 0.2932],
    [4, 40, 0.557766667],
    [6, 40, 0.799666667],
    [8, 40, 1.329433333],
    [10, 40, 1.421666667],
    [0, 50, -0.029733333],
    [2, 50, 0.287633333],
    [4, 50, 0.5732],
    [6, 50, 0.8412],
    [8, 50, 1.0414],
    [10, 50, 1.081433333],
    # control DOM=0
    [0, 0, 0],
    [2, 0, 0.39728],
    [4, 0, 0.78636],
    [6, 0, 1.157966667],
    [8, 0, 1.474773333],
    [10, 0, 1.79202],
]


def predict_q(params_vec, C_cd, C_dom, Qmax_cd=3.0, Qmax_dom=5.0):
    """
    params_vec = [b_cd, b_dom, q0_cd, q0_dom]
    C_cd, C_dom can be arrays
    Returns (q_cd_pred, q_dom_pred)
    """
    b_cd, b_dom, q0_cd, q0_dom = params_vec
    denom = 1.0 + b_cd * C_cd + b_dom * C_dom
    # Extended Langmuir
    q_cd = Qmax_cd * (b_cd * C_cd) / denom
    q_dom = Qmax_dom * (b_dom * C_dom) / denom
    # Add small baseline offsets to accommodate small negative observations
    return q_cd + q0_cd, q_dom + q0_dom


def residuals(params_vec, dom_rows, cd_rows, Qmax_cd=3.0, Qmax_dom=5.0, w_dom=1.0, w_cd=1.0):
    # Build arrays
    Ccd_dom = np.array([r[0] for r in dom_rows], dtype=float)
    Cdom_dom = np.array([r[1] for r in dom_rows], dtype=float)
    qdom_obs = np.array([r[2] for r in dom_rows], dtype=float)

    Ccd_cd = np.array([r[0] for r in cd_rows], dtype=float)
    Cdom_cd = np.array([r[1] for r in cd_rows], dtype=float)
    qcd_obs = np.array([r[2] for r in cd_rows], dtype=float)

    qcd_pred_all, qdom_pred_all = predict_q(params_vec, np.concatenate([Ccd_cd, Ccd_dom]), np.concatenate([Cdom_cd, Cdom_dom]), Qmax_cd, Qmax_dom)
    qcd_pred = qcd_pred_all[: len(Ccd_cd)]
    qdom_pred = qdom_pred_all[len(Ccd_cd) :]

    # Weights to balance scales (optional): inverse of std or fixed
    res_cd = w_cd * (qcd_pred - qcd_obs)
    res_dom = w_dom * (qdom_pred - qdom_obs)
    return np.concatenate([res_cd, res_dom])


def fit_params(params: Params):
    # Initial guesses: b_cd ~ 0.5, b_dom ~ 0.1, small baselines
    x0 = np.array([0.5, 0.1, 0.0, 0.0], dtype=float)
    bounds_lower = [1e-6, 1e-6, -0.5, -0.5]
    bounds_upper = [100.0, 100.0, 0.5, 0.5]

    def fun(x):
        return residuals(x, DOM_rows, CD_rows, Qmax_cd=params.Qmax_Cd, Qmax_dom=params.Qmax_DOM, w_dom=1.0, w_cd=1.0)

    if least_squares is None:
        raise RuntimeError("scipy is required (scipy.optimize.least_squares)")

    result = least_squares(fun, x0=x0, bounds=(bounds_lower, bounds_upper), max_nfev=20000)
    return result


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main():
    params = Params()
    res = fit_params(params)
    b_cd, b_dom, q0_cd, q0_dom = res.x

    # Build predictions for diagnostics
    Ccd_dom = np.array([r[0] for r in DOM_rows], dtype=float)
    Cdom_dom = np.array([r[1] for r in DOM_rows], dtype=float)
    qdom_obs = np.array([r[2] for r in DOM_rows], dtype=float)

    Ccd_cd = np.array([r[0] for r in CD_rows], dtype=float)
    Cdom_cd = np.array([r[1] for r in CD_rows], dtype=float)
    qcd_obs = np.array([r[2] for r in CD_rows], dtype=float)

    qcd_pred_all, qdom_pred_all = predict_q(res.x, np.concatenate([Ccd_cd, Ccd_dom]), np.concatenate([Cdom_cd, Cdom_dom]), params.Qmax_Cd, params.Qmax_DOM)
    qcd_pred = qcd_pred_all[: len(Ccd_cd)]
    qdom_pred = qdom_pred_all[len(Ccd_cd) :]

    r2_cd = r2_score(qcd_obs, qcd_pred)
    r2_dom = r2_score(qdom_obs, qdom_pred)

    print("Fitted parameters (competitive Langmuir, Qmax fixed):")
    print(f"  b_Cd   (L/mg): {b_cd:.4g}")
    print(f"  b_DOM  (L/mg): {b_dom:.4g}")
    print(f"  q0_Cd  (mg/g): {q0_cd:.4g}  [baseline offset]")
    print(f"  q0_DOM (mg/g): {q0_dom:.4g} [baseline offset]")
    print()
    print("Fit diagnostics:")
    print(f"  R2 Cd  : {r2_cd:.4f}")
    print(f"  R2 DOM : {r2_dom:.4f}")
    print(f"  success: {res.success}, nfev: {res.nfev}, message: {res.message}")

    # Show a few predicted vs observed pairs
    print("\nSample predictions (Cd block): Cd, DOM, q_obs, q_pred")
    for i in range(min(10, len(Ccd_cd))):
        print(f"  {Ccd_cd[i]:.1f}, {Cdom_cd[i]:.1f}, {qcd_obs[i]:.4f}, {qcd_pred[i]:.4f}")

    print("\nSample predictions (DOM block): Cd, DOM, q_obs, q_pred")
    for i in range(min(10, len(Ccd_dom))):
        print(f"  {Ccd_dom[i]:.1f}, {Cdom_dom[i]:.1f}, {qdom_obs[i]:.4f}, {qdom_pred[i]:.4f}")


if __name__ == "__main__":
    main()