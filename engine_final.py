"""
engine_final.py
================
Consolidated engine for the v196 rebuild, combining (per explicit user
instruction to use "all of the work done yesterday"):
  (a) the normalized (unit-invariant) inner CES nest, pinned at the
      pre-AI optimal allocation, for ALL sectors (methodological
      consistency across Law/IB/BigFour) -- NormalizedCES_Test_v2.md.
  (b) Law's data-derived inner-nest shares (alpha_A/E/R) from
      LawCESShares_v1.md, with the outer share alpha_P reported at three
      points spanning its identified-feasible band (0.05/0.10/0.14),
      since alpha_P is NOT point-identified by the available data.
  (c) IB's corrected static calibration (Director+MD senior population)
      from IB_StaticCalibration_v1.md.
  (d) BigFour unchanged (out of scope for this recalibration pass; only
      the normalization is applied, for internal consistency).

IB DYNAMIC PARAMETERS: an earlier draft flagged n_bar, delta_A and delta_P for IB as still
encoding the old population boundary (VP+Director+MD). That was resolved before the published runs:
the IB block below uses the Director/MD boundary (n_bar = 10, delta_P = 6.5%), as in the paper
(Appendix D). The flag IB_DYNAMICS_VALID is True and is not read by any script.
"""
from scipy.optimize import minimize, brentq

BETA = 1.0 / 1.05
P0 = 100.0
SHOCK = 5
RHO_1 = -0.6
REGIME = {'pre': ((0.5 - 1.0) / 0.5, 2.00), 'post': ((1.5 - 1.0) / 1.5, 0.50)}

IB_DYNAMICS_VALID = True  # resolved; see module docstring (not read by any script)
FAST = False  # module-level toggle: True uses a cheaper (3-start) inner
              # optimizer for building surrogate grids during entry-
              # matching search; final reported surrogates/paths are always
              # rebuilt with FAST=False (the full multi-start) for the
              # numbers actually reported.

# ---- Law: data-derived inner shares, three-point alpha_P sensitivity ----
# alpha_K, eta from LawCESShares_v1.md's verified T1/T2 table (exact points).
# Cohort/wage parameters: cal20.py's "NEW" 2026 Law block (matches the
# CALIB['Law'] cohort side already in engine_v19.py -- unchanged by this
# exercise, since normalization leaves pre-shock statics/cohort untouched).
_LAW_COMMON = dict(alpha_A=0.1845, alpha_E=0.2652,  # alpha_R = 1 - these = 0.5503
                    Gamma=3000.0, w_P=558.0, w_A=372.0, w_E=80.0, r_K=50.0,
                    delta_A=0.17, delta_P=0.038, leverage=1.8, n_bar=8)
_LAW_PPP_TARGET = 1890.0
_LAW_POINTS = {  # alpha_P: (alpha_K, eta)   -- from LawCESShares_v1.md table
    # 0.09 = BASELINE (29 Sep 2026): a normalization inside the partially identified band
    # (0, 0.15), chosen because it aligns the salaried-senior margin (MRP_P ~ $558k
    # non-equity-partner pay). NOT an identified estimate. alpha_K from the data-derived
    # alpha_K:alpha_N ratio (0.4056); eta solved jointly with A_demand for T1/T2 (raw nest).
    0.09: (0.3691, 2.626),
    0.05: (0.385, 2.21),
    0.10: (0.365, 2.76),
    0.14: (0.349, 3.53),
}

# ---- IB: corrected static calibration (IB_StaticCalibration_v1.md /
# panel_calib_v2.py), with n_bar and delta_P now reconciled to the SAME
# panel run that produced the static shares -- panel_calib_v2.py computes
# an internal entry-to-(Director+MD) tenure of 9.5yr (2.5 analyst + 3.5
# associate + 3.5 VP, M&I stage lengths) and a pooled senior attrition of
# 6.5%/yr, neither of which had been carried into CALIB before now (that
# was the "do not rerun IB dynamics" gap). n_bar rounded to 10 (nearest
# integer; the model's cohort accounting is annual).
_IB = dict(alpha_P=0.1803, alpha_K=0.2751, alpha_A=0.1572, alpha_E=0.0098,
           Gamma=2500.0, w_P=3405.0, w_A=344.513, w_E=86.0, r_K=50.0,
           delta_A=0.22, eta=5.3679, delta_P=0.065, leverage=3.9562, n_bar=10)
# w_P (29 Sep 2026, user decision): annual economic cost of senior (Director+MD)
# capacity, calibrated as the residual compensation attributable to that tier in the
# same panel fit (panel_calib_v2.py pooled implied senior pay, $3,405k), NOT the
# $365k H-1B base salary previously used. $365k priced seniors at ~1/10 of their
# marginal product, which made the patient firm's problem unbounded
# (IBDynamicRebuild_v2.md). Firm-year range of the same figure: $1,292k-$7,287k.
_IB_PPP_TARGET = 4883.45

CALIB = {}
for ap, (ak, eta) in _LAW_POINTS.items():
    CALIB[f'Law_{int(round(ap*100)):02d}'] = dict(_LAW_COMMON, alpha_P=ap, alpha_K=ak, eta=eta)
CALIB['IB'] = dict(_IB)
# ---- Audit: engine_v19.CALIB['BigFour'] unchanged (boundary case, not recalibrated).
# Sources: ParameterLedger_v2.md section 3. eta held at 3.6965; A_demand fitted to T1 only.
CALIB['BigFour'] = dict(alpha_P=0.40, alpha_K=0.15, alpha_A=0.55, alpha_E=0.25,
                        Gamma=600.0, w_P=290.0, w_A=98.0, w_E=49.0, r_K=50.0,
                        delta_A=0.21, eta=3.6965, delta_P=0.078, leverage=7.5, n_bar=12)
PPP_TARGET = {**{k: _LAW_PPP_TARGET for k in CALIB if k.startswith('Law')},
              'IB': _IB_PPP_TARGET, 'BigFour': 1147.0}
SECTORS = list(CALIB)


class Sector:
    """Normalized-nest primitives. A_demand is fitted post-construction via
    fit_static() to satisfy T1 (and, for Law/IB, T2), exactly as
    calibrate_v19.py does -- this is unaffected by normalization, since
    normalization is exact (not approximate) at the pre-shock point."""

    def __init__(self, name, regime_post=None, normalize=True, **override):
        self.name = name
        for k, v in {**CALIB[name], **override}.items():
            setattr(self, k, v)
        self.beta = BETA
        self.regimes = {'pre': REGIME['pre'], 'post': regime_post or REGIME['post']}
        self.A_demand = getattr(self, 'A_demand', 1.0)
        self._normed = False
        if normalize:
            self.normalize()

    def normalize(self):
        """(Re-)pin the normalized nest at the CURRENT A_demand/eta. Must be
        called again after A_demand changes (e.g. after fit_static's
        brentq), since the pre-shock optimal (E,R) -- and hence the
        normalization anchor -- depends on A_demand through the revenue
        term. Calling this before A_demand is finalized silently pins the
        anchor at the wrong point; verified by grid search to produce
        non-global-optimum, near-zero E0/R0 artifacts when skipped."""
        _, E0, R0 = self._solve_raw(P0, self.Ahat_star, 'pre')
        aR = 1 - self.alpha_A - self.alpha_E
        rho = REGIME['pre'][0]
        a = dict(A=self.alpha_A, E=self.alpha_E, R=aR)
        self.x0 = dict(A=self.Ahat_star, E=E0, R=R0)
        tot = sum(a[k] * self.x0[k] ** rho for k in a)
        self.pi = {k: a[k] * self.x0[k] ** rho / tot for k in a}
        self.X0 = tot ** (1 / rho)
        self._normed = True

    @property
    def surv(self):
        return (1.0 - self.delta_A) ** self.n_bar

    @property
    def e_ss(self):
        return self.leverage * P0 * self.delta_A / (1.0 - self.surv)

    @property
    def Ahat_star(self):
        return self.leverage * P0

    @property
    def p_promote(self):
        return self.delta_P * P0 / (self.surv * self.e_ss)

    def inner(self, A, E, R, rho_2):
        x = dict(A=A, E=E, R=R)
        if abs(rho_2) < 1e-10:
            out = self.X0
            for k in x:
                out *= (x[k] / self.x0[k]) ** self.pi[k]
            return out
        return self.X0 * sum(self.pi[k] * (x[k] / self.x0[k]) ** rho_2 for k in x) ** (1.0 / rho_2)

    def _inner_raw(self, A, E, R, rho_2):
        aR = 1.0 - self.alpha_A - self.alpha_E
        if abs(rho_2) < 1e-10:
            return A ** self.alpha_A * E ** self.alpha_E * R ** aR
        return (self.alpha_A * A ** rho_2 + self.alpha_E * E ** rho_2 + aR * R ** rho_2) ** (1.0 / rho_2)

    def output(self, P, K, A, E, R, rho_2, normed=True):
        eps = 1e-10
        P, K, A, E, R = (max(v, eps) for v in (P, K, A, E, R))
        X = self.inner(A, E, R, rho_2) if normed else self._inner_raw(A, E, R, rho_2)
        aN = 1.0 - self.alpha_P - self.alpha_K
        Y = (self.alpha_P * P ** RHO_1 + self.alpha_K * K ** RHO_1 + aN * X ** RHO_1) ** (1.0 / RHO_1)
        return self.Gamma * Y

    def price(self, Y):
        return self.A_demand * max(Y, 1e-10) ** (-1.0 / self.eta)

    def _solve_core(self, P, Ahat, regime, normed):
        rho_2, w_R = self.regimes[regime]

        def neg(x):
            K, E, R = x
            if K <= 0 or E <= 0 or R <= 0:
                return 1e15
            Y = self.output(P, K, Ahat, E, R, rho_2, normed=normed)
            return -(self.price(Y) * Y - self.r_K * K - self.w_E * E - w_R * R)

        # --- analytic FOC ratio for E:R as a starting direction, then a
        # 1-D scale search along that direction, then a 3D polish. This
        # replaces a fixed same-order-of-magnitude multi-start, which was
        # verified (by exhaustive grid check) to land on badly non-global
        # local optima once the data-derived shares make alpha_R >> alpha_A.
        if normed and getattr(self, 'x0', None):
            pi, x0 = self.pi, self.x0
        else:
            aR = 1.0 - self.alpha_A - self.alpha_E
            pi, x0 = dict(A=self.alpha_A, E=self.alpha_E, R=aR), dict(A=1.0, E=1.0, R=1.0)

        def eff(i, rho):
            return pi[i] * x0[i] ** (-rho) if abs(rho) > 1e-12 else pi[i] / max(x0[i], 1e-12)

        eE, eR = eff('E', rho_2), eff('R', rho_2)
        try:
            ER = ((self.w_E / max(w_R, 1e-9)) * (eR / eE)) ** (1.0 / (rho_2 - 1.0)) \
                if abs(rho_2 - 1.0) > 1e-8 else 1.0
        except (ValueError, ZeroDivisionError, OverflowError):
            ER = 1.0
        e_unit = ER if (ER > 0 and ER < 1e12) else 1.0
        r_unit = 1.0
        try:
            KE = ((self.r_K / self.w_E) * (self.alpha_E / self.alpha_K)) ** (1.0 / (RHO_1 - 1.0))
        except (ValueError, ZeroDivisionError, OverflowError):
            KE = 1.0
        k_unit = KE * e_unit if (KE > 0 and KE < 1e12) else e_unit

        import math

        def neg_scale(logs):
            s = 10 ** logs[0]
            return neg([s * k_unit, s * e_unit, s * r_unit])

        e0 = max(10.0, Ahat * 0.8)
        scale0 = e0 / max(e_unit, 1e-9)
        ls_center = max(-10, min(10, math.log10(max(scale0, 1e-12))))
        ls_grid = (0, -4, 4) if FAST else (0, -3, -1, 1, 3, -6, 6, -9, 9)
        best_s = None
        for ls0 in ls_grid:
            r = minimize(neg_scale, [ls_center + ls0], method='Nelder-Mead',
                         options={'xatol': 1e-3, 'fatol': 1e-8, 'maxiter': 200})
            if best_s is None or r.fun < best_s.fun:
                best_s = r
        s_star = 10 ** best_s.x[0]
        g0 = [s_star * k_unit, s_star * e_unit, s_star * r_unit]

        if FAST:
            starts = [g0, [g0[0] * 0.1, g0[1] * 0.1, g0[2] * 0.1], [g0[0] * 10, g0[1] * 10, g0[2] * 10]]
        else:
            starts = [g0, [g0[0] * 0.1, g0[1] * 0.1, g0[2] * 0.1], [g0[0] * 10, g0[1] * 10, g0[2] * 10],
                      [e0, e0, e0], [0.3 * e0, e0, 3 * e0], [3 * e0, 0.3 * e0, e0],
                      [e0, 3 * e0, 0.3 * e0], [e0, e0, 30 * e0], [e0, 0.5 * e0, 100 * e0]]
        best = None
        for g in starts:
            g = [max(1e-6, v) for v in g]
            r = minimize(neg, g, bounds=[(1e-6, None)] * 3, method='L-BFGS-B',
                         options={'ftol': 1e-14, 'gtol': 1e-9, 'maxiter': 3000})
            if best is None or r.fun < best.fun:
                best = r
        return -best.fun, best.x[1], best.x[2]

    def _solve_raw(self, P, Ahat, regime):
        return self._solve_core(P, Ahat, regime, normed=False)

    def raw_operating_surplus(self, P, Ahat, regime):
        return self._solve_core(P, Ahat, regime, normed=False)[0]

    def operating_surplus(self, P, Ahat, regime):
        if not self._normed:
            raise RuntimeError(f"{self.name}: operating_surplus() called before "
                                "normalize() -- fit A_demand on raw_operating_surplus "
                                "first, then call normalize(), per fit_static().")
        return self._solve_core(P, Ahat, regime, normed=True)[0]

    def mrp_A(self, P, Ahat, regime, h=1.0):
        f = self.operating_surplus
        return (f(P, Ahat + h, regime) - f(P, Ahat - h, regime)) / (2 * h)

    def mrp_P(self, P, Ahat, regime, h=1.0):
        f = self.operating_surplus
        return (f(P + h, Ahat, regime) - f(P - h, Ahat, regime)) / (2 * h)


def ppp_raw(m):
    Ah = m.Ahat_star
    return (m.raw_operating_surplus(P0, Ah, 'pre') - m.w_A * Ah) / P0


def ppp(m):
    Ah = m.Ahat_star
    return (m.operating_surplus(P0, Ah, 'pre') - m.w_A * Ah) / P0


def fit_static(name, **override):
    """Fit A_demand to hold T1 (profit per partner = PPP_TARGET), THEN
    pin the normalization anchor at that final A_demand, THEN re-verify T1
    (must hold exactly, since normalized==raw at the pre-shock point when
    x0 is the optimum for the FINAL A_demand) and report the T2 gap.

    Order matters: normalizing BEFORE A_demand is fitted pins x0 at the
    wrong (E,R) -- since the pre-shock profit-maximizing E,R depend on
    A_demand through the revenue term -- and was verified (by direct grid
    search against the exact objective) to silently produce a badly
    non-global-optimum, near-zero E0/R0 anchor once alpha_R >> alpha_A.
    That bug was caught and fixed in this rebuild; it is not present in
    the reported results below."""
    m = Sector(name, normalize=False, **override)
    target = PPP_TARGET[name]

    def f(Ad):
        m.A_demand = Ad
        return ppp_raw(m) - target
    m.A_demand = brentq(f, 1e-3, 1e9, xtol=1e-6, rtol=1e-12)
    m.normalize()
    resid = ppp(m) - target
    if abs(resid) > 0.5:
        raise RuntimeError(f"{name}: T1 broke after normalize() (resid={resid:.3f}); "
                            "normalization is not reproducing the raw nest at the pre-shock point.")
    return m


if __name__ == '__main__':
    for s in SECTORS:
        m = fit_static(s)
        t2 = m.mrp_A(P0, m.Ahat_star, 'pre') - m.w_A
        print(f"{s:8s} eta={m.eta:.4f} A_demand={m.A_demand:.4f} "
              f"PPP={ppp(m):.2f} (target {PPP_TARGET[s]:.2f}) T2 gap={t2:+.3f}  "
              f"p_promote={m.p_promote:.3f}", flush=True)
